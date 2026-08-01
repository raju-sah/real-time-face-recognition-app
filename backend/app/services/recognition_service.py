import os
import cv2
import json
import re
import secrets
import numpy as np
import gc
from datetime import datetime, timezone
import insightface
import onnxruntime
from dotenv import load_dotenv

load_dotenv()


def _patch_onnx_sessions():
    """Force single-threaded, low-memory ONNX sessions.

    insightface builds sessions with default options, which lets onnxruntime
    allocate a large memory arena per session and one thread per core. On
    Render's 512MB free tier that can exceed the memory limit, so override
    the session constructor before the model loads.
    """
    original = onnxruntime.InferenceSession.__init__

    def patched(self, path_or_bytes, sess_options=None, **kwargs):
        if sess_options is None:
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.enable_mem_pattern = False
            sess_options = opts
        return original(self, path_or_bytes, sess_options, **kwargs)

    onnxruntime.InferenceSession.__init__ = patched


_patch_onnx_sessions()


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b)
    if norm_a.size == 0 or norm_b == 0:
        return np.zeros(0)
    return dot_product / (norm_a * norm_b)


def sanitize_name(name):
    name = re.sub(r"[^A-Za-z0-9_\- ]+", "", name).strip().replace(" ", "_")
    return name[:40] if name else ""


class FaceRecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceRecognitionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print("Initializing FaceRecognitionService...")
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        self.gallery_dir = os.path.join(self.project_root, "backend", "gallery")
        os.makedirs(self.gallery_dir, exist_ok=True)

        # Keep the model pack inside the project dir so build.sh can
        # pre-download it and it ships with the deployable image.
        self.model_root = os.path.join(self.project_root, ".insightface")

        self.model = insightface.app.FaceAnalysis(
            name="buffalo_s",
            root=self.model_root,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.model.prepare(ctx_id=-1, det_size=(640, 640))

        self.SIMILARITY_THRESHOLD = 0.40
        self.EMBEDDING_SIZE = 512

        self.enroll_sessions = {}

        self._load_gallery()

        self._initialized = True
        print("FaceRecognitionService initialized successfully.")

    # ------------------------------------------------------------------ gallery

    def _load_gallery(self):
        self.people = []
        for filename in sorted(os.listdir(self.gallery_dir)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.gallery_dir, filename)
            try:
                with open(path, "r") as f:
                    person = json.load(f)
                if person.get("embeddings"):
                    self.people.append(person)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

        self._rebuild_index()
        print(
            f"Gallery loaded: {len(self.people)} people, "
            f"{len(self.gallery_embeddings) if self.gallery_embeddings.size else 0} embeddings."
        )

    def _rebuild_index(self):
        embeddings = []
        labels = []
        self.gallery_meta = []
        for idx, person in enumerate(self.people):
            for emb in person["embeddings"]:
                embeddings.append(np.asarray(emb, dtype=np.float32))
                labels.append(idx)
        if embeddings:
            self.gallery_embeddings = np.vstack(embeddings).astype(np.float32)
            self.gallery_labels = np.array(labels, dtype=int)
        else:
            self.gallery_embeddings = np.empty((0, self.EMBEDDING_SIZE), dtype=np.float32)
            self.gallery_labels = np.array([], dtype=int)

    def _persist_person(self, person):
        name = sanitize_name(person["name"]) or "user"
        path = os.path.join(self.gallery_dir, f"{name}.json")
        suffix = 2
        while os.path.exists(path):
            path = os.path.join(self.gallery_dir, f"{name}_{suffix}.json")
            suffix += 1
        with open(path, "w") as f:
            json.dump(person, f)
        person["file"] = os.path.basename(path)
        return path

    def list_users(self):
        return [
            {
                "name": p["name"],
                "user_id": p["user_id"],
                "created_at": p["created_at"],
                "embedding_count": len(p["embeddings"]),
            }
            for p in self.people
        ]

    def remove_user(self, user_id):
        for idx, person in enumerate(self.people):
            if person.get("user_id") == user_id:
                path = os.path.join(self.gallery_dir, person.get("file", f"{sanitize_name(person['name'])}.json"))
                if os.path.exists(path):
                    os.remove(path)
                self.people.pop(idx)
                self._rebuild_index()
                return True
        return False

    def auto_name(self):
        n = 0
        names = [p["name"] for p in self.people]
        names += [s["name"] for s in self.enroll_sessions.values()]
        for name in names:
            m = re.match(r"^TestUser-(\d+)$", name or "", re.IGNORECASE)
            if m:
                n = max(n, int(m.group(1)))
        return f"TestUser-{n + 1}"

    def _unique_name(self, base):
        if not any(p["name"] == base for p in self.people):
            return base
        n = 2
        while any(p["name"] == f"{base}_{n}" for p in self.people):
            n += 1
        return f"{base}_{n}"

    def _has_near_duplicate(self, embeddings, new_emb, threshold=0.98):
        if not embeddings:
            return False
        embs = np.asarray(embeddings, dtype=np.float32)
        sim = cosine_similarity(embs, new_emb)
        return bool(np.any(sim >= threshold))

    # ------------------------------------------------------------------ extract

    def _extract(self, image_rgb):
        """Detect faces + compute embeddings in one ONNX pass.

        Returns [(det, embedding)] where det keeps the old MTCNN shape:
        {"box": [x, y, w, h], "keypoints": {left_eye, right_eye, nose,
        mouth_left, mouth_right}}.
        """
        results = []
        for face in self.model.get(image_rgb):
            x1, y1, x2, y2 = face.bbox.astype(int)
            det = {
                "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "keypoints": {
                    "left_eye": face.kps[0].tolist(),
                    "right_eye": face.kps[1].tolist(),
                    "nose": face.kps[2].tolist(),
                    "mouth_left": face.kps[3].tolist(),
                    "mouth_right": face.kps[4].tolist(),
                },
            }
            results.append((det, face.normed_embedding.astype(np.float32)))
        return results

    # ------------------------------------------------------------------ pose/quality

    def _estimate_pose(self, det):
        kp = det.get("keypoints", {})
        try:
            le = kp["left_eye"]
            re_ = kp["right_eye"]
            nose = kp["nose"]
            ml = kp["mouth_left"]
            mr = kp["mouth_right"]
        except KeyError:
            return 0.0, 0.0

        eye_dist = np.hypot(re_[0] - le[0], re_[1] - le[1])
        if eye_dist < 1:
            return 0.0, 0.0

        ex = (le[0] + re_[0]) / 2.0
        ey = (le[1] + re_[1]) / 2.0
        mx = (ml[0] + mr[0]) / 2.0
        my = (ml[1] + mr[1]) / 2.0
        face_h = np.hypot(my - ey, mx - ex)
        if face_h < 1:
            face_h = eye_dist

        yaw = (nose[0] - ex) / eye_dist
        pitch = (nose[1] - ey) / face_h

        # insightface 5-point landmarks sit lower than MTCNN's, so the raw
        # pitch reads ~0.25 too high for a neutral face. Shift it back onto
        # the scale the pose thresholds below were tuned for (neutral ≈ 0.37).
        pitch -= 0.25
        return float(yaw), float(pitch)

    def _pose_matches(self, yaw, pitch, target):
        target = (target or "front").lower()
        if target == "front":
            return abs(yaw) <= 0.30 and 0.20 <= pitch <= 0.54
        if target == "left":
            return abs(yaw) >= 0.30 or yaw >= 0.30 or yaw <= -0.30
        if target == "right":
            return abs(yaw) >= 0.30 or yaw <= -0.30 or yaw >= 0.30
        if target == "up":
            return pitch <= 0.26
        if target == "down":
            return pitch >= 0.48
        return True

    def _get_pose_guidance(self, yaw, pitch, target):
        target = (target or "front").lower()
        matched = self._pose_matches(yaw, pitch, target)
        dirs = {"left": False, "right": False, "up": False, "down": False}
        
        if matched:
            return {
                "guidance": "Perfect! Hold pose still...",
                "matched": True,
                "directions": dirs,
            }

        if target == "front":
            if yaw > 0.30:
                dirs["left"] = True
                msg = "Turn head slightly to your LEFT"
            elif yaw < -0.30:
                dirs["right"] = True
                msg = "Turn head slightly to your RIGHT"
            elif pitch < 0.20:
                dirs["down"] = True
                msg = "Tilt head slightly DOWN"
            elif pitch > 0.54:
                dirs["up"] = True
                msg = "Tilt head slightly UP"
            else:
                msg = "Look straight ahead at the camera"
        elif target == "left":
            dirs["left"] = True
            msg = "Turn your head to your LEFT"
        elif target == "right":
            dirs["right"] = True
            msg = "Turn your head to your RIGHT"
        elif target == "up":
            dirs["up"] = True
            msg = "Tilt your chin UP towards the ceiling"
        elif target == "down":
            dirs["down"] = True
            msg = "Tilt your head DOWN slightly"
        else:
            msg = "Adjust head position to target pose"

        return {
            "guidance": msg,
            "matched": False,
            "directions": dirs,
        }

    def _quality_check(self, face_bgr, box):
        x, y, w, h = box
        issues = []
        if w < 100:
            issues.append("face too small — move closer")
        if face_bgr.size == 0:
            issues.append("empty face crop")

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < 15:
            issues.append("image too blurry")

        brightness = gray.mean()
        if brightness < 15:
            issues.append("camera appears dark — check lighting")
        elif brightness > 225:
            issues.append("image too bright")

        return issues, float(sharpness)

    # ------------------------------------------------------------------ enroll

    def start_enroll(self, name="", existing_user_id=""):
        if existing_user_id:
            person = next((p for p in self.people if p["user_id"] == existing_user_id), None)
            if person is None:
                return {"status": "error", "message": "Existing user not found"}
            user_id = secrets.token_hex(8)
            self.enroll_sessions[user_id] = {
                "name": person["name"],
                "existing_user_id": person["user_id"],
                "embeddings": [],
                "poses": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            return {"status": "ok", "user_id": user_id, "name": person["name"], "existing": True}

        clean = sanitize_name(name)
        if clean:
            existing_person = next(
                (p for p in self.people if p["name"].lower() == clean.lower()), None
            )
            if existing_person:
                return {
                    "status": "already_exists",
                    "existing_name": existing_person["name"],
                    "existing_user_id": existing_person["user_id"],
                    "reason": "name",
                    "message": f"Profile '{existing_person['name']}' already exists. Please delete it before enrolling again.",
                }

        user_id = secrets.token_hex(8)
        final_name = clean or self.auto_name()
        self.enroll_sessions[user_id] = {
            "name": final_name,
            "existing_user_id": None,
            "embeddings": [],
            "poses": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return {"status": "ok", "user_id": user_id, "name": final_name, "existing": False}

    def enroll_sample(self, image_path, user_id, target_pose, force=False):
        session = self.enroll_sessions.get(user_id)
        if session is None:
            return {"status": "error", "message": "Unknown enrollment session"}

        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"status": "error", "message": "Could not read image"}

            h, w = image.shape[:2]
            if max(h, w) > 1280:
                scale = 1280 / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self._extract(image_rgb)
            if not detections:
                return {
                    "status": "no_face",
                    "message": "No face detected in camera view",
                    "guidance": {"guidance": "Position your face inside the target area", "matched": False, "directions": {}},
                }

            det, embedding = detections[0]
            yaw, pitch = self._estimate_pose(det)
            x, y, width, height = det["box"]
            x, y = max(0, x), max(0, y)
            face_bgr = image[y : y + height, x : x + width]

            guidance = self._get_pose_guidance(yaw, pitch, target_pose)
            issues, sharpness = self._quality_check(face_bgr, (x, y, width, height))

            if issues and not force:
                return {
                    "status": "poor_quality",
                    "message": ", ".join(issues),
                    "pose": {"yaw": yaw, "pitch": pitch},
                    "guidance": guidance,
                    "quality": {"sharpness": round(sharpness, 1)},
                }

            if not guidance["matched"] and not force:
                return {
                    "status": "waiting",
                    "message": guidance["guidance"],
                    "pose": {"yaw": yaw, "pitch": pitch},
                    "guidance": guidance,
                }

            embedding = embedding.astype(float)

            # Duplicate face check against existing gallery profiles (only for new enrollment, not existing_user_id session)
            if not session.get("existing_user_id") and len(session["embeddings"]) == 0 and self.gallery_embeddings.shape[0] > 0 and not force:
                similarities = cosine_similarity(self.gallery_embeddings, embedding)
                if similarities.size > 0:
                    max_idx = int(np.argmax(similarities))
                    max_sim = float(similarities[max_idx])
                    if max_sim >= self.SIMILARITY_THRESHOLD:
                        matched_person = self.people[self.gallery_labels[max_idx]]
                        return {
                            "status": "already_exists",
                            "existing_name": matched_person["name"],
                            "existing_user_id": matched_person["user_id"],
                            "reason": "face",
                            "confidence": round(max_sim, 2),
                            "message": f"This face is already enrolled under profile '{matched_person['name']}' ({int(max_sim * 100)}% match). Delete existing profile to re-enroll.",
                            "pose": {"yaw": yaw, "pitch": pitch},
                            "guidance": guidance,
                        }

            session["embeddings"].append(embedding.tolist())
            session["poses"].append(target_pose)

            del image, image_rgb, face_bgr
            gc.collect()

            return {
                "status": "captured",
                "message": "Sample captured successfully!",
                "pose": {"yaw": yaw, "pitch": pitch},
                "guidance": {"guidance": "Captured!", "matched": True, "directions": {}},
                "captured_count": len(session["embeddings"]),
                "poses": session["poses"],
            }

        except Exception as e:
            print(f"Error in enroll_sample: {e}")
            return {"status": "error", "message": str(e)}

    def complete_enroll(self, user_id, name=""):
        session = self.enroll_sessions.pop(user_id, None)
        if session is None:
            return None

        if session.get("existing_user_id"):
            person = next((p for p in self.people if p["user_id"] == session["existing_user_id"]), None)
            if person is None:
                return None

            added = 0
            for emb, pose in zip(session["embeddings"], session["poses"]):
                emb_arr = np.asarray(emb, dtype=np.float32)
                if self._has_near_duplicate(person["embeddings"], emb_arr):
                    continue
                person["embeddings"].append(emb_arr.tolist())
                person["poses"].append(pose)
                added += 1

            filename = person.get("file", f"{sanitize_name(person['name'])}.json")
            path = os.path.join(self.gallery_dir, filename)
            with open(path, "w") as f:
                json.dump(person, f)
            self._rebuild_index()

            return {
                "name": person["name"],
                "user_id": person["user_id"],
                "embedding_count": len(person["embeddings"]),
                "added_count": added,
                "existing": True,
            }

        final_name = self._unique_name(sanitize_name(name) or session["name"] or self.auto_name())

        if not session["embeddings"]:
            return {
                "name": final_name,
                "user_id": user_id,
                "embedding_count": 0,
                "added_count": 0,
                "existing": False,
                "skipped": True,
            }

        person = {
            "name": final_name,
            "user_id": user_id,
            "created_at": session["created_at"],
            "poses": session["poses"],
            "embeddings": session["embeddings"],
        }

        self._persist_person(person)
        self.people.append(person)
        self._rebuild_index()

        return {
            "name": final_name,
            "user_id": user_id,
            "embedding_count": len(session["embeddings"]),
            "added_count": len(session["embeddings"]),
            "existing": False,
        }

    def abort_enroll(self, user_id):
        return bool(self.enroll_sessions.pop(user_id, None))

    # ------------------------------------------------------------------ recognize

    def recognize(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not read image"}

            h, w = image.shape[:2]
            if max(h, w) > 1280:
                scale = 1280 / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self._extract(image_rgb)
            if not detections:
                return {"faces": [], "error": "No face detected"}

            has_gallery = self.gallery_embeddings.shape[0] > 0
            faces = []

            for det, embedding in detections:
                x, y, width, height = det["box"]
                x, y = max(0, x), max(0, y)
                face_image = image[y : y + height, x : x + width]
                if face_image.size == 0:
                    continue

                person_name = "Unknown"
                confidence = 0.0

                if has_gallery:
                    similarities = cosine_similarity(self.gallery_embeddings, embedding)
                    if similarities.size > 0:
                        max_idx = int(np.argmax(similarities))
                        max_similarity = float(similarities[max_idx])
                        confidence = max_similarity
                        if max_similarity >= self.SIMILARITY_THRESHOLD:
                            person = self.people[self.gallery_labels[max_idx]]
                            person_name = person["name"]

                faces.append({
                    "person": person_name,
                    "confidence": confidence,
                    "box": [int(x), int(y), int(width), int(height)],
                    "success": person_name != "Unknown",
                })

            del image, image_rgb
            gc.collect()

            return {"faces": faces}

        except Exception as e:
            print(f"Error in recognition service: {e}")
            return {"error": str(e)}


recognition_service = FaceRecognitionService()
