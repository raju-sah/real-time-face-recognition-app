from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import base64
import secrets
import shutil
from dotenv import load_dotenv

load_dotenv()

from backend.app.services.recognition_service import recognition_service

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR_RELATIVE = os.getenv("UPLOAD_DIR", "backend/uploads")
UPLOAD_DIR = os.path.join(ROOT_DIR, UPLOAD_DIR_RELATIVE)
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_temp(contents: bytes, filename: str = "upload.jpg") -> str:
    ext = os.path.splitext(filename)[1] or ".jpg"
    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)
    with open(temp_file_path, "wb") as buffer:
        buffer.write(contents)
    return temp_file_path


def _cleanup(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _get_session_id(x_session_id: str = Header(None)) -> str:
    return x_session_id or secrets.token_hex(8)


@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}


@app.get("/users")
async def list_users(session_id: str = Header(None)):
    sid = _get_session_id(session_id)
    return {"users": recognition_service.list_users(sid)}


@app.delete("/users/{user_id}")
async def delete_user(user_id: str, session_id: str = Header(None)):
    sid = _get_session_id(session_id)
    removed = recognition_service.remove_user(sid, user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@app.post("/enroll/start")
async def enroll_start(
    name: str = Form(""),
    existing_user_id: str = Form(""),
    session_id: str = Header(None),
):
    sid = _get_session_id(session_id)
    result = recognition_service.start_enroll(sid, name, existing_user_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message", "User not found"))
    return result


@app.post("/enroll/sample")
async def enroll_sample(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    target_pose: str = Form("front"),
    force: bool = Form(False),
    session_id: str = Header(None),
):
    sid = _get_session_id(session_id)
    temp_file_path = _save_temp(await file.read(), file.filename or "sample.jpg")
    try:
        result = recognition_service.enroll_sample(sid, temp_file_path, user_id, target_pose, force=force)
    finally:
        _cleanup(temp_file_path)
    return result


@app.post("/enroll/complete")
async def enroll_complete(payload: dict, session_id: str = Header(None)):
    sid = _get_session_id(session_id)
    user_id = payload.get("user_id")
    name = payload.get("name", "")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    result = recognition_service.complete_enroll(sid, user_id, name)
    if result is None:
        raise HTTPException(status_code=404, detail="Enrollment session not found")
    return result


@app.post("/enroll/abort")
async def enroll_abort(payload: dict, session_id: str = Header(None)):
    sid = _get_session_id(session_id)
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    return {"ok": recognition_service.abort_enroll(sid, user_id)}


@app.post("/recognize")
async def recognize_image(file: UploadFile = File(...), session_id: str = Header(None)):
    sid = _get_session_id(session_id)
    temp_file_path = _save_temp(await file.read(), file.filename or "upload.jpg")
    try:
        return _format_recognize_result(recognition_service.recognize(sid, temp_file_path))
    finally:
        _cleanup(temp_file_path)


@app.post("/recognize/base64")
async def recognize_base64(payload: dict, session_id: str = Header(None)):
    sid = _get_session_id(session_id)
    image_base64 = payload.get("image_base64")
    if not image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        contents = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    temp_file_path = _save_temp(contents, "frame.jpg")
    try:
        return _format_recognize_result(recognition_service.recognize(sid, temp_file_path))
    finally:
        _cleanup(temp_file_path)


def _format_recognize_result(result):
    if "error" in result and not result.get("faces"):
        return {"success": False, "faces": [], "error": result["error"]}
    faces = []
    for f in result.get("faces", []):
        faces.append({
            "prediction": f["person"],
            "confidence": f["confidence"],
            "box": f["box"],
            "success": f["person"] != "Unknown",
        })
    return {"success": any(f["success"] for f in faces), "faces": faces}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run(app, host=host, port=port)
