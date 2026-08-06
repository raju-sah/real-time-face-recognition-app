import { useState, ChangeEvent, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  User,
  ShieldCheck,
  AlertCircle,
  AlertTriangle,
  Loader2,
  Camera,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Cpu,
  Zap,
  Video,
  Play,
  StopCircle,
  Users,
  Trash2,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  X,
  Image as ImageIcon,
  Mail,
  Link as LinkIcon,
  GitBranch,
  AtSign,
  Globe,
  UserPlus,
  Eye,
  Scan,
  Sparkles,
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface FaceResult {
  success: boolean;
  prediction?: string;
  confidence?: number;
  box?: number[];
}

interface RecognitionResult {
  success: boolean;
  faces: FaceResult[];
  error?: string;
}

interface EnrolledUser {
  name: string;
  user_id: string;
  created_at: string;
  embedding_count: number;
}

const POSE_SEQUENCE = [
  { key: 'front', label: 'Look straight' },
  { key: 'left', label: 'Turn left' },
  { key: 'right', label: 'Turn right' },
  { key: 'up', label: 'Tilt up' },
  { key: 'down', label: 'Look down' },
  { key: 'front', label: 'Face forward' },
];

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const api = axios.create({ baseURL: API_URL, timeout: 30000 });

type RecSubMode = 'camera' | 'upload';

interface EnrollUiState {
  seqIdx: number;
  captured: string[];
  status: string;
  pose: { yaw?: number; pitch?: number } | null;
  guidance: {
    guidance: string;
    matched: boolean;
    directions: { left?: boolean; right?: boolean; up?: boolean; down?: boolean };
  } | null;
  done: boolean;
  finalName: string | null;
  existing: boolean;
  manualCapturing: boolean;
  existingDuplicate?: {
    name: string;
    user_id: string;
    reason: 'name' | 'face';
    message?: string;
  } | null;
}

const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -12 },
};

const staggerChildren = {
  animate: {
    transition: { staggerChildren: 0.06, delayChildren: 0.05 },
  },
};

const fadeInUp = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
};

const scaleIn = {
  initial: { opacity: 0, scale: 0.92 },
  animate: { opacity: 1, scale: 1, transition: { duration: 0.35, ease: [0.16, 1, 0.3, 1] } },
};

const slideInRight = {
  initial: { opacity: 0, x: 60 },
  animate: { opacity: 1, x: 0, transition: { duration: 0.35, ease: [0.16, 1, 0.3, 1] } },
  exit: { opacity: 0, x: 60, transition: { duration: 0.25 } },
};

function App() {
  const [showProfilesDrawer, setShowProfilesDrawer] = useState(false);
  const [isEnrolling, setIsEnrolling] = useState(false);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const setActiveStream = (s: MediaStream | null) => {
    streamRef.current = s;
    setStream(s);
  };

  const stopCamera = useCallback(() => {
    const s = streamRef.current;
    if (s) {
      s.getTracks().forEach((t) => t.stop());
    }
    streamRef.current = null;
    setStream(null);
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const busyRef = useRef(false);
  const enrollRef = useRef<{
    running: boolean;
    userId: string;
    seqIdx: number;
    captured: string[];
    interval: ReturnType<typeof setInterval> | null;
  } | null>(null);

  const recRef = useRef<{
    running: boolean;
    interval: ReturnType<typeof setInterval> | null;
    lastResult: RecognitionResult | null;
  } | null>(null);
  const rafRef = useRef<number | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<RecognitionResult | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  const [enrollName, setEnrollName] = useState('');
  const [existingUserId, setExistingUserId] = useState('');
  const [enrollStarting, setEnrollStarting] = useState(false);
  const [serverWakeSeconds, setServerWakeSeconds] = useState(0);
  const [enrollUi, setEnrollUi] = useState<EnrollUiState>({
    seqIdx: 0,
    captured: [],
    status: '',
    pose: null,
    guidance: null,
    done: false,
    finalName: null,
    existing: false,
    manualCapturing: false,
  });

  const [recResult, setRecResult] = useState<RecognitionResult | null>(null);
  const [recRunning, setRecRunning] = useState(false);
  const [recSubMode, setRecSubMode] = useState<RecSubMode>('camera');

  const [users, setUsers] = useState<EnrolledUser[]>([]);
  const [usersLoading, setUsersLoading] = useState(false);

  const startCamera = async (): Promise<boolean> => {
    stopCamera();
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      setActiveStream(mediaStream);
      const deadline = Date.now() + 4000;
      while (Date.now() < deadline) {
        const video = videoRef.current;
        if (video && video.srcObject && video.videoWidth > 0 && video.readyState >= 1) return true;
        await new Promise<void>((r) => setTimeout(r, 80));
      }
      return true;
    } catch (err) {
      console.error('Camera error:', err);
      setError('Camera access denied. Check browser permissions.');
      return false;
    }
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) return;
    video.srcObject = stream;
    video.play().catch(() => {});
  }, [stream]);

  const captureFrame = (): string | null => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || video.readyState < 1) return null;
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas');
    }
    const canvas = captureCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    if (!dataUrl || dataUrl === 'data:,' || dataUrl === 'data:,') return null;
    return dataUrl;
  };

  const drawOverlay = (result: RecognitionResult | null) => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !video.videoWidth || video.readyState < 1) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (result && result.faces) {
      for (const face of result.faces) {
        if (!face.box || !face.prediction) continue;
        const [x, y, w, h] = face.box;
        const color = face.success ? '#34d399' : '#f87171';

        ctx.shadowColor = color;
        ctx.shadowBlur = 15;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.shadowBlur = 0;

        ctx.save();
        const label = `${face.prediction} ${((face.confidence ?? 0) * 100).toFixed(0)}%`;
        ctx.font = 'bold 18px "DM Sans", sans-serif';
        const tw = ctx.measureText(label).width;

        const labelY = Math.max(35, y - 8);

        ctx.translate(x + w, labelY);
        ctx.scale(-1, 1);

        const paddingX = 12;
        const badgeHeight = 30;
        const badgeWidth = tw + paddingX * 2;

        ctx.fillStyle = 'rgba(12, 10, 9, 0.9)';
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(0, -badgeHeight, badgeWidth, badgeHeight, 6);
        } else {
          ctx.rect(0, -badgeHeight, badgeWidth, badgeHeight);
        }
        ctx.fill();

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(2, -badgeHeight + 4);
        ctx.lineTo(2, -4);
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, paddingX + 2, -badgeHeight / 2);

        ctx.restore();
      }
    }
  };

  const renderFrame = (result: RecognitionResult | null): string | null => {
    const dataUrl = captureFrame();
    drawOverlay(result);
    return dataUrl;
  };

  const fetchUsers = async () => {
    setUsersLoading(true);
    try {
      const resp = await api.get('/users');
      setUsers(resp.data.users ?? []);
    } catch {
      setUsers([]);
    } finally {
      setUsersLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const resetEnrollState = () => {
    setEnrollUi({
      seqIdx: 0,
      captured: [],
      status: '',
      pose: null,
      guidance: null,
      done: false,
      finalName: null,
      existing: false,
      manualCapturing: false,
      existingDuplicate: null,
    });
    setEnrollName('');
    setExistingUserId('');
  };

  const handleStartEnrolling = () => {
    stopRecognition();
    resetEnrollState();
    setIsEnrolling(true);
    setEnrollmentComplete(false);
  };

  const handleCancelEnrolling = () => {
    abortEnroll();
    setIsEnrolling(false);
  };

  const startEnroll = async () => {
    setError(null);
    setEnrollStarting(true);
    setServerWakeSeconds(0);
    const wakeTimer = setInterval(() => {
      setServerWakeSeconds((s) => s + 1);
    }, 1000);

    try {
      const form = new URLSearchParams();
      form.append('name', enrollName);
      if (existingUserId) form.append('existing_user_id', existingUserId);
      const resp = await api.post('/enroll/start', form, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      });
      const data = resp.data;

      if (data.status === 'already_exists') {
        setEnrollUi({
          seqIdx: 0,
          captured: [],
          status: data.message || `Profile '${data.existing_name}' already exists.`,
          pose: null,
          guidance: null,
          done: false,
          finalName: null,
          existing: false,
          manualCapturing: false,
          existingDuplicate: {
            name: data.existing_name,
            user_id: data.existing_user_id,
            reason: data.reason || 'name',
            message: data.message,
          },
        });
        return;
      }

      const ok = await startCamera();
      if (!ok) return;

      enrollRef.current = {
        running: true,
        userId: data.user_id,
        seqIdx: 0,
        captured: [],
        interval: null,
      };
      setEnrollUi({
        seqIdx: 0,
        captured: [],
        status: POSE_SEQUENCE[0].label,
        pose: null,
        guidance: null,
        done: false,
        finalName: null,
        existing: data.existing ?? false,
        manualCapturing: false,
        existingDuplicate: null,
      });
      enrollRef.current.interval = setInterval(() => runEnrollFrame(false), 550);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Enrollment failed to start';
      setError(msg);
      stopCamera();
    } finally {
      clearInterval(wakeTimer);
      setEnrollStarting(false);
    }
  };

  const runEnrollFrame = async (force = false) => {
    const e = enrollRef.current;
    if (!e || !e.running) return;
    if (busyRef.current && !force) return;
    busyRef.current = true;
    if (force) {
      setEnrollUi((prev) => ({ ...prev, manualCapturing: true }));
    }
    try {
      const dataUrl = renderFrame(null);
      if (!dataUrl) {
        setEnrollUi((prev) => ({ ...prev, status: 'Waiting for camera...', manualCapturing: false }));
        return;
      }
      const fetched = await fetch(dataUrl);
      const rawBlob = await fetched.blob();
      if (rawBlob.size === 0) {
        setEnrollUi((prev) => ({ ...prev, status: 'Waiting for camera...', manualCapturing: false }));
        return;
      }
      const blob = new Blob([rawBlob], { type: 'image/jpeg' });
      const fd = new FormData();
      fd.append('file', blob, 'frame.jpg');
      fd.append('user_id', e.userId);
      fd.append('target_pose', POSE_SEQUENCE[e.seqIdx].key);
      if (force) {
        fd.append('force', 'true');
      }

      const resp = await api.post('/enroll/sample', fd);
      const data = resp.data;

      if (data.status === 'already_exists') {
        if (e) {
          e.running = false;
          if (e.interval) clearInterval(e.interval);
        }
        stopCamera();
        setEnrollUi((prev) => ({
          ...prev,
          status: data.message || `Profile '${data.existing_name}' already exists.`,
          manualCapturing: false,
          existingDuplicate: {
            name: data.existing_name,
            user_id: data.existing_user_id,
            reason: data.reason || 'face',
            message: data.message,
          },
        }));
        return;
      }

      if (data.status === 'captured') {
        e.captured.push(POSE_SEQUENCE[e.seqIdx].key);
        if (e.seqIdx < POSE_SEQUENCE.length - 1) {
          e.seqIdx += 1;
          setEnrollUi((prev) => ({
            ...prev,
            seqIdx: e.seqIdx,
            captured: [...e.captured],
            status: POSE_SEQUENCE[e.seqIdx].label,
            pose: data.pose ?? null,
            guidance: data.guidance ?? null,
            done: false,
            finalName: null,
            manualCapturing: false,
          }));
        } else {
          setEnrollUi((prev) => ({
            ...prev,
            captured: [...e.captured],
            status: 'Saving profile...',
            pose: data.pose ?? null,
            guidance: data.guidance ?? null,
            manualCapturing: false,
          }));
          await completeEnroll();
        }
      } else {
        setEnrollUi((prev) => ({
          ...prev,
          status: data.message || 'Adjust position',
          pose: data.pose ?? null,
          guidance: data.guidance ?? null,
          manualCapturing: false,
        }));
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Request failed';
      setEnrollUi((prev) => ({ ...prev, status: msg, manualCapturing: false }));
    } finally {
      busyRef.current = false;
    }
  };

  const captureManually = () => {
    runEnrollFrame(true);
  };

  const completeEnroll = async () => {
    const e = enrollRef.current;
    if (!e) return;
    e.running = false;
    if (e.interval) clearInterval(e.interval);
    try {
      const resp = await api.post('/enroll/complete', { user_id: e.userId, name: enrollName });
      const data = resp.data;
      setEnrollUi((prev) => ({
        ...prev,
        done: true,
        existing: data.existing ?? false,
        finalName: data.name,
        status: data.existing
          ? `Added ${data.added_count} sample${data.added_count === 1 ? '' : 's'} to ${data.name} — ${data.embedding_count} total`
          : `Saved with ${data.embedding_count} samples`,
      }));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to save';
      setEnrollUi((prev) => ({ ...prev, status: msg }));
    } finally {
      stopCamera();
      fetchUsers();
    }
  };

  const abortEnroll = async () => {
    const e = enrollRef.current;
    if (e) {
      e.running = false;
      if (e.interval) clearInterval(e.interval);
      try {
        await api.post('/enroll/abort', { user_id: e.userId });
      } catch {
        /* ignore */
      }
    }
    stopCamera();
    setEnrollUi({
      seqIdx: 0,
      captured: [],
      status: '',
      pose: null,
      guidance: null,
      done: false,
      finalName: null,
      existing: false,
      manualCapturing: false,
      existingDuplicate: null,
    });
  };

  const startRecognition = async () => {
    setError(null);
    const ok = await startCamera();
    if (!ok) return;
    setRecResult(null);
    recRef.current = { running: true, interval: null, lastResult: null };
    setRecRunning(true);
    recRef.current.interval = setInterval(runRecFrame, 250);

    const animateOverlay = () => {
      const r = recRef.current;
      if (!r || !r.running) return;
      drawOverlay(r.lastResult);
      rafRef.current = requestAnimationFrame(animateOverlay);
    };
    rafRef.current = requestAnimationFrame(animateOverlay);
  };

  const runRecFrame = async () => {
    const r = recRef.current;
    if (!r || !r.running || busyRef.current) return;
    busyRef.current = true;
    try {
      const dataUrl = captureFrame();
      if (!dataUrl) return;
      const resp = await api.post('/recognize/base64', { image_base64: dataUrl });
      r.lastResult = resp.data;
      setRecResult(resp.data);
    } catch {
      r.lastResult = { success: false, faces: [], error: 'request_failed' };
      setRecResult(r.lastResult);
    } finally {
      busyRef.current = false;
    }
  };

  const stopRecognition = () => {
    const r = recRef.current;
    if (r) {
      r.running = false;
      if (r.interval) clearInterval(r.interval);
    }
    recRef.current = null;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setRecRunning(false);
    stopCamera();
    setRecResult(null);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setUploadResult(null);
    setError(null);
  };

  const uploadAndRecognize = async () => {
    if (!selectedFile) return;
    setUploadLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append('file', selectedFile);
      const resp = await api.post('/recognize', fd);
      setUploadResult(resp.data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed';
      setError(msg);
    } finally {
      setUploadLoading(false);
    }
  };

  const hasUsers = users.length > 0;
  const [enrollmentComplete, setEnrollmentComplete] = useState(false);

  // Auto-transition to recognition after enrollment completes
  useEffect(() => {
    if (enrollUi.done && hasUsers) {
      setEnrollmentComplete(true);
    }
  }, [enrollUi.done, hasUsers]);

  const showEnrollForm = !hasUsers || isEnrolling;
  const showRecognize = (hasUsers && !isEnrolling) || enrollmentComplete;

  const buttonHover = { scale: 1.02, transition: { duration: 0.15 } };
  const buttonTap = { scale: 0.97, transition: { duration: 0.1 } };

  return (
    <div className="relative min-h-screen overflow-hidden bg-base-950">
      {showProfilesDrawer && (
        <motion.div
          className="fixed inset-0 z-50 flex justify-end bg-black/80 backdrop-blur-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onClick={() => setShowProfilesDrawer(false)}
        >
          <motion.div
            className="w-full max-w-md bg-base-900/95 backdrop-blur-xl border-l border-base-700/40 p-6 flex flex-col h-full shadow-2xl"
            initial={{ x: 60 }}
            animate={{ x: 0 }}
            exit={{ x: 60 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-4 border-b border-base-700/40">
              <div className="flex items-center gap-3">
                <Users className="h-5 w-5 text-accent-400" />
                <h2 className="font-display font-semibold text-base-50">Profiles</h2>
                <span className="rounded-full bg-accent-400/10 border border-accent-400/20 px-2.5 py-0.5 text-xs font-bold text-accent-400">
                  {users.length}
                </span>
              </div>
              <motion.button
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowProfilesDrawer(false)}
                className="rounded-lg p-2 text-base-400 hover:text-base-50 hover:bg-base-700/50 transition-colors cursor-pointer"
              >
                <X className="h-5 w-5" />
              </motion.button>
            </div>

            <div className="flex-1 overflow-y-auto py-4 space-y-2">
              {usersLoading ? (
                <div className="flex items-center justify-center py-12 text-sm text-base-400 gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-accent-400" /> Loading...
                </div>
              ) : users.length === 0 ? (
                <div className="py-12 text-center text-base-500 text-sm">
                  <User className="h-10 w-10 mx-auto mb-2 opacity-20" />
                  No profiles yet
                </div>
              ) : (
                users.map((u) => (
                  <motion.div
                    key={u.user_id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className="flex items-center justify-between rounded-xl border border-base-700/40 bg-base-950/60 p-3 hover:border-accent-400/20 hover:bg-accent-400/[0.02] transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-accent-400/20 to-mint-400/20 text-accent-400 border border-accent-400/20 font-display font-bold text-sm">
                        {u.name.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <p className="font-medium text-sm text-base-50">{u.name}</p>
                        <p className="text-xs text-base-400">
                          {u.embedding_count} samples · {u.created_at.slice(0, 10)}
                        </p>
                      </div>
                    </div>

                    <motion.button
                      whileTap={{ scale: 0.9 }}
                      onClick={async () => {
                        try {
                          await api.delete(`/users/${encodeURIComponent(u.user_id)}`);
                          fetchUsers();
                        } catch {
                          /* ignore */
                        }
                      }}
                      className="rounded-lg p-2 text-base-400 hover:bg-red-500/10 hover:text-red-400 transition-colors cursor-pointer"
                      title="Delete profile"
                    >
                      <Trash2 className="h-4 w-4" />
                    </motion.button>
                  </motion.div>
                ))
              )}
            </div>
          </motion.div>
        </motion.div>
      )}

      <div className="relative z-10 mx-auto max-w-4xl px-4 py-6 sm:py-8">
        <motion.header
          className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between pb-6 border-b border-base-700/40"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-accent-400/20 to-mint-400/20 border border-accent-400/20">
              <Scan className="h-5 w-5 text-accent-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-display font-bold text-base-50 tracking-tight">NeuroVision</h1>
                <span className="inline-flex items-center gap-1.5 rounded-full bg-mint-400/10 border border-mint-400/20 px-2 py-0.5 text-[10px] font-semibold text-mint-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-mint-400" /> Online
                </span>
              </div>
              <p className="text-xs text-base-400 font-display">Identity Recognition System</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowProfilesDrawer(true)}
              className="flex items-center gap-2 rounded-xl bg-base-800/60 border border-base-700/40 px-3 py-2 text-xs font-medium text-base-300 hover:border-accent-400/30 hover:text-base-50 transition-all cursor-pointer"
            >
              <Users className="h-4 w-4 text-accent-400" />
              <span>{users.length}</span>
            </motion.button>
          </div>
        </motion.header>

        <AnimatePresence mode="wait">
          {error && (
            <motion.div
              className="mb-6 flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs font-medium text-red-300"
              initial={{ opacity: 0, y: -8, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -8, height: 0 }}
              transition={{ duration: 0.2 }}
            >
              <AlertCircle className="h-4 w-4 shrink-0" /> {error}
              <motion.button
                whileTap={{ scale: 0.9 }}
                className="ml-auto p-1 hover:bg-red-500/20 rounded-lg transition-colors"
                onClick={() => setError(null)}
              >
                <XCircle className="h-4 w-4" />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {showEnrollForm && (
            <>
              {isEnrolling && hasUsers && (
                <motion.div
                  className="mb-6 flex items-center justify-between"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-400/10 border border-accent-400/20">
                      <UserPlus className="h-5 w-5 text-accent-400" />
                    </div>
                    <div>
                      <h2 className="text-base font-display font-semibold text-base-50">Enroll New Identity</h2>
                      <p className="text-xs text-base-400">Add a face to recognize</p>
                    </div>
                  </div>
                  <motion.button
                    whileTap={{ scale: 0.95 }}
                    onClick={handleCancelEnrolling}
                    className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-base-400 hover:text-base-50 hover:bg-base-700/50 transition-colors cursor-pointer"
                  >
                    <X className="h-3.5 w-3.5" /> Cancel
                  </motion.button>
                </motion.div>
              )}

              {!hasUsers && !isEnrolling && !stream && !enrollUi.done && (
                <motion.div
                  className="mb-10 space-y-8"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
                >

                  <div className="flex flex-col lg:flex-row gap-6 max-w-4xl mx-auto">
                    {/* Futuristic SVG Visualization Card */}
                    <motion.div
                      className="flex-1 glass-card p-8 flex flex-col items-center justify-center min-h-[380px]"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: 0.2 }}
                    >
                      <svg viewBox="0 0 200 200" className="w-full max-w-[240px]" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                          <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.6" />
                            <stop offset="100%" stopColor="#34d399" stopOpacity="0.6" />
                          </linearGradient>
                          <radialGradient id="coreGrad" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.3" />
                            <stop offset="60%" stopColor="#34d399" stopOpacity="0.1" />
                            <stop offset="100%" stopColor="transparent" />
                          </radialGradient>
                          <filter id="glow">
                            <feGaussianBlur stdDeviation="3" result="blur" />
                            <feMerge>
                              <feMergeNode in="blur" />
                              <feMergeNode in="SourceGraphic" />
                            </feMerge>
                          </filter>
                        </defs>

                        <circle cx="100" cy="100" r="80" fill="none" stroke="url(#ringGrad)" stroke-width="1" opacity="0.5">
                          <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="20s" repeatCount="indefinite" />
                        </circle>

                        <circle cx="100" cy="100" r="65" fill="none" stroke="url(#ringGrad)" stroke-width="0.5" opacity="0.3">
                          <animateTransform attributeName="transform" type="rotate" from="360 100 100" to="0 100 100" dur="15s" repeatCount="indefinite" />
                        </circle>

                        <ellipse cx="100" cy="100" rx="30" ry="38" fill="none" stroke="#fbbf24" stroke-width="1.5" opacity="0.7" filter="url(#glow)">
                          <animate attributeName="ry" values="38;40;38" dur="3s" repeatCount="indefinite" />
                          <animate attributeName="rx" values="30;32;30" dur="3s" repeatCount="indefinite" />
                        </ellipse>

                        <circle cx="88" cy="92" r="4" fill="#34d399" opacity="0.8" filter="url(#glow)">
                          <animate attributeName="opacity" values="0.8;0.3;0.8" dur="2s" repeatCount="indefinite" />
                        </circle>
                        <circle cx="112" cy="92" r="4" fill="#34d399" opacity="0.8" filter="url(#glow)">
                          <animate attributeName="opacity" values="0.8;0.3;0.8" dur="2s" repeatCount="indefinite" begin="0.3s" />
                        </circle>

                        <line x1="70" y1="100" x2="130" y2="100" stroke="#fbbf24" stroke-width="1" opacity="0">
                          <animate attributeName="opacity" values="0;0.8;0" dur="2.5s" repeatCount="indefinite" />
                          <animate attributeName="y1" values="70;130;70" dur="2.5s" repeatCount="indefinite" />
                          <animate attributeName="y2" values="70;130;70" dur="2.5s" repeatCount="indefinite" />
                        </line>

                        <circle r="2" fill="#fbbf24" opacity="0.6">
                          <animateMotion dur="8s" repeatCount="indefinite" path="M100,100 m-50,0 a50,50 0 1,1 100,0 a50,50 0 1,1 -100,0" />
                        </circle>
                        <circle r="1.5" fill="#34d399" opacity="0.4">
                          <animateMotion dur="12s" repeatCount="indefinite" path="M100,100 m-60,0 a60,60 0 1,0 120,0 a60,60 0 1,0 -120,0" />
                        </circle>

                        <line x1="100" y1="60" x2="100" y2="40" stroke="#fbbf24" stroke-width="0.5" opacity="0.4">
                          <animate attributeName="opacity" values="0.4;0.1;0.4" dur="3s" repeatCount="indefinite" />
                        </line>
                        <line x1="100" y1="140" x2="100" y2="160" stroke="#34d399" stroke-width="0.5" opacity="0.4">
                          <animate attributeName="opacity" values="0.4;0.1;0.4" dur="3s" repeatCount="indefinite" begin="1.5s" />
                        </line>

                        <circle cx="100" cy="100" r="8" fill="url(#coreGrad)" stroke="#fbbf24" stroke-width="0.5" opacity="0.6">
                          <animate attributeName="r" values="8;10;8" dur="2s" repeatCount="indefinite" />
                        </circle>
                      </svg>
                      <p className="text-sm text-base-300 mt-6 text-center font-medium">Real-time neural mapping</p>
                      <p className="text-xs text-base-500 mt-2 text-center">Watch your identity come to life</p>
                    </motion.div>

                    {/* Form Card */}
                    <motion.div
                      className="flex-1 glass-card p-8 flex flex-col items-center justify-center min-h-[380px]"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: 0.3 }}
                    >
                      <div className="space-y-6 flex-1 flex flex-col justify-center w-full max-w-sm mx-auto">
                        <div>
                          <h3 className="text-base font-display font-semibold text-base-50 mb-1">Get Started</h3>
                          <p className="text-sm text-base-400">Enter a name and we'll map your face</p>
                        </div>

                        <div>
                          <label className="mb-1.5 block text-xs font-medium text-base-400 uppercase tracking-wider font-display">Identity Name</label>
                          <input
                            type="text"
                            value={enrollName}
                            onChange={(e) => setEnrollName(e.target.value)}
                            placeholder="e.g. Raju (leave empty for auto name)"
                            className="w-full rounded-xl border border-base-700/40 bg-base-950 px-4 py-3.5 text-sm text-base-50 placeholder:text-base-600 outline-none focus:border-accent-400/50 transition-colors"
                          />
                        </div>

                        <motion.button
                          whileHover={enrollStarting ? undefined : buttonHover}
                          whileTap={enrollStarting ? undefined : buttonTap}
                          onClick={startEnroll}
                          disabled={enrollStarting}
                          className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent-400 to-amber-500 px-5 py-4 text-sm font-semibold text-base-950 hover:opacity-90 active:scale-[0.98] transition-all cursor-pointer shadow-lg shadow-accent-400/20 disabled:opacity-70 disabled:cursor-wait"
                        >
                          {enrollStarting ? (
                            <>
                              <Loader2 className="h-4 w-4 animate-spin" />
                              {serverWakeSeconds < 3
                                ? 'Connecting...'
                                : `Waking up free server... ${serverWakeSeconds}s`}
                            </>
                          ) : (
                            <>
                              <Camera className="h-4 w-4" />
                              Enroll My Face
                            </>
                          )}
                        </motion.button>
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              )}

              {enrollUi.existingDuplicate ? (
                <motion.div
                  className="space-y-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-6 text-center"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-amber-500/20">
                    <AlertTriangle className="h-7 w-7 text-amber-400" />
                  </div>
                  <h2 className="text-lg font-display font-bold text-amber-400">
                    {enrollUi.existingDuplicate.reason === 'face' ? 'Face Already Mapped' : 'Profile Exists'}
                  </h2>
                  <p className="text-xs text-base-300 max-w-md mx-auto leading-relaxed">
                    {enrollUi.existingDuplicate.message ||
                      `Profile for '${enrollUi.existingDuplicate.name}' already exists.`}
                  </p>
                  <div className="flex flex-col sm:flex-row justify-center gap-3 pt-2">
                    <motion.button
                      whileHover={buttonHover}
                      whileTap={buttonTap}
                      onClick={async () => {
                        if (!enrollUi.existingDuplicate?.user_id) return;
                        try {
                          await api.delete(`/users/${encodeURIComponent(enrollUi.existingDuplicate.user_id)}`);
                          await fetchUsers();
                          setEnrollUi({
                            seqIdx: 0,
                            captured: [],
                            status: '',
                            pose: null,
                            guidance: null,
                            done: false,
                            finalName: null,
                            existing: false,
                            manualCapturing: false,
                            existingDuplicate: null,
                          });
                          setError(null);
                        } catch {
                          setError('Delete failed');
                        }
                      }}
                      className="flex items-center justify-center gap-2 rounded-xl bg-red-600 px-5 py-2.5 text-xs font-semibold text-white hover:bg-red-500 transition-all cursor-pointer"
                    >
                      <Trash2 className="h-4 w-4" /> Delete & Re-enroll
                    </motion.button>
                    <motion.button
                      whileHover={buttonHover}
                      whileTap={buttonTap}
                      onClick={() => {
                        setEnrollUi({
                          seqIdx: 0,
                          captured: [],
                          status: '',
                          pose: null,
                          guidance: null,
                          done: false,
                          finalName: null,
                          existing: false,
                          manualCapturing: false,
                          existingDuplicate: null,
                        });
                        setShowProfilesDrawer(true);
                      }}
                      className="flex items-center justify-center gap-2 rounded-xl border border-base-700 bg-base-800 px-5 py-2.5 text-xs font-semibold text-base-200 hover:bg-base-700 transition-colors cursor-pointer"
                    >
                      <Users className="h-4 w-4 text-accent-400" /> Manage Profiles
                    </motion.button>
                  </div>
                </motion.div>
              ) : !stream && !enrollUi.done ? (
                <motion.div
                  className="space-y-5 max-w-md mx-auto py-4"
                  variants={staggerChildren}
                  initial="initial"
                  animate="animate"
                >
                  {hasUsers && (
                    <>
                      <motion.div variants={fadeInUp}>
                        <label className="mb-1.5 block text-xs font-medium text-base-400 uppercase tracking-wider font-display">Existing Profile</label>
                        <select
                          value={existingUserId}
                          onChange={(e) => {
                            setExistingUserId(e.target.value);
                            setEnrollName('');
                          }}
                          className="w-full rounded-xl border border-base-700/40 bg-base-950 px-4 py-3 text-sm text-base-50 outline-none focus:border-accent-400/50 transition-colors appearance-none cursor-pointer"
                        >
                          <option value="">— New Person —</option>
                          {users.map((u) => (
                            <option key={u.user_id} value={u.user_id}>
                              {u.name} ({u.embedding_count} samples)
                            </option>
                          ))}
                        </select>
                      </motion.div>

                      {existingUserId && (
                        <motion.div variants={fadeInUp} className="rounded-xl border border-accent-400/20 bg-accent-400/[0.03] p-4 flex items-start gap-3 text-xs text-base-400">
                          <ShieldCheck className="h-5 w-5 shrink-0 text-accent-400 mt-0.5" />
                          <p>Additional samples improve accuracy across angles and lighting conditions.</p>
                        </motion.div>
                      )}

                      {!existingUserId && (
                        <motion.div variants={fadeInUp}>
                          <label className="mb-1.5 block text-xs font-medium text-base-400 uppercase tracking-wider font-display">
                            Identity Name
                          </label>
                          <input
                            type="text"
                            value={enrollName}
                            onChange={(e) => setEnrollName(e.target.value)}
                            placeholder="Enter name (optional)"
                            className="w-full rounded-xl border border-base-700/40 bg-base-950 px-4 py-3 text-sm text-base-50 placeholder:text-base-600 outline-none focus:border-accent-400/50 transition-colors"
                          />
                        </motion.div>
                      )}

                      <motion.button
                        variants={fadeInUp}
                        whileHover={enrollStarting ? undefined : buttonHover}
                        whileTap={enrollStarting ? undefined : buttonTap}
                        onClick={startEnroll}
                        disabled={enrollStarting}
                        className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent-400 to-amber-500 px-5 py-3.5 text-sm font-semibold text-base-950 hover:opacity-90 active:scale-[0.98] transition-all cursor-pointer shadow-lg shadow-accent-400/20 disabled:opacity-70 disabled:cursor-wait"
                      >
                        {enrollStarting ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            {serverWakeSeconds < 3
                              ? 'Connecting...'
                              : `Waking up free server... ${serverWakeSeconds}s`}
                          </>
                        ) : (
                          <>
                            <Camera className="h-4 w-4" />
                            {existingUserId ? 'Add Samples' : 'Enroll My Face'}
                          </>
                        )}
                      </motion.button>
                    </>
                  )}
                </motion.div>
              ) : enrollUi.done ? (
                <motion.div
                  className="space-y-4 text-center py-6 max-w-md mx-auto"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, type: 'spring', stiffness: 200 }}
                >
                  <motion.div
                    className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-mint-400/20 text-mint-400 border border-mint-400/30 glow-mint"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.4, delay: 0.1, type: 'spring', stiffness: 200 }}
                  >
                    <CheckCircle2 className="h-8 w-8" />
                  </motion.div>
                  <h2 className="text-xl font-display font-bold text-base-50">
                    {enrollUi.existing ? 'Profile Updated' : 'Identity Mapped'}
                  </h2>
                  <p className="text-sm text-base-300">
                    <span className="rounded bg-accent-400/10 border border-accent-400/20 px-2 py-0.5 font-mono text-accent-400 font-bold text-xs">
                      {enrollUi.finalName}
                    </span>
                  </p>
                  <p className="text-xs text-base-400">{enrollUi.status}</p>

                  <motion.div
                    className="flex justify-center gap-3 pt-4"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: 0.2 }}
                  >
                    {hasUsers && (
                      <motion.button
                        whileHover={buttonHover}
                        whileTap={buttonTap}
                        onClick={() => {
                          resetEnrollState();
                          setIsEnrolling(false);
                          setRecSubMode('camera');
                        }}
                        className="flex items-center gap-2 rounded-xl bg-mint-400 px-5 py-2.5 text-xs font-semibold text-base-950 hover:bg-mint-500 transition-all cursor-pointer"
                      >
                        <Eye className="h-4 w-4" /> Start Recognition
                      </motion.button>
                    )}
                    <motion.button
                      whileHover={buttonHover}
                      whileTap={buttonTap}
                      onClick={resetEnrollState}
                      className="flex items-center gap-2 rounded-xl border border-base-700 bg-base-800 px-5 py-2.5 text-xs font-semibold text-base-300 hover:bg-base-700 transition-colors cursor-pointer"
                    >
                      <RefreshCw className="h-4 w-4" /> Add Another
                    </motion.button>
                  </motion.div>
                </motion.div>
              ) : (
                <motion.div
                  className="space-y-5"
                  variants={staggerChildren}
                  initial="initial"
                  animate="animate"
                >
                  <motion.div variants={fadeInUp} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-accent-400 font-display font-bold text-sm">
                        {enrollUi.seqIdx + 1}
                      </span>
                      <span className="text-base-600 text-xs">/</span>
                      <span className="text-base-400 text-xs">{POSE_SEQUENCE.length}</span>
                      <span className="text-base-300 text-sm font-medium ml-2">{POSE_SEQUENCE[enrollUi.seqIdx].label}</span>
                    </div>
                    <div className="flex gap-1">
                      {POSE_SEQUENCE.map((_, i) => (
                        <motion.div
                          key={i}
                          className={cn(
                            'h-1.5 w-6 rounded-full transition-all duration-500',
                            i < enrollUi.captured.length
                              ? 'bg-mint-400'
                              : i === enrollUi.seqIdx
                                ? 'bg-accent-400'
                                : 'bg-base-700',
                          )}
                          initial={{ scaleX: 0 }}
                          animate={{ scaleX: 1 }}
                          transition={{ duration: 0.3, delay: i * 0.05 }}
                        />
                      ))}
                    </div>
                  </motion.div>

                  <motion.div
                    className="relative mx-auto w-full max-w-lg overflow-hidden rounded-2xl bg-base-950 border border-base-700/40 shadow-2xl"
                    variants={fadeInUp}
                  >
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="aspect-[4/3] w-full object-cover"
                      style={{ transform: 'scaleX(-1)' }}
                    />
                    <canvas
                      ref={overlayRef}
                      className="pointer-events-none absolute inset-0 h-full w-full"
                      style={{ transform: 'scaleX(-1)' }}
                    />

                    <motion.div
                      className="pointer-events-none absolute inset-0 flex items-center justify-center"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      <motion.div
                        className={cn(
                          'relative flex h-60 w-44 items-center justify-center rounded-[50%] border-2 transition-all duration-500',
                          enrollUi.guidance?.matched
                            ? 'border-mint-400 bg-mint-400/10 shadow-[0_0_40px_rgba(52,211,153,0.3)]'
                            : enrollUi.pose
                              ? 'border-accent-400/60 bg-accent-400/5 shadow-[0_0_30px_rgba(251,191,36,0.15)]'
                              : 'border-base-600/40 bg-base-950/50',
                        )}
                        animate={enrollUi.guidance?.matched ? { scale: [1, 1.05, 1] } : {}}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      >
                        <div className="absolute inset-4 rounded-[50%] border border-dashed border-base-600/20" />
                        <div className={cn(
                          'absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full transition-colors duration-300',
                          enrollUi.guidance?.matched ? 'bg-mint-400' : 'bg-accent-400/40',
                        )} />
                      </motion.div>
                    </motion.div>

                    {enrollUi.guidance?.directions?.up && (
                      <motion.div
                        className="absolute top-6 left-1/2 -translate-x-1/2 flex flex-col items-center text-accent-400"
                        animate={{ y: [0, -8, 0] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        <ArrowUp className="h-6 w-6 stroke-[2.5]" />
                      </motion.div>
                    )}
                    {enrollUi.guidance?.directions?.down && (
                      <motion.div
                        className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col items-center text-accent-400"
                        animate={{ y: [0, 8, 0] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        <ArrowDown className="h-6 w-6 stroke-[2.5]" />
                      </motion.div>
                    )}
                    {enrollUi.guidance?.directions?.left && (
                      <motion.div
                        className="absolute left-6 top-1/2 -translate-y-1/2 text-accent-400"
                        animate={{ x: [0, -6, 0] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        <ArrowLeft className="h-6 w-6 stroke-[2.5]" />
                      </motion.div>
                    )}
                    {enrollUi.guidance?.directions?.right && (
                      <motion.div
                        className="absolute right-6 top-1/2 -translate-y-1/2 text-accent-400"
                        animate={{ x: [0, 6, 0] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        <ArrowRight className="h-6 w-6 stroke-[2.5]" />
                      </motion.div>
                    )}

                    {enrollUi.guidance?.matched && (
                      <motion.div
                        className="absolute inset-0 flex items-center justify-center"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div className="flex items-center gap-2 rounded-full bg-mint-400 px-4 py-2 text-sm font-bold text-base-950 shadow-lg shadow-mint-400/30">
                          <Sparkles className="h-4 w-4" /> Captured
                        </div>
                      </motion.div>
                    )}
                  </motion.div>

                  <motion.div
                    variants={fadeInUp}
                    className={cn(
                      'rounded-xl border p-4 transition-all duration-300',
                      enrollUi.guidance?.matched
                        ? 'border-mint-400/30 bg-mint-400/10'
                        : enrollUi.pose
                          ? 'border-accent-400/30 bg-accent-400/10'
                          : 'border-base-700/40 bg-base-950/60',
                    )}
                  >
                    <div className="flex items-center gap-3">
                      {enrollUi.guidance?.matched ? (
                        <CheckCircle2 className="h-5 w-5 text-mint-400 shrink-0" />
                      ) : (
                        <Loader2 className="h-5 w-5 text-accent-400 shrink-0 animate-spin" />
                      )}
                      <p className="text-sm font-medium text-base-200">
                        {enrollUi.status || POSE_SEQUENCE[enrollUi.seqIdx].label}
                      </p>
                    </div>
                  </motion.div>

                  <motion.div variants={fadeInUp} className="flex items-center gap-3">
                    <motion.button
                      whileHover={buttonHover}
                      whileTap={buttonTap}
                      onClick={captureManually}
                      disabled={enrollUi.manualCapturing}
                      className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent-400 to-amber-500 px-5 py-3 text-xs font-semibold text-base-950 hover:opacity-90 active:scale-[0.98] transition-all disabled:opacity-50 cursor-pointer"
                    >
                      {enrollUi.manualCapturing ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Camera className="h-4 w-4" />
                      )}
                      {enrollUi.manualCapturing ? 'Capturing...' : 'Capture'}
                    </motion.button>

                    <motion.button
                      whileHover={buttonHover}
                      whileTap={buttonTap}
                      onClick={() => {
                        abortEnroll();
                        if (isEnrolling) setIsEnrolling(false);
                      }}
                      className="flex items-center gap-2 rounded-xl border border-base-700/40 bg-base-800/60 px-4 py-3 text-xs font-medium text-base-400 hover:bg-base-700/50 hover:text-base-200 transition-colors cursor-pointer"
                    >
                      <StopCircle className="h-4 w-4 text-red-400" /> Cancel
                    </motion.button>
                  </motion.div>
                </motion.div>
              )}
            </>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {showRecognize && (
            <motion.div
              className="glass-card p-6 sm:p-8"
              key="recognize"
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            >
              <motion.div
                className="mb-6 flex items-center justify-between"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-mint-400/10 border border-mint-400/20">
                    <Eye className="h-5 w-5 text-mint-400" />
                  </div>
                  <div>
                    <h2 className="text-base font-display font-semibold text-base-50">Recognition Active</h2>
                    <p className="text-xs text-base-400">{users.length} profile{users.length === 1 ? '' : 's'} enrolled</p>
                  </div>
                </div>

                <motion.button
                  whileHover={buttonHover}
                  whileTap={buttonTap}
                  onClick={handleStartEnrolling}
                  className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-accent-400 to-amber-500 px-4 py-2 text-xs font-semibold text-base-950 hover:opacity-90 active:scale-[0.98] transition-all cursor-pointer shadow-lg shadow-accent-400/20"
                >
                  <UserPlus className="h-4 w-4" /> Add Person
                </motion.button>
              </motion.div>

              <motion.div
                className="mb-6 flex rounded-xl bg-base-950 border border-base-700/40 p-1"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.05 }}
              >
                <motion.button
                  whileTap={{ scale: 0.97 }}
                  onClick={() => {
                    stopRecognition();
                    setRecSubMode('camera');
                  }}
                  className={cn(
                    'flex-1 flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition-all cursor-pointer',
                    recSubMode === 'camera'
                      ? 'bg-base-700/60 text-base-50'
                      : 'text-base-400 hover:text-base-200',
                  )}
                >
                  <Video className="h-3.5 w-3.5 text-mint-400" /> Live
                </motion.button>
                <motion.button
                  whileTap={{ scale: 0.97 }}
                  onClick={() => {
                    stopRecognition();
                    setRecSubMode('upload');
                  }}
                  className={cn(
                    'flex-1 flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition-all cursor-pointer',
                    recSubMode === 'upload'
                      ? 'bg-base-700/60 text-base-50'
                      : 'text-base-400 hover:text-base-200',
                  )}
                >
                  <Upload className="h-3.5 w-3.5 text-accent-400" /> Upload
                </motion.button>
              </motion.div>

              <AnimatePresence mode="wait">
                {recSubMode === 'camera' && (
                  <motion.div
                    key="camera"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.25 }}
                    className="stagger-children"
                  >
                    {!stream ? (
                      <motion.div
                        className="space-y-4 text-center py-10 max-w-sm mx-auto"
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4 }}
                      >
                        <motion.div
                          className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-mint-400/10 text-mint-400 border border-mint-400/20 glow-mint"
                          initial={{ scale: 0.8, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ duration: 0.4, type: 'spring', stiffness: 200 }}
                        >
                          <Zap className="h-8 w-8" />
                        </motion.div>
                        <p className="text-sm text-base-400">
                          Activate camera to identify enrolled faces in real time.
                        </p>
                        <motion.button
                          whileHover={buttonHover}
                          whileTap={buttonTap}
                          onClick={startRecognition}
                          className="inline-flex items-center gap-2 rounded-xl bg-mint-400 px-6 py-3 text-xs font-semibold text-base-950 hover:bg-mint-500 active:scale-[0.98] transition-all cursor-pointer"
                        >
                          <Play className="h-4 w-4" /> Start Recognition
                        </motion.button>
                      </motion.div>
                    ) : (
                      <motion.div
                        className="space-y-4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.3 }}
                      >
                        <motion.div
                          className="relative mx-auto w-full max-w-lg overflow-hidden rounded-2xl bg-base-950 border border-base-700/40 shadow-2xl"
                          initial={{ opacity: 0, scale: 0.98 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.3 }}
                        >
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="aspect-[4/3] w-full object-cover"
                            style={{ transform: 'scaleX(-1)' }}
                          />
                          <canvas
                            ref={overlayRef}
                            className="pointer-events-none absolute inset-0 h-full w-full"
                            style={{ transform: 'scaleX(-1)' }}
                          />

                          {recRunning && (
                            <motion.div
                              className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-base-950/80 border border-mint-400/30 px-3 py-1 text-[11px] font-bold text-mint-400"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ duration: 0.3 }}
                            >
                              <motion.span
                                className="h-2 w-2 rounded-full bg-mint-400"
                                animate={{ opacity: [1, 0.3, 1] }}
                                transition={{ duration: 1.5, repeat: Infinity }}
                              /> SCANNING
                            </motion.div>
                          )}
                        </motion.div>

                        <motion.div
                          className="rounded-xl border border-base-700/40 bg-base-950/60 p-4"
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3, delay: 0.1 }}
                        >
                          {recResult && !recResult.error ? (
                            recResult.faces.length === 0 ? (
                              <p className="flex items-center gap-2 text-xs text-base-400">
                                <XCircle className="h-4 w-4 text-base-600" /> No face detected
                              </p>
                            ) : (
                              <motion.div
                                className="space-y-2"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 0.3 }}
                              >
                                <p className="flex items-center gap-2 text-xs text-base-400">
                                  <Eye className="h-4 w-4 text-accent-400" />
                                  {recResult.faces.length} face{recResult.faces.length === 1 ? '' : 's'} found
                                </p>
                                <div className="flex flex-wrap gap-2">
                                  {recResult.faces.map((f, i) => (
                                    <motion.span
                                      key={i}
                                      initial={{ opacity: 0, y: 4 }}
                                      animate={{ opacity: 1, y: 0 }}
                                      transition={{ duration: 0.2, delay: i * 0.05 }}
                                      className={cn(
                                        'inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium',
                                        f.success
                                          ? 'bg-mint-400/10 text-mint-400 border border-mint-400/20'
                                          : 'bg-red-500/10 text-red-400 border border-red-500/20',
                                      )}
                                    >
                                      {f.success ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                                      {f.prediction} · {((f.confidence ?? 0) * 100).toFixed(0)}%
                                    </motion.span>
                                  ))}
                                </div>
                              </motion.div>
                            )
                          ) : (
                            <p className="flex items-center gap-2 text-xs text-base-400">
                              <Loader2 className="h-4 w-4 animate-spin text-accent-400" /> Analyzing...
                            </p>
                          )}
                        </motion.div>

                        <motion.button
                          whileHover={buttonHover}
                          whileTap={buttonTap}
                          onClick={stopRecognition}
                          className="flex items-center justify-center gap-2 rounded-xl bg-red-600 px-5 py-2.5 text-xs font-semibold text-white hover:bg-red-500 transition-colors cursor-pointer"
                        >
                          <StopCircle className="h-4 w-4" /> Stop
                        </motion.button>
                      </motion.div>
                    )}
                  </motion.div>
                )}

                {recSubMode === 'upload' && (
                  <motion.div
                    key="upload"
                    className="flex flex-col gap-5 sm:flex-row"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.25 }}
                  >
                    <motion.div
                      className="flex-1 space-y-4"
                      variants={staggerChildren}
                      initial="initial"
                      animate="animate"
                    >
                      <motion.label
                        variants={fadeInUp}
                        className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-base-700/40 bg-base-950 p-8 text-center hover:border-accent-400/30 transition-colors"
                      >
                        {previewUrl ? (
                          <motion.img
                            src={previewUrl}
                            alt="preview"
                            className="max-h-56 rounded-xl object-contain shadow-lg"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.3 }}
                          />
                        ) : (
                          <>
                            <ImageIcon className="h-10 w-10 text-base-600" />
                            <span className="text-xs text-base-400">Drop image or click to upload</span>
                          </>
                        )}
                        <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                      </motion.label>

                      <motion.button
                        variants={fadeInUp}
                        whileHover={buttonHover}
                        whileTap={buttonTap}
                        onClick={uploadAndRecognize}
                        disabled={!selectedFile || uploadLoading}
                        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent-400 to-amber-500 px-5 py-3 text-xs font-semibold text-base-950 hover:opacity-90 disabled:opacity-30 disabled:cursor-not-allowed transition-all cursor-pointer"
                      >
                        {uploadLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Cpu className="h-4 w-4" />}
                        {uploadLoading ? 'Processing...' : 'Identify'}
                      </motion.button>
                    </motion.div>

                    <motion.div
                      className="flex-1 rounded-2xl border border-base-700/40 bg-base-950 p-5 flex flex-col justify-center"
                      initial={{ opacity: 0, x: 12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: 0.1 }}
                    >
                      {uploadResult ? (
                        uploadResult.error ? (
                          <div className="flex items-center gap-2 text-xs text-red-400">
                            <XCircle className="h-4 w-4 shrink-0" /> {uploadResult.error}
                          </div>
                        ) : uploadResult.faces.length === 0 ? (
                          <div className="flex flex-col items-center justify-center gap-2 text-center text-base-500 py-8">
                            <User className="h-10 w-10 opacity-20" />
                            <p className="text-xs">No face detected</p>
                          </div>
                        ) : (
                          <motion.div
                            className="space-y-3"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.3 }}
                          >
                            <p className="flex items-center gap-2 text-xs text-base-400">
                              <Eye className="h-4 w-4 text-accent-400" />
                              {uploadResult.faces.length} face{uploadResult.faces.length === 1 ? '' : 's'} found
                            </p>
                            {uploadResult.faces.map((f, i) => (
                              <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.3, delay: i * 0.08 }}
                                className="space-y-2 rounded-xl border border-base-700/40 bg-base-800/40 p-3"
                              >
                                <div className="flex items-center gap-2">
                                  {f.success ? (
                                    <CheckCircle2 className="h-4 w-4 text-mint-400 shrink-0" />
                                  ) : (
                                    <XCircle className="h-4 w-4 text-red-400 shrink-0" />
                                  )}
                                  <span className="text-sm font-medium text-base-50">{f.prediction}</span>
                                </div>
                                <div className="space-y-1">
                                  <div className="flex justify-between text-xs text-base-400">
                                    <span>Confidence</span>
                                    <span className="font-mono">{((f.confidence ?? 0) * 100).toFixed(0)}%</span>
                                  </div>
                                  <div className="h-1.5 w-full rounded-full bg-base-700 overflow-hidden">
                                    <motion.div
                                      className={cn('h-full rounded-full', f.success ? 'bg-mint-400' : 'bg-red-400')}
                                      initial={{ width: 0 }}
                                      animate={{ width: `${Math.max(0, Math.min(100, (f.confidence ?? 0) * 100))}%` }}
                                      transition={{ duration: 0.6, ease: 'easeOut' }}
                                    />
                                  </div>
                                </div>
                              </motion.div>
                            ))}
                          </motion.div>
                        )
                      ) : (
                        <div className="flex flex-col items-center justify-center gap-2 text-center text-base-600 py-8">
                          <User className="h-10 w-10 opacity-20" />
                          <p className="text-xs">Upload to identify</p>
                        </div>
                      )}
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <motion.footer
        className="relative z-10 border-t border-base-700/30 py-6 mt-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <div className="mx-auto flex max-w-4xl flex-col items-center gap-4 px-4">
          <div className="flex items-center gap-3">
            {[
              { icon: Mail, href: 'mailto:try.rajusah@gmail.com', label: 'Email', title: 'try.rajusah@gmail.com' },
              { icon: LinkIcon, href: 'https://linkedin.com/in/rajusah18', label: 'LinkedIn', title: 'linkedin.com/in/rajusah18' },
              { icon: GitBranch, href: 'https://github.com/raju-sah', label: 'GitHub', title: 'github.com/raju-sah' },
              { icon: AtSign, href: 'https://instagram.com/okay.raju', label: 'Instagram', title: 'instagram.com/okay.raju' },
              { icon: Globe, href: 'https://sahraju.com.np/', label: 'Portfolio', title: 'sahraju.com.np' },
            ].map(({ icon: Icon, href, label, title }) => (
              <motion.a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={label}
                title={title}
                whileHover={{ scale: 1.1, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="flex h-9 w-9 items-center justify-center rounded-full border border-base-700/40 bg-base-800/40 text-base-400 transition-all hover:border-accent-400/30 hover:text-accent-400"
              >
                <Icon className="h-3.5 w-3.5" />
              </motion.a>
            ))}
          </div>
          <p className="text-[11px] text-base-600 font-display">
            © {new Date().getFullYear()} NeuroVision · Raju Sah
          </p>
        </div>
      </motion.footer>
    </div>
  );
}

export default App;