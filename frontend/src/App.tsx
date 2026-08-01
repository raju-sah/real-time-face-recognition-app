import { useState, ChangeEvent, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Upload,
  User,
  ShieldCheck,
  AlertCircle,
  AlertTriangle,
  Loader2,
  Camera,
  Scan,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Cpu,
  Fingerprint,
  Zap,
  Activity,
  Video,
  Play,
  StopCircle,
  Users,
  Trash2,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Sparkles,
  X,
  Image as ImageIcon,
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
  { key: 'front', label: 'Look straight at the camera' },
  { key: 'left', label: 'Turn your head to your LEFT' },
  { key: 'right', label: 'Turn your head to your RIGHT' },
  { key: 'up', label: 'Tilt your chin UP' },
  { key: 'down', label: 'Look slightly DOWN' },
  { key: 'front', label: 'Face straight again' },
];

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const api = axios.create({ baseURL: API_URL, timeout: 30000 });

type View = 'enroll' | 'recognize';
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

function App() {
  const [view, setView] = useState<View>('enroll');
  const [recSubMode, setRecSubMode] = useState<RecSubMode>('camera');
  const [showProfilesDrawer, setShowProfilesDrawer] = useState(false);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
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

  // Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<RecognitionResult | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  // Enroll state
  const [enrollName, setEnrollName] = useState('');
  const [existingUserId, setExistingUserId] = useState('');
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

  // Live Recognize state
  const [recResult, setRecResult] = useState<RecognitionResult | null>(null);
  const [recRunning, setRecRunning] = useState(false);

  // Enrolled Users
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
      setError('Unable to access camera. Please check permissions.');
      return false;
    }
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) return;
    video.srcObject = stream;
    video.play().catch(() => {});
  }, [stream]);

  const renderFrame = (result: RecognitionResult | null): string | null => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !video.videoWidth || video.readyState < 1) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    if (!dataUrl || dataUrl === 'data:,' || dataUrl === 'data:,') return null;

    if (result && result.faces) {
      for (const face of result.faces) {
        if (!face.box || !face.prediction) continue;
        const [x, y, w, h] = face.box;
        const rx = canvas.width - x - w;
        const color = face.success ? '#22c55e' : '#ef4444';
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(rx, y, w, h);
        ctx.fillStyle = color;
        const label = `${face.prediction} ${((face.confidence ?? 0) * 100).toFixed(0)}%`;
        ctx.font = 'bold 16px sans-serif';
        const tw = ctx.measureText(label).width;
        const labelY = Math.max(0, y - 28);
        ctx.fillRect(rx, labelY, tw + 14, 26);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, rx + 7, Math.max(18, labelY + 18));
      }
    }
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

  const handleSwitchView = (newView: View) => {
    if (view === newView) return;
    stopCamera();
    enrollRef.current = null;
    recRef.current = null;
    setRecRunning(false);
    setView(newView);
  };

  // ------------------------------------------------------------ enrollment
  const startEnroll = async () => {
    setError(null);

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
      const msg = err instanceof Error ? err.message : 'Failed to start enrollment';
      setError(msg);
      stopCamera();
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
        setEnrollUi((prev) => ({ ...prev, status: 'Waiting for camera frames...', manualCapturing: false }));
        return;
      }
      const fetched = await fetch(dataUrl);
      const rawBlob = await fetched.blob();
      if (rawBlob.size === 0) {
        setEnrollUi((prev) => ({ ...prev, status: 'Waiting for camera frames...', manualCapturing: false }));
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
            status: 'All samples captured — saving profile...',
            pose: data.pose ?? null,
            guidance: data.guidance ?? null,
            manualCapturing: false,
          }));
          await completeEnroll();
        }
      } else {
        setEnrollUi((prev) => ({
          ...prev,
          status: data.message || 'Adjust head position',
          pose: data.pose ?? null,
          guidance: data.guidance ?? null,
          manualCapturing: false,
        }));
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Enroll request failed';
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
          ? `Added ${data.added_count} new sample${data.added_count === 1 ? '' : 's'} to ${data.name} — ${data.embedding_count} total`
          : `Profile saved with ${data.embedding_count} samples`,
      }));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to save profile';
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

  // ------------------------------------------------------------ live recognition
  const startRecognition = async () => {
    setError(null);
    const ok = await startCamera();
    if (!ok) return;
    setRecResult(null);
    recRef.current = { running: true, interval: null, lastResult: null };
    setRecRunning(true);
    recRef.current.interval = setInterval(runRecFrame, 400);
  };

  const runRecFrame = async () => {
    const r = recRef.current;
    if (!r || !r.running || busyRef.current) return;
    busyRef.current = true;
    try {
      const dataUrl = renderFrame(r.lastResult);
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
    setRecRunning(false);
    stopCamera();
    setRecResult(null);
  };

  // ------------------------------------------------------------ upload
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

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-blue-500 selection:text-white">
      {/* ------------------------------------------------------ SLIDE-OVER PROFILES DRAWER */}
      {showProfilesDrawer && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/70 backdrop-blur-sm transition-opacity">
          <div className="w-full max-w-md bg-zinc-900 border-l border-zinc-800 p-6 flex flex-col h-full shadow-2xl">
            <div className="flex items-center justify-between pb-4 border-b border-zinc-800">
              <div className="flex items-center gap-2.5">
                <Users className="h-5 w-5 text-blue-400" />
                <h2 className="font-semibold text-lg text-white">Enrolled Profiles</h2>
                <span className="rounded-full bg-blue-500/20 px-2.5 py-0.5 text-xs font-bold text-blue-300">
                  {users.length}
                </span>
              </div>
              <button
                onClick={() => setShowProfilesDrawer(false)}
                className="rounded-lg p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-white transition-colors cursor-pointer"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto py-4 space-y-2">
              {usersLoading ? (
                <div className="flex items-center justify-center py-12 text-sm text-zinc-500 gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-400" /> Loading profiles...
                </div>
              ) : users.length === 0 ? (
                <div className="py-12 text-center text-zinc-500 text-sm">
                  <User className="h-10 w-10 mx-auto mb-2 opacity-30" />
                  No enrolled profiles yet. Start by enrolling a face!
                </div>
              ) : (
                users.map((u) => (
                  <div
                    key={u.user_id}
                    className="flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3 hover:border-zinc-700 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 text-blue-400 border border-blue-500/20 font-bold text-sm">
                        {u.name.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <p className="font-semibold text-sm text-zinc-100">{u.name}</p>
                        <p className="text-xs text-zinc-500">
                          {u.embedding_count} samples · {u.created_at.slice(0, 10)}
                        </p>
                      </div>
                    </div>

                    <button
                      onClick={async () => {
                        try {
                          await api.delete(`/users/${encodeURIComponent(u.user_id)}`);
                          fetchUsers();
                        } catch {
                          /* ignore */
                        }
                      }}
                      className="rounded-lg p-2 text-zinc-500 hover:bg-red-500/10 hover:text-red-400 transition-colors cursor-pointer"
                      title="Delete profile"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------ MAIN APPLICATION WRAPPER */}
      <div className="mx-auto max-w-4xl px-4 py-6 sm:py-8">
        {/* Header */}
        <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-zinc-800/80 pb-6">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 via-indigo-600 to-violet-600 shadow-lg shadow-blue-500/20">
              <Scan className="h-6 w-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">NeuroVision</h1>
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" /> Live API
                </span>
              </div>
              <p className="text-xs text-zinc-400">Minimalist AI Face Recognition Studio</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Profile Drawer Trigger Button */}
            <button
              onClick={() => setShowProfilesDrawer(true)}
              className="flex items-center gap-2 rounded-xl bg-zinc-900 border border-zinc-800 px-3 py-2 text-xs font-semibold text-zinc-300 hover:border-zinc-700 hover:text-white transition-all cursor-pointer"
            >
              <Users className="h-4 w-4 text-blue-400" />
              <span>{users.length}</span>
            </button>
          </div>
        </header>

        {error && (
          <div className="mb-6 flex items-center gap-2 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-xs font-medium text-red-300">
            <AlertCircle className="h-4 w-4 shrink-0" /> {error}
            <button className="ml-auto p-1" onClick={() => setError(null)}>
              <XCircle className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* Unified Mode Switcher Pill */}
        <div className="mb-6 flex rounded-xl bg-zinc-900 border border-zinc-800 p-1 w-full">
          <button
            onClick={() => handleSwitchView('enroll')}
            className={cn(
              'flex-1 flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold transition-all cursor-pointer',
              view === 'enroll'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-zinc-400 hover:text-white',
            )}
          >
            <Fingerprint className="h-4 w-4" /> Enroll
          </button>
          <button
            onClick={() => handleSwitchView('recognize')}
            className={cn(
              'flex-1 flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold transition-all cursor-pointer',
              view === 'recognize'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-zinc-400 hover:text-white',
            )}
          >
            <Scan className="h-4 w-4" /> Recognize
          </button>
        </div>

        {/* ------------------------------------------------------ ENROLLMENT MODE */}
        {view === 'enroll' && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur-xl p-6 shadow-xl">
            {enrollUi.existingDuplicate ? (
              <div className="space-y-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-6 text-center shadow-xl">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-amber-500/20">
                  <AlertTriangle className="h-7 w-7 text-amber-400" />
                </div>
                <h2 className="text-lg font-bold text-amber-200">
                  {enrollUi.existingDuplicate.reason === 'face'
                    ? 'Face Already Enrolled'
                    : 'Profile Already Exists'}
                </h2>
                <p className="text-xs text-zinc-300 max-w-md mx-auto leading-relaxed">
                  {enrollUi.existingDuplicate.message ||
                    `Profile for '${enrollUi.existingDuplicate.name}' is already enrolled. Re-enrollment is restricted for existing profiles.`}
                </p>
                <div className="flex flex-col sm:flex-row justify-center gap-3 pt-2">
                  <button
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
                        setError('Failed to delete existing profile');
                      }
                    }}
                    className="flex items-center justify-center gap-2 rounded-xl bg-red-600 px-5 py-2.5 text-xs font-semibold text-white hover:bg-red-500 shadow-lg shadow-red-500/20 active:scale-[0.98] transition-all cursor-pointer"
                  >
                    <Trash2 className="h-4 w-4" /> Delete '{enrollUi.existingDuplicate.name}' & Re-enroll
                  </button>

                  <button
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
                    className="flex items-center justify-center gap-2 rounded-xl border border-zinc-700 bg-zinc-800 px-5 py-2.5 text-xs font-semibold text-zinc-200 hover:bg-zinc-700 transition-colors cursor-pointer"
                  >
                    <Users className="h-4 w-4 text-blue-400" /> Manage Enrolled Profiles
                  </button>
                </div>
              </div>
            ) : !stream && !enrollUi.done ? (
              <div className="space-y-5 max-w-lg mx-auto py-4">
                <div>
                  <label className="mb-1.5 block text-xs font-semibold text-zinc-300 uppercase tracking-wider">Target Profile</label>
                  <select
                    value={existingUserId}
                    onChange={(e) => {
                      setExistingUserId(e.target.value);
                      setEnrollName('');
                    }}
                    className="w-full rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-3 text-sm text-zinc-200 outline-none focus:border-blue-500 transition-colors"
                  >
                    <option value="">New Person</option>
                    {users.map((u) => (
                      <option key={u.user_id} value={u.user_id}>
                        {u.name} ({u.embedding_count} samples)
                      </option>
                    ))}
                  </select>
                </div>

                {!existingUserId && (
                  <div>
                    <label className="mb-1.5 block text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                      Name
                    </label>
                    <input
                      type="text"
                      value={enrollName}
                      onChange={(e) => setEnrollName(e.target.value)}
                      placeholder="e.g. Raju (leave empty for auto name)"
                      className="w-full rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-3 text-sm text-zinc-200 outline-none focus:border-blue-500 transition-colors"
                    />
                  </div>
                )}

                <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-4 flex items-start gap-3 text-xs text-zinc-400">
                  <ShieldCheck className="h-5 w-5 shrink-0 text-blue-400 mt-0.5" />
                  <p>
                    {existingUserId
                      ? 'Adding additional samples to this profile improves recognition accuracy from different angles.'
                      : 'You will be guided through 6 automatic head poses. Captures occur automatically when target angles are matched.'}
                  </p>
                </div>

                <button
                  onClick={startEnroll}
                  className="w-full flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3.5 text-sm font-semibold text-white hover:bg-blue-500 shadow-lg shadow-blue-500/20 active:scale-[0.98] transition-all cursor-pointer"
                >
                  <Camera className="h-4 w-4" />
                  {existingUserId ? 'Add More Samples' : 'Start Guided Enrollment'}
                </button>
              </div>
            ) : enrollUi.done ? (
              <div className="space-y-4 text-center py-6 max-w-md mx-auto">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  <CheckCircle2 className="h-8 w-8" />
                </div>
                <h2 className="text-xl font-bold text-white">
                  {enrollUi.existing ? 'Profile Updated' : 'Enrollment Complete'}
                </h2>
                <p className="text-xs text-zinc-300">
                  Profile registered as{' '}
                  <span className="rounded bg-blue-500/20 px-2 py-0.5 font-mono text-blue-300 font-bold">
                    {enrollUi.finalName}
                  </span>
                </p>
                <p className="text-xs text-zinc-400">{enrollUi.status}</p>

                <div className="flex justify-center gap-3 pt-2">
                  <button
                    onClick={() => handleSwitchView('recognize')}
                    className="flex items-center gap-2 rounded-xl bg-emerald-600 px-5 py-2.5 text-xs font-semibold text-white hover:bg-emerald-500 transition-all cursor-pointer"
                  >
                    <Video className="h-4 w-4" /> Live Recognition
                  </button>
                  <button
                    onClick={() => {
                      setEnrollUi({ seqIdx: 0, captured: [], status: '', pose: null, guidance: null, done: false, finalName: null, existing: false, manualCapturing: false });
                      setEnrollName('');
                      setExistingUserId('');
                    }}
                    className="flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900 px-5 py-2.5 text-xs font-semibold text-zinc-300 hover:bg-zinc-800 transition-all cursor-pointer"
                  >
                    <RefreshCw className="h-4 w-4" /> Enroll Another
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                {/* Sleek Step Indicator Bar */}
                <div className="flex items-center justify-between text-xs font-medium text-zinc-400">
                  <span className="text-blue-400 font-semibold">
                    Step {enrollUi.seqIdx + 1} of {POSE_SEQUENCE.length}: {POSE_SEQUENCE[enrollUi.seqIdx].label}
                  </span>
                  <span>{enrollUi.captured.length}/{POSE_SEQUENCE.length} Captured</span>
                </div>

                <div className="grid grid-cols-6 gap-1.5">
                  {POSE_SEQUENCE.map((p, i) => {
                    const isFilled = i < enrollUi.captured.length;
                    const isCurrent = i === enrollUi.seqIdx;
                    return (
                      <div
                        key={`${p.key}-${i}`}
                        className={cn(
                          'h-1.5 rounded-full transition-all duration-300',
                          isFilled
                            ? 'bg-emerald-400'
                            : isCurrent
                              ? 'bg-blue-500 animate-pulse'
                              : 'bg-zinc-800',
                        )}
                        title={p.label}
                      />
                    );
                  })}
                </div>

                {/* Camera Viewport with Oval Target Guide */}
                <div className="relative mx-auto w-full max-w-lg overflow-hidden rounded-2xl bg-black border border-zinc-800 shadow-2xl">
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

                  {/* Dynamic Target Alignment Oval Overlay */}
                  <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                    <div
                      className={cn(
                        'relative flex h-60 w-44 items-center justify-center rounded-[50%] border-2 transition-all duration-300',
                        enrollUi.guidance?.matched
                          ? 'border-emerald-400 bg-emerald-500/10 shadow-[0_0_30px_rgba(52,211,153,0.5)] animate-pulse'
                          : enrollUi.pose
                            ? 'border-amber-400/80 bg-amber-500/5 shadow-[0_0_20px_rgba(251,191,36,0.25)]'
                            : 'border-zinc-500/40 bg-black/20',
                      )}
                    >
                      {/* Directional Cues */}
                      {enrollUi.guidance?.directions?.up && (
                        <div className="absolute top-2 flex flex-col items-center text-amber-400 animate-bounce">
                          <ArrowUp className="h-6 w-6 stroke-[3]" />
                          <span className="text-[9px] font-extrabold uppercase tracking-wider bg-black/80 px-2 py-0.5 rounded border border-amber-500/30">TILT UP</span>
                        </div>
                      )}
                      {enrollUi.guidance?.directions?.down && (
                        <div className="absolute bottom-2 flex flex-col items-center text-amber-400 animate-bounce">
                          <span className="text-[9px] font-extrabold uppercase tracking-wider bg-black/80 px-2 py-0.5 rounded border border-amber-500/30">TILT DOWN</span>
                          <ArrowDown className="h-6 w-6 stroke-[3]" />
                        </div>
                      )}
                      {enrollUi.guidance?.directions?.left && (
                        <div className="absolute left-2 flex items-center gap-1 text-amber-400 animate-pulse">
                          <ArrowLeft className="h-6 w-6 stroke-[3]" />
                          <span className="text-[9px] font-extrabold uppercase tracking-wider bg-black/80 px-1 py-0.5 rounded border border-amber-500/30">TURN LEFT</span>
                        </div>
                      )}
                      {enrollUi.guidance?.directions?.right && (
                        <div className="absolute right-2 flex items-center gap-1 text-amber-400 animate-pulse">
                          <span className="text-[9px] font-extrabold uppercase tracking-wider bg-black/80 px-1 py-0.5 rounded border border-amber-500/30">TURN RIGHT</span>
                          <ArrowRight className="h-6 w-6 stroke-[3]" />
                        </div>
                      )}

                      {enrollUi.guidance?.matched && (
                        <div className="flex items-center gap-1.5 rounded-full bg-emerald-500 px-3 py-1 text-xs font-bold text-black shadow-lg">
                          <Sparkles className="h-4 w-4 animate-spin" /> MATCHED!
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Status Guidance Banner Card */}
                <div
                  className={cn(
                    'rounded-xl border p-4 transition-all duration-300',
                    enrollUi.guidance?.matched
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200'
                      : enrollUi.pose
                        ? 'border-amber-500/40 bg-amber-500/10 text-amber-200'
                        : 'border-zinc-800 bg-zinc-950 text-zinc-300',
                  )}
                >
                  <div className="flex items-center gap-3">
                    {enrollUi.guidance?.matched ? (
                      <CheckCircle2 className="h-6 w-6 text-emerald-400 shrink-0 animate-bounce" />
                    ) : enrollUi.guidance?.directions?.up ? (
                      <ArrowUp className="h-6 w-6 text-amber-400 shrink-0 animate-bounce" />
                    ) : enrollUi.guidance?.directions?.down ? (
                      <ArrowDown className="h-6 w-6 text-amber-400 shrink-0 animate-bounce" />
                    ) : enrollUi.guidance?.directions?.left ? (
                      <ArrowLeft className="h-6 w-6 text-amber-400 shrink-0 animate-bounce" />
                    ) : enrollUi.guidance?.directions?.right ? (
                      <ArrowRight className="h-6 w-6 text-amber-400 shrink-0 animate-bounce" />
                    ) : (
                      <Activity className="h-5 w-5 text-blue-400 shrink-0 animate-pulse" />
                    )}
                    <div>
                      <p className="text-sm font-bold text-white">
                        {enrollUi.status || POSE_SEQUENCE[enrollUi.seqIdx].label}
                      </p>
                      <p className="text-xs text-zinc-400">
                        {enrollUi.guidance?.matched
                          ? 'Pose aligned properly! Holding still to capture...'
                          : 'Follow visual directional prompts or click manual capture.'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center gap-3">
                  <button
                    onClick={captureManually}
                    disabled={enrollUi.manualCapturing}
                    className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-5 py-3 text-xs font-semibold text-white hover:from-blue-500 hover:to-indigo-500 shadow-lg shadow-blue-500/20 active:scale-[0.98] transition-all disabled:opacity-50 cursor-pointer"
                  >
                    {enrollUi.manualCapturing ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Camera className="h-4 w-4" />
                    )}
                    {enrollUi.manualCapturing ? 'Capturing...' : '📸 Capture Pose Manually'}
                  </button>

                  <button
                    onClick={abortEnroll}
                    className="flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900 px-4 py-3 text-xs font-semibold text-zinc-400 hover:bg-zinc-800 hover:text-white transition-colors cursor-pointer"
                  >
                    <StopCircle className="h-4 w-4 text-red-400" /> Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ------------------------------------------------------ RECOGNITION MODE */}
        {view === 'recognize' && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur-xl p-6 shadow-xl space-y-5">
            {/* Recognition Sub-mode Switcher */}
            <div className="flex items-center justify-between border-b border-zinc-800/80 pb-4">
              <div className="flex rounded-xl bg-zinc-950 border border-zinc-800 p-1">
                <button
                  onClick={() => {
                    stopRecognition();
                    setRecSubMode('camera');
                  }}
                  className={cn(
                    'flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-all cursor-pointer',
                    recSubMode === 'camera'
                      ? 'bg-zinc-800 text-white shadow-sm'
                      : 'text-zinc-400 hover:text-white',
                  )}
                >
                  <Video className="h-3.5 w-3.5 text-green-400" /> Live Camera
                </button>
                <button
                  onClick={() => {
                    stopRecognition();
                    setRecSubMode('upload');
                  }}
                  className={cn(
                    'flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-all cursor-pointer',
                    recSubMode === 'upload'
                      ? 'bg-zinc-800 text-white shadow-sm'
                      : 'text-zinc-400 hover:text-white',
                  )}
                >
                  <Upload className="h-3.5 w-3.5 text-violet-400" /> Upload Image
                </button>
              </div>

              <span className="text-xs text-zinc-400">
                {recSubMode === 'camera' ? 'Real-time feed recognition' : 'Single photo match'}
              </span>
            </div>

            {/* Sub-mode: Live Camera */}
            {recSubMode === 'camera' && (
              <div>
                {!stream ? (
                  <div className="space-y-4 text-center py-8 max-w-md mx-auto">
                    <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-green-500/10 text-green-400 border border-green-500/20">
                      <Zap className="h-7 w-7" />
                    </div>
                    <p className="text-xs text-zinc-400">
                      Opens your camera to continuously identify enrolled individuals in real time.
                    </p>
                    <button
                      onClick={startRecognition}
                      className="inline-flex items-center gap-2 rounded-xl bg-green-600 px-6 py-3 text-xs font-semibold text-white hover:bg-green-500 shadow-lg shadow-green-500/20 active:scale-[0.98] transition-all cursor-pointer"
                    >
                      <Play className="h-4 w-4" /> Start Live Recognition
                    </button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="relative mx-auto w-full max-w-lg overflow-hidden rounded-2xl bg-black border border-zinc-800 shadow-2xl">
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
                        <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/70 border border-green-500/30 px-3 py-1 text-[11px] font-bold text-green-400">
                          <Activity className="h-3 w-3 animate-pulse" /> RECOGNIZING
                        </div>
                      )}
                    </div>

                    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                      {recResult && !recResult.error ? (
                        recResult.faces.length === 0 ? (
                          <p className="flex items-center gap-2 text-xs text-zinc-400">
                            <XCircle className="h-4 w-4 text-zinc-500" /> No face in frame
                          </p>
                        ) : (
                          <div className="space-y-1.5">
                            <p className="flex items-center gap-2 text-xs text-zinc-500">
                              <Activity className="h-4 w-4 text-blue-400" />
                              {recResult.faces.length} face{recResult.faces.length === 1 ? '' : 's'} detected
                            </p>
                            <div className="flex flex-wrap gap-2">
                              {recResult.faces.map((f, i) => (
                                <span
                                  key={i}
                                  className={cn(
                                    'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium',
                                    f.success ? 'bg-green-500/15 text-green-300' : 'bg-red-500/15 text-red-300',
                                  )}
                                >
                                  {f.success ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                                  {f.prediction} · {((f.confidence ?? 0) * 100).toFixed(0)}%
                                </span>
                              ))}
                            </div>
                          </div>
                        )
                      ) : (
                        <p className="flex items-center gap-2 text-xs text-zinc-400">
                          <Loader2 className="h-4 w-4 animate-spin text-blue-400" /> Scanning frame for faces...
                        </p>
                      )}
                    </div>

                    <button
                      onClick={stopRecognition}
                      className="flex items-center justify-center gap-2 rounded-xl bg-red-600 px-5 py-2.5 text-xs font-semibold text-white hover:bg-red-500 transition-colors cursor-pointer"
                    >
                      <StopCircle className="h-4 w-4" /> Stop Live Recognition
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Sub-mode: Image Upload */}
            {recSubMode === 'upload' && (
              <div className="flex flex-col gap-5 sm:flex-row">
                <div className="flex-1 space-y-4">
                  <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-zinc-800 bg-zinc-950 p-8 text-center hover:border-blue-500/80 transition-colors">
                    {previewUrl ? (
                      <img src={previewUrl} alt="preview" className="max-h-56 rounded-xl object-contain shadow-lg" />
                    ) : (
                      <>
                        <ImageIcon className="h-10 w-10 text-zinc-600" />
                        <span className="text-xs text-zinc-400">Click or drag image to identify face</span>
                      </>
                    )}
                    <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                  </label>

                  <button
                    onClick={uploadAndRecognize}
                    disabled={!selectedFile || uploadLoading}
                    className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-xs font-semibold text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer"
                  >
                    {uploadLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Cpu className="h-4 w-4" />}
                    {uploadLoading ? 'Processing Image...' : 'Recognize Face'}
                  </button>
                </div>

                <div className="flex-1 rounded-2xl border border-zinc-800 bg-zinc-950 p-5 flex flex-col justify-center">
                  {uploadResult ? (
                    uploadResult.error ? (
                      <div className="flex items-center gap-2 text-xs text-red-400">
                        <XCircle className="h-4 w-4 shrink-0" /> {uploadResult.error}
                      </div>
                    ) : uploadResult.faces.length === 0 ? (
                      <div className="flex flex-col items-center justify-center gap-2 text-center text-zinc-500 py-8">
                        <XCircle className="h-8 w-8 opacity-40" />
                        <p className="text-xs">No face detected in the image</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <p className="flex items-center gap-2 text-xs text-zinc-400">
                          <Activity className="h-4 w-4 text-blue-400" />
                          {uploadResult.faces.length} face{uploadResult.faces.length === 1 ? '' : 's'} detected
                        </p>
                        {uploadResult.faces.map((f, i) => (
                          <div key={i} className="space-y-1.5 rounded-xl border border-zinc-800 bg-zinc-900/50 p-3">
                            <div className="flex items-center gap-2">
                              {f.success ? (
                                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                              ) : (
                                <XCircle className="h-4 w-4 text-red-400 shrink-0" />
                              )}
                              <span className="text-sm font-semibold text-white">{f.prediction}</span>
                            </div>
                            <div className="space-y-1">
                              <div className="flex justify-between text-xs text-zinc-400">
                                <span>Confidence</span>
                                <span className="font-mono">{((f.confidence ?? 0) * 100).toFixed(0)}%</span>
                              </div>
                              <div className="h-1.5 w-full rounded-full bg-zinc-800 overflow-hidden">
                                <div
                                  className={cn('h-full rounded-full transition-all duration-300', f.success ? 'bg-emerald-400' : 'bg-red-400')}
                                  style={{ width: `${Math.max(0, Math.min(100, (f.confidence ?? 0) * 100))}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )
                  ) : (
                    <div className="flex flex-col items-center justify-center gap-2 text-center text-zinc-600 py-8">
                      <User className="h-8 w-8 opacity-40" />
                      <p className="text-xs">Upload a photo to see identification metrics</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
