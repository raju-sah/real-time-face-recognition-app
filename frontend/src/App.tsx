import { useState, ChangeEvent, useRef, useEffect } from 'react';
import axios from 'axios';
import { 
  Upload, 
  User, 
  ShieldCheck, 
  AlertCircle, 
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
  SwitchCamera
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface RecognitionResult {
  success: boolean;
  prediction?: string;
  confidence?: number;
  box?: number[];
  error?: string;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'upload' | 'camera'>('upload');
  const [stream, setStream] = useState<MediaStream | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

  // Handle mode switching and camera cleanup
  useEffect(() => {
    if (mode !== 'camera' && stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  }, [mode]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } 
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error("Camera error:", err);
      setError("Unable to access camera. Please check permissions.");
      setMode('upload');
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResult(null);
            setError(null);
            // Stop camera after capture
            if (stream) {
              stream.getTracks().forEach(track => track.stop());
              setStream(null);
            }
          }
        }, 'image/jpeg', 0.9);
      }
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (mode === 'camera') startCamera();
  };

  const handleRecognize = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/recognize`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Failed to connect to the AI engine.');
    } finally {
      setLoading(false);
    }
  };

  const neuralPaths = [
    "M-100,200 C150,100 350,300 500,250 S850,100 1200,200",
    "M-100,500 C200,400 400,600 600,500 S900,400 1200,500",
    "M-100,800 C150,700 350,900 550,850 S850,700 1200,800",
    "M200,-100 C100,150 300,350 250,500 S100,850 200,1200",
    "M500,-100 C400,200 600,400 500,600 S400,900 500,1200",
    "M800,-100 C700,150 900,350 850,500 S700,850 800,1200"
  ];

  return (
    <div className="min-h-screen bg-gradient-mesh flex flex-col items-center justify-center p-4 md:p-8 relative overflow-hidden">
      {/* Dynamic Background Elements */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="orb w-[600px] h-[600px] bg-brand-primary/10 -top-48 -left-24 animate-float" />
        <div className="orb w-[500px] h-[500px] bg-brand-secondary/10 top-1/2 -right-24 animate-float-delayed" />
        <div className="orb w-[400px] h-[400px] bg-brand-primary/5 bottom-0 left-1/4 animate-pulse-slow" />
        
        {/* Neural Network SVG Background */}
        <svg className="absolute inset-0 w-full h-full opacity-30" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="line-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="var(--color-brand-primary)" stopOpacity="0.1" />
              <stop offset="100%" stopColor="var(--color-brand-secondary)" stopOpacity="0.1" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          
          <g filter="url(#glow)">
            {neuralPaths.map((path, i) => (
              <path key={i} d={path} className="stroke-brand-primary/20 fill-none stroke-[1]" />
            ))}
            <g className="stroke-brand-primary/40 fill-none stroke-[1.5] stroke-dasharray-[10,100] animate-[flow-line_10s_linear_infinite]">
              {neuralPaths.map((path, i) => (
                <path key={`flow-${i}`} d={path} />
              ))}
            </g>
          </g>

          {[
            {x: 150, y: 100, d: 0}, {x: 500, y: 250, d: 1}, {x: 850, y: 100, d: 0.5},
            {x: 200, y: 400, d: 1.5}, {x: 600, y: 500, d: 2}, {x: 900, y: 400, d: 0.8},
            {x: 150, y: 700, d: 1.2}, {x: 550, y: 850, d: 0.3}, {x: 850, y: 700, d: 1.7},
            {x: 250, y: 500, d: 0.4}, {x: 500, y: 600, d: 1.1}, {x: 850, y: 500, d: 1.9}
          ].map((node, i) => (
            <g key={i}>
              <circle cx={node.x} cy={node.y} r="4" className="fill-brand-primary/40 animate-pulse-node" style={{ animationDelay: `${node.d}s` }} />
              <circle cx={node.x} cy={node.y} r="1.5" className="fill-brand-primary shadow-[0_0_8px_rgba(244,63,94,0.8)]" />
            </g>
          ))}
        </svg>

        {[...Array(20)].map((_, i) => (
          <div 
            key={i}
            className="particle"
            style={{
              width: `${Math.random() * 2 + 1}px`,
              height: `${Math.random() * 2 + 1}px`,
              left: `${Math.random() * 100}%`,
              bottom: `-20px`,
              animationDuration: `${Math.random() * 15 + 15}s`,
              animationDelay: `${Math.random() * 30}s`,
              backgroundColor: i % 2 === 0 ? 'var(--color-brand-primary)' : 'var(--color-brand-secondary)',
              opacity: 0.2
            }}
          />
        ))}
      </div>

      {/* Header Section */}
      <div className="text-center mb-10 space-y-4 animate-in fade-in slide-in-from-top duration-1000 relative z-10">
        {/* <div className="inline-flex items-center space-x-2 px-4 py-1.5 rounded-full bg-brand-primary/10 border border-brand-primary/20 text-brand-primary text-[10px] font-bold uppercase tracking-[0.2em]">
          <Activity className="w-3.5 h-3.5 animate-pulse" />
          <span>Neural Pulse Synchronized</span>
        </div> */}
        <h1 className="text-5xl md:text-7xl font-black tracking-tighter">
          <span className="text-white drop-shadow-2xl">Neuro</span>
          <span className="text-gradient drop-shadow-[0_0_20px_rgba(244,63,94,0.4)]">Vision</span>
        </h1>
        
        {/* Mode Switcher */}
        <div className="flex items-center justify-center space-x-2 mt-6">
          <button 
            onClick={() => setMode('upload')}
            className={cn(
              "px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all",
              mode === 'upload' ? "bg-brand-primary text-white" : "glass text-slate-500 hover:text-slate-300"
            )}
          >
            File Upload
          </button>
          <button 
            onClick={() => { setMode('camera'); startCamera(); }}
            className={cn(
              "px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all",
              mode === 'camera' ? "bg-brand-primary text-white" : "glass text-slate-500 hover:text-slate-300"
            )}
          >
            Live Camera
          </button>
        </div>
      </div>

      {/* Main Container */}
      <div className="w-full max-w-5xl grid md:grid-cols-2 gap-8 items-stretch relative z-10">
        
        {/* Input Section */}
        <div className="flex flex-col space-y-6 animate-in fade-in slide-in-from-left duration-700 delay-200">
          <div 
            className={cn(
              "glass rounded-[2.5rem] p-10 transition-all duration-700 relative overflow-hidden flex-1",
              !previewUrl && mode === 'upload' && "py-28 flex flex-col items-center justify-center border-dashed border-2 border-brand-primary/10 hover:border-brand-primary/40 cursor-pointer group hover:bg-brand-primary/[0.03]",
              previewUrl && "bg-white/[0.02]"
            )}
            onClick={() => !previewUrl && mode === 'upload' && fileInputRef.current?.click()}
          >
            {mode === 'camera' && !previewUrl ? (
              <div className="relative rounded-3xl overflow-hidden aspect-video bg-black/40 border border-white/5 shadow-2xl">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  className="w-full h-full object-cover grayscale-[0.3] brightness-110"
                />
                <div className="absolute inset-0 border-2 border-brand-primary/20 pointer-events-none" />
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2">
                  <button 
                    onClick={capturePhoto}
                    className="w-16 h-16 rounded-full bg-white/10 backdrop-blur-md border-2 border-white/40 flex items-center justify-center hover:bg-brand-primary/20 hover:border-brand-primary transition-all group"
                  >
                    <div className="w-12 h-12 rounded-full bg-white group-hover:bg-brand-primary transition-colors" />
                  </button>
                </div>
                <div className="absolute top-4 left-4 flex items-center space-x-2 text-[10px] font-bold text-brand-primary uppercase tracking-widest">
                  <div className="w-2 h-2 rounded-full bg-brand-primary animate-pulse" />
                  <span>Live Feed</span>
                </div>
              </div>
            ) : !previewUrl ? (
              <>
                <div className="w-24 h-24 rounded-3xl bg-brand-primary/10 flex items-center justify-center mb-8 group-hover:scale-110 group-hover:rotate-3 group-hover:bg-brand-primary/20 transition-all duration-500 shadow-inner">
                  <Upload className="w-12 h-12 text-brand-primary animate-pulse" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-3 text-center">Initialize Identity</h3>
                <p className="text-slate-500 text-center text-sm max-w-[200px] font-light">
                  Drop identity frame or tap to scan high-res biometric input
                </p>
              </>
            ) : (
              <div className="relative overflow-hidden rounded-3xl border border-white/5 aspect-square md:aspect-auto group/preview shadow-2xl">
                <img 
                  src={previewUrl} 
                  alt="Identity Preview" 
                  className="w-full h-full object-cover transition-transform duration-1000 group-hover/preview:scale-110"
                />
                
                {loading && (
                  <div className="absolute inset-0 z-10">
                    <div className="absolute top-0 left-0 right-0 h-1.5 bg-brand-primary shadow-[0_0_30px_rgba(244,63,94,1)] animate-scan" />
                    <div className="absolute inset-0 bg-brand-primary/20 backdrop-blur-[1px]" />
                  </div>
                )}

                <div className="absolute top-4 right-4 flex space-x-2">
                  <button 
                    onClick={(e) => { e.stopPropagation(); reset(); }}
                    className="p-3 rounded-full glass bg-black/60 text-white/80 hover:text-white hover:scale-110 transition-all"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
            <input 
              type='file' 
              ref={fileInputRef}
              className="hidden" 
              onChange={handleFileChange} 
              accept="image/*" 
            />
          </div>

          <button
            onClick={handleRecognize}
            disabled={!selectedFile || loading}
            className={cn(
              "w-full py-6 rounded-2xl font-bold text-lg flex items-center justify-center space-x-4 transition-all duration-500 overflow-hidden relative group",
              !selectedFile || loading 
              ? 'bg-slate-900/50 text-slate-700 cursor-not-allowed border border-white/5' 
              : 'bg-gradient-to-r from-brand-primary to-brand-secondary text-white shadow-[0_0_40px_rgba(244,63,94,0.25)] hover:shadow-[0_0_60px_rgba(244,63,94,0.4)] hover:-translate-y-1 active:scale-[0.98]'
            )}
          >
            {loading ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                <span className="tracking-widest uppercase text-xs font-black">Synthesizing Layers...</span>
              </>
            ) : (
              <>
                <Scan className="w-6 h-6 group-hover:scale-125 group-hover:rotate-12 transition-transform" />
                <span>Initialize Identification</span>
              </>
            )}
          </button>
        </div>

        {/* Results Section */}
        <div className="flex flex-col space-y-6 animate-in fade-in slide-in-from-right duration-700 delay-400">
          {error && (
            <div className="glass bg-red-500/10 border-red-500/20 p-8 rounded-[2.5rem] flex items-start space-x-5 animate-in fade-in zoom-in duration-500">
              <div className="p-4 bg-red-500/20 rounded-2xl shadow-lg">
                <AlertCircle className="w-8 h-8 text-red-500" />
              </div>
              <div>
                <h4 className="font-bold text-red-500 text-xl font-display">System Failure</h4>
                <p className="text-red-200/60 text-sm mt-2 leading-relaxed font-light">{error}</p>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="glass p-12 rounded-[2.5rem] flex-1 flex flex-col items-center justify-center text-center space-y-8 border-dashed border-2 border-white/[0.02]">
              <div className="relative">
                <div className="w-24 h-24 rounded-full border border-brand-primary/10 flex items-center justify-center animate-pulse-slow">
                  <Camera className="w-10 h-10 text-brand-primary/20" />
                </div>
                <div className="absolute inset-0 border-2 border-brand-primary/20 rounded-full animate-ping opacity-10" />
              </div>
              <div className="space-y-2">
                <p className="text-slate-400 text-sm font-medium tracking-wide">Awaiting Signal</p>
                <p className="text-slate-600 text-xs font-light max-w-[200px]">
                  Provide biometric visual data to activate neural mapping sequences.
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="glass p-10 rounded-[2.5rem] flex-1 space-y-10 animate-in fade-in zoom-in duration-700 relative overflow-hidden group">
              <div className="absolute -top-32 -right-32 w-80 h-80 bg-brand-primary/10 rounded-full blur-[120px] group-hover:bg-brand-primary/20 transition-colors duration-1000" />
              
              <div className="flex items-center justify-between relative z-10">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-brand-primary animate-pulse" />
                  <h3 className="text-lg font-bold text-white/90 font-display">Neural Inference</h3>
                </div>
                {result.success ? (
                  <span className="px-4 py-1.5 rounded-full bg-brand-primary/10 text-brand-primary text-[10px] font-black border border-brand-primary/20 flex items-center space-x-2 tracking-[0.2em] uppercase">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Identified</span>
                  </span>
                ) : (
                  <span className="px-4 py-1.5 rounded-full bg-slate-800 text-slate-400 text-[10px] font-black border border-white/5 flex items-center space-x-2 tracking-[0.2em] uppercase">
                    <XCircle className="w-3.5 h-3.5" />
                    <span>Unknown</span>
                  </span>
                )}
              </div>

              {result.success ? (
                <div className="space-y-8 relative z-10">
                  <div className="flex items-center space-x-8">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-brand-primary to-brand-secondary flex items-center justify-center shadow-[0_25px_60px_rgba(244,63,94,0.35)] relative group/avatar">
                      <User className="w-12 h-12 text-white group-hover/avatar:scale-110 transition-transform" />
                      <div className="absolute -bottom-3 -right-3 w-10 h-10 bg-slate-950 rounded-2xl border border-white/10 flex items-center justify-center shadow-2xl">
                        <ShieldCheck className="w-5 h-5 text-brand-primary" />
                      </div>
                    </div>
                    <div>
                      <p className="text-slate-500 text-[10px] uppercase tracking-[0.4em] font-black">Target Profile</p>
                      <h2 className="text-4xl font-black text-white mt-2 tracking-tight">{result.prediction}</h2>
                    </div>
                  </div>

                  <div className="space-y-5 bg-white/[0.03] p-7 rounded-[2rem] border border-white/5 shadow-inner">
                    <div className="flex justify-between items-end">
                      <div>
                        <p className="text-slate-500 text-[10px] uppercase tracking-widest font-black">Recognition Match</p>
                        <p className="text-4xl font-black text-white mt-1">{(result.confidence! * 100).toFixed(1)}<span className="text-brand-primary text-2xl">%</span></p>
                      </div>
                      <div className="text-right">
                        <Cpu className="w-5 h-5 text-brand-primary/40 ml-auto mb-1" />
                        <span className="text-[10px] text-slate-600 font-mono">LATENT_MATCH: {result.confidence?.toFixed(4)}</span>
                      </div>
                    </div>
                    <div className="h-4 w-full bg-white/[0.03] rounded-full overflow-hidden p-1 border border-white/5">
                      <div 
                        className="h-full bg-gradient-to-r from-brand-primary via-rose-400 to-brand-secondary rounded-full shadow-[0_0_20px_rgba(244,63,94,0.6)] transition-all duration-[2s] ease-out"
                        style={{ width: `${result.confidence! * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="glass bg-white/[0.01] rounded-3xl p-5 border border-white/5 hover:bg-white/[0.04] transition-all group/item">
                      <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1 group-hover/item:text-brand-primary transition-colors">Vector Dist</p>
                      <p className="text-white font-bold text-base">0.421</p>
                    </div>
                    <div className="glass bg-white/[0.01] rounded-3xl p-5 border border-white/5 hover:bg-white/[0.04] transition-all group/item">
                      <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1 group-hover/item:text-brand-primary transition-colors">Processing</p>
                      <p className="text-white font-bold text-base">GPU_ACCEL</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-14 space-y-8 relative z-10">
                  <div className="relative mx-auto w-24 h-24">
                    <div className="absolute inset-0 bg-brand-primary/20 rounded-full blur-2xl animate-pulse" />
                    <div className="relative w-24 h-24 bg-brand-primary/5 rounded-[2rem] border border-brand-primary/10 flex items-center justify-center shadow-inner">
                      <AlertCircle className="w-10 h-10 text-brand-primary/50" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <p className="text-white font-black text-2xl tracking-tight font-display">Identity Mismatch</p>
                    <p className="text-slate-500 text-sm px-12 leading-relaxed font-light">
                      The neural signature extracted from this frame does not correlate with any verified biometric profiles in our dataset.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Hidden Canvas for Capturing Frames */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Footer */}
      <div className="mt-20 flex flex-wrap justify-center gap-10 text-slate-700 relative z-10 opacity-40 hover:opacity-100 transition-opacity duration-700">
        {[
          { label: 'Layer: FastAPI', icon: Activity },
          { label: 'Net: FaceNet-3D', icon: Cpu },
          { label: 'UI: Neuro-Flux', icon: Zap }
        ].map((tag, i) => (
          <div key={i} className="flex items-center space-x-3 group cursor-default">
            <tag.icon className="w-3.5 h-3.5 text-brand-primary group-hover:scale-125 transition-transform" />
            <span className="text-[10px] uppercase tracking-[0.4em] font-black group-hover:text-slate-400 transition-colors">{tag.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
