import { useState, ChangeEvent } from 'react';
import axios from 'axios';

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

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleRecognize = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/recognize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Failed to connect to the backend server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center py-10 px-4">
      <h1 className="text-4xl font-bold mb-8 text-blue-400">Real-Time Face Recognition</h1>
      
      <div className="bg-gray-800 p-8 rounded-xl shadow-2xl w-full max-w-2xl">
        <div className="flex flex-col items-center mb-6">
          <label className="w-full flex flex-col items-center px-4 py-6 bg-gray-700 rounded-lg shadow-lg tracking-wide border border-blue-500 cursor-pointer hover:bg-blue-600 transition-colors mb-4">
            <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
              <path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z"></path>
            </svg>
            <span className="mt-2 text-base leading-normal">Select an image</span>
            <input type='file' className="hidden" onChange={handleFileChange} accept="image/*" />
          </label>
          
          {previewUrl && (
            <div className="mt-4 relative group">
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="max-h-80 rounded-lg shadow-md border-2 border-gray-600"
              />
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all rounded-lg"></div>
            </div>
          )}
        </div>

        <button
          onClick={handleRecognize}
          disabled={!selectedFile || loading}
          className={`w-full py-3 rounded-lg font-bold text-lg transition-all ${
            !selectedFile || loading 
            ? 'bg-gray-600 cursor-not-allowed' 
            : 'bg-blue-600 hover:bg-blue-700 shadow-lg active:scale-95'
          }`}
        >
          {loading ? 'Processing...' : 'Identify Person'}
        </button>

        {error && (
          <div className="mt-6 p-4 bg-red-900 border border-red-700 text-red-200 rounded-lg">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-8 p-6 bg-gray-700 rounded-lg border-l-4 border-blue-500">
            {result.success ? (
              <div className="flex flex-col items-center">
                <p className="text-gray-400 text-sm uppercase tracking-widest mb-1">Identity Detected</p>
                <h2 className="text-3xl font-black text-blue-300 mb-2">{result.prediction}</h2>
                <div className="flex items-center space-x-2">
                  <span className="text-gray-300">Confidence:</span>
                  <span className="bg-blue-900 text-blue-200 px-3 py-1 rounded-full text-sm font-bold">
                    {(result.confidence! * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center">
                <p className="text-orange-400 font-bold">Recognition Failed</p>
                <p className="text-gray-300 mt-2">{result.error}</p>
              </div>
            )}
          </div>
        )}
      </div>
      
      <p className="mt-10 text-gray-500 text-sm">
        Powered by FastAPI + React + MTCNN + FaceNet
      </p>
    </div>
  );
}

export default App;
