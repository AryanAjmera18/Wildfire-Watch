import React, { useState, useEffect, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import Globe from './components/3d/Globe';
import { Activity, Upload, AlertTriangle, CheckCircle, Brain, Flame, BarChart2, Layers, Eye } from 'lucide-react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function App() {
  const [activeTab, setActiveTab] = useState('demo');
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('binary');
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await axios.get('http://127.0.0.1:8000/stats');
      setStats(res.data);
    } catch (err) {
      console.error("Error fetching stats:", err);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      setPrediction(null);
      setExplanation(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode);

    try {
      // Predict
      const predRes = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPrediction(predRes.data);

      // Explain (Grad-CAM)
      const expRes = await axios.post('http://127.0.0.1:8000/explain', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setExplanation(expRes.data);

    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative w-full h-screen bg-slate-900 overflow-hidden text-white flex flex-col">

      {/* 3D Background */}
      <div className="absolute inset-0 z-0 opacity-50">
        <Canvas camera={{ position: [0, 0, 5] }}>
          <Suspense fallback={null}>
            <Globe />
            <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
          </Suspense>
        </Canvas>
      </div>

      {/* Header */}
      <header className="relative z-10 flex justify-between items-center p-6 bg-slate-900/80 backdrop-blur-md border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-red-500/20 rounded-lg border border-red-500/50">
            <Flame className="w-6 h-6 text-red-500" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tighter">Wildfire Watch</h1>
            <p className="text-slate-400 text-xs">AI-Powered Aerial Surveillance</p>
          </div>
        </div>

        <nav className="flex gap-2 bg-slate-800/50 p-1 rounded-lg">
          {['demo', 'performance', 'methodology'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === tab ? 'bg-red-500 text-white shadow-lg' : 'text-slate-400 hover:text-white'
                }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 overflow-y-auto p-8">

        {/* LIVE DEMO TAB */}
        {activeTab === 'demo' && (
          <div className="flex gap-8 h-full">
            {/* Left: Controls */}
            <div className="w-1/3 flex flex-col gap-6">
              <div className="glass-panel p-6">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Upload className="w-5 h-5 text-blue-400" />
                  Input Source
                </h2>

                <div className="flex gap-2 mb-6">
                  <button
                    onClick={() => setMode('binary')}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${mode === 'binary' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-400'}`}
                  >
                    Binary
                  </button>
                  <button
                    onClick={() => setMode('severity')}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${mode === 'severity' ? 'bg-orange-600 text-white' : 'bg-slate-700 text-slate-400'}`}
                  >
                    Severity
                  </button>
                </div>

                <div className="border-2 border-dashed border-slate-600 rounded-xl p-8 flex flex-col items-center justify-center gap-4 hover:border-red-500/50 transition-colors cursor-pointer bg-slate-800/20">
                  <input
                    type="file"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="flex flex-col items-center cursor-pointer w-full">
                    <Upload className="w-10 h-10 text-slate-500 mb-2" />
                    <span className="text-slate-400 text-sm">Upload drone footage</span>
                  </label>
                </div>

                {file && (
                  <div className="mt-4 bg-slate-800/50 p-3 rounded-lg flex items-center justify-between">
                    <span className="text-xs truncate max-w-[150px]">{file.name}</span>
                    <button
                      onClick={handleUpload}
                      disabled={loading}
                      className="bg-red-600 hover:bg-red-500 px-4 py-1.5 rounded-md text-xs font-bold transition-colors disabled:opacity-50"
                    >
                      {loading ? 'Analyzing...' : 'RUN'}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Results */}
            <div className="w-2/3 flex flex-col gap-6">
              {prediction && (
                <div className="glass-panel p-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h2 className="text-2xl font-bold flex items-center gap-2">
                        {prediction.label}
                      </h2>
                      <p className="text-slate-400 text-sm">Confidence: {prediction.confidence.toFixed(2)}%</p>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${prediction.label.includes('No Fire') ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                      {prediction.model_used.toUpperCase()}
                    </div>
                  </div>

                  {explanation && (
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-slate-500 mb-2 uppercase">Original Input</p>
                        <img src={`data:image/png;base64,${explanation.original}`} className="rounded-lg w-full border border-slate-700" />
                      </div>
                      <div>
                        <p className="text-xs text-slate-500 mb-2 uppercase">AI Attention (Grad-CAM)</p>
                        <img src={`data:image/png;base64,${explanation.gradcam}`} className="rounded-lg w-full border border-slate-700" />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* PERFORMANCE TAB */}
        {activeTab === 'performance' && stats && (
          <div className="grid grid-cols-2 gap-6">
            <div className="glass-panel p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-purple-400" />
                Model Benchmarks
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats.benchmark}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="Model" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                    <Legend />
                    <Bar dataKey="Accuracy" fill="#8884d8" />
                    <Bar dataKey="F1 Score (Macro)" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="glass-panel p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5 text-pink-400" />
                Faithfulness (Perturbation Test)
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats.faithfulness}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" label={{ value: 'Mask %', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#94a3b8" label={{ value: 'Confidence Drop', angle: -90, position: 'insideLeft' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                    <Bar dataKey="drop" fill="#f43f5e" name="Confidence Drop" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* METHODOLOGY TAB */}
        {activeTab === 'methodology' && (
          <div className="grid grid-cols-2 gap-6">
            <div className="glass-panel p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Layers className="w-5 h-5 text-yellow-400" />
                Cluster-Assist Labeling
              </h2>
              <img src="http://127.0.0.1:8000/static/cluster_grid.jpg" className="rounded-lg w-full border border-slate-700" />
              <p className="mt-4 text-slate-400 text-sm">
                We used K-Means clustering to group thousands of images into 50 semantic clusters, enabling rapid semi-supervised labeling.
              </p>
            </div>

            <div className="glass-panel p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-green-400" />
                Robust Augmentation
              </h2>
              <img src="http://127.0.0.1:8000/static/augmentation_demo.jpg" className="rounded-lg w-full border border-slate-700" />
              <p className="mt-4 text-slate-400 text-sm">
                To prevent overfitting, we applied dynamic augmentations (rotation, color jitter, random erasing) during training.
              </p>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
