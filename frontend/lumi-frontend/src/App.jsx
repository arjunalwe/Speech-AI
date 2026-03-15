import React, { useState, useRef, useEffect } from 'react';
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const BACKEND_URL = "http://localhost:8000/analyze";
const SILENCE_THRESHOLD = 0.05;

export default function LumiApp() {
  const [view, setView] = useState('home');
  const [profile, setProfile] = useState({ userId: "user_123", name: "", age: "", selectedDisorders: [] });
  const [exercises, setExercises] = useState([]);
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0);
  const [lastResult, setLastResult] = useState(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const videoRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const telemetryRef = useRef([]);
  
  const isRecordingRef = useRef(false);

  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const loopsRef = useRef({ face: null, audio: null });
  const metricsRef = useRef({ maxVolume: 0, longestSilence: 0, silenceStart: 0, startTime: 0 });

  const initMediaPipe = async () => {
    if (faceLandmarkerRef.current) return;
    try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm");
        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
        runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true
        });
    } catch (err) {
        console.error("Failed to load MediaPipe:", err);
        alert("Could not load facial tracking. Please refresh the page.");
    }
  };

  const getBlendshape = (result, name) => {
    if (!result.faceBlendshapes?.length) return 0.0;
    const match = result.faceBlendshapes[0].categories.find(c => c.categoryName === name);
    return match ? Number(match.score.toFixed(4)) : 0.0;
  };

  const trackFace = () => {
    if (!isRecordingRef.current) return; 
    
    if (faceLandmarkerRef.current && videoRef.current && videoRef.current.readyState >= 2) {
      const now = performance.now();
      const result = faceLandmarkerRef.current.detectForVideo(videoRef.current, now);
      if (result.faceLandmarks?.length > 0) {
        telemetryRef.current.push({
          time_ms: Math.round(now - metricsRef.current.startTime),
          shapes: {
            mouthPucker: getBlendshape(result, "mouthPucker"),
            mouthSmileLeft: getBlendshape(result, "mouthSmileLeft"),
            mouthSmileRight: getBlendshape(result, "mouthSmileRight"),
            jawOpen: getBlendshape(result, "jawOpen"),
            browInnerUp: getBlendshape(result, "browInnerUp")
          },
          pose: { pitch: 0, yaw: 0, roll: 0 }
        });
      }
    }
    loopsRef.current.face = requestAnimationFrame(trackFace);
  };

  const trackAudio = () => {
    if (!isRecordingRef.current) return;
    
    if (analyserRef.current) {
      const dataArray = new Float32Array(analyserRef.current.frequencyBinCount);
      analyserRef.current.getFloatTimeDomainData(dataArray);

      let maxInFrame = 0;
      for (let i = 0; i < dataArray.length; i++) {
        if (Math.abs(dataArray[i]) > maxInFrame) maxInFrame = Math.abs(dataArray[i]);
      }

      if (maxInFrame > metricsRef.current.maxVolume) metricsRef.current.maxVolume = maxInFrame;

      const now = performance.now();
      if (maxInFrame < SILENCE_THRESHOLD) {
        metricsRef.current.longestSilence = Math.max(metricsRef.current.longestSilence, now - metricsRef.current.silenceStart);
      } else {
        metricsRef.current.silenceStart = now;
      }
    }
    loopsRef.current.audio = requestAnimationFrame(trackAudio);
  };

  const stopHardware = () => {
      isRecordingRef.current = false; 
      if (loopsRef.current.face) cancelAnimationFrame(loopsRef.current.face);
      if (loopsRef.current.audio) cancelAnimationFrame(loopsRef.current.audio);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
          streamRef.current.getTracks().forEach(t => t.stop());
      }
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
          audioCtxRef.current.close();
      }
  }

  const handleRecordToggle = async () => {
    if (isRecording) {
      setIsRecording(false);
      setIsProcessing(true);
      stopHardware();
    } else {
      setIsProcessing(true);
      await initMediaPipe();

      try {
        streamRef.current = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        if (videoRef.current) {
            videoRef.current.srcObject = streamRef.current;
            await videoRef.current.play();
        }
      } catch (err) {
          alert("Microphone/Camera access denied.");
          setIsProcessing(false);
          return;
      }

      audioChunksRef.current = [];
      telemetryRef.current = [];
      metricsRef.current = { maxVolume: 0, longestSilence: 0, silenceStart: performance.now(), startTime: performance.now() };

      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioCtxRef.current.createAnalyser();
      const source = audioCtxRef.current.createMediaStreamSource(streamRef.current);
      source.connect(analyserRef.current);

      mediaRecorderRef.current = new MediaRecorder(streamRef.current);
      mediaRecorderRef.current.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        
        reader.onloadend = async () => {
          const base64Audio = reader.result.split(",")[1];
          const currentTemplate = exercises[currentExerciseIndex].backend_template;

          const payload = {
            user_id: profile.userId,
            name: profile.name,
            age: parseInt(profile.age),
            session_mode: currentTemplate.session_mode,
            session_type: currentTemplate.session_type,
            target_word: currentTemplate.target_word,
            target_phoneme: currentTemplate.target_phoneme,
            audio_base64: base64Audio,
            frames: telemetryRef.current,
            audio_metrics: {
              max_volume_spike: parseFloat(metricsRef.current.maxVolume.toFixed(3)),
              longest_silence_ms: Math.round(metricsRef.current.longestSilence)
            }
          };

          try {
            const response = await fetch(BACKEND_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload)
            });
            const data = await response.json();

            setLastResult(data);
            setView('feedback');
            
          } catch (err) {
            console.error("Backend failed:", err);
            alert("Connection to Lumi backend failed.");
          }
          setIsProcessing(false);
        };
      };

      mediaRecorderRef.current.start();
      isRecordingRef.current = true; 
      setIsRecording(true);
      setIsProcessing(false);
      trackFace();
      trackAudio();
    }
  };

  const toggleDisorder = (id) => {
    setProfile(prev => ({
      ...prev, selectedDisorders: prev.selectedDisorders.includes(id)
        ? prev.selectedDisorders.filter(item => item !== id) : [...prev.selectedDisorders, id]
    }));
  };
  
  const handleProfileChange = (e) => setProfile(prev => ({ ...prev, [e.target.name]: e.target.value }));
  
  const handleReturnToDashboard = () => {
      stopHardware();
      setView('dashboard');
  }

  const currentExercise = exercises[currentExerciseIndex];

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 p-4 flex items-center justify-center">
      <div className="bg-white w-full max-w-4xl min-h-[700px] rounded-3xl shadow-2xl border border-slate-100 overflow-hidden flex flex-col">

        {view !== 'home' && (
          <header className="px-8 py-6 border-b flex justify-between items-center bg-white z-10">
            <h1 
                className="text-2xl font-black text-indigo-600 tracking-tighter cursor-pointer"
                onClick={handleReturnToDashboard}
            >
                LUMI
            </h1>
            <div className="flex gap-4 items-center">
              <span className="text-sm font-medium text-slate-500">
                {profile.name ? `${profile.name} (Age ${profile.age})` : profile.userId}
              </span>
              <div className="w-8 h-8 bg-indigo-100 rounded-full border-2 border-indigo-200"></div>
            </div>
          </header>
        )}

        <main className="flex-1 p-8 overflow-y-auto flex flex-col justify-center">

          {view === 'home' && (
            <div className="text-center py-20 animate-in fade-in">
              <h1 className="text-7xl font-black text-indigo-600 mb-6">Lumi</h1>
              <p className="text-xl text-slate-500 mb-10">Your AI Speech Language Pathologist</p>
              <button onClick={() => setView('profile_setup')} className="bg-indigo-600 text-white px-12 py-4 rounded-full font-bold text-lg hover:bg-indigo-700 shadow-xl transition">Get Started</button>
            </div>
          )}

          {view === 'profile_setup' && (
            <div className="max-w-md mx-auto w-full animate-in fade-in">
              <h2 className="text-3xl font-bold mb-8 text-center">Meet the patient.</h2>
              <input type="text" name="name" value={profile.name} onChange={handleProfileChange} className="w-full p-4 border-2 rounded-xl mb-4 bg-slate-50 focus:bg-white focus:border-indigo-600 transition outline-none" placeholder="First Name (e.g. Alex)" />
              <input type="number" name="age" value={profile.age} onChange={handleProfileChange} className="w-full p-4 border-2 rounded-xl mb-8 bg-slate-50 focus:bg-white focus:border-indigo-600 transition outline-none" placeholder="Age (e.g. 6)" />
              <button onClick={() => setView('register')} disabled={!profile.name || !profile.age} className="w-full bg-indigo-600 text-white py-4 rounded-xl font-bold disabled:opacity-50 hover:bg-indigo-700 transition shadow-lg">Next Step</button>
            </div>
          )}

          {view === 'register' && (
            <div className="max-w-md mx-auto w-full animate-in fade-in">
              <h2 className="text-3xl font-bold mb-8 text-center">Select Focus Areas</h2>
              {['articulation', 'stuttering', 'pragmatics'].map(id => (
                <div key={id} onClick={() => toggleDisorder(id)} className={`p-5 border-2 rounded-2xl mb-4 cursor-pointer transition ${profile.selectedDisorders.includes(id) ? 'border-indigo-600 bg-indigo-50 shadow-sm' : 'border-slate-200 hover:border-indigo-300'}`}>
                  <p className="font-bold text-lg capitalize text-center">{id}</p>
                </div>
              ))}
              <button onClick={() => setView('dashboard')} disabled={profile.selectedDisorders.length === 0} className="w-full mt-6 bg-indigo-600 text-white py-4 rounded-xl font-bold disabled:opacity-50 hover:bg-indigo-700 transition shadow-lg">Generate Path</button>
            </div>
          )}

          {view === 'dashboard' && (
            <div className="max-w-2xl mx-auto w-full animate-in fade-in">
              <h2 className="text-3xl font-bold mb-8 text-slate-800">Your Practice Dashboard</h2>
              {profile.selectedDisorders.map(id => (
                <button
                  key={id}
                  onClick={() => {
                    const starterWord = id === "articulation" ? "Rabbit" : "How was your day today?";
                    const starterPhoneme = id === "articulation" ? "r" : "";

                    setExercises([{
                      frontend_display: {
                        visual_audio_instructions: `Hi ${profile.name}! Let's practice some ${id} exercises together. I am going to listen to your voice and watch how your mouth moves.`,
                        system_question: starterWord
                      },
                      backend_template: {
                        session_mode: "practice",
                        session_type: id,
                        target_word: starterWord,
                        target_phoneme: starterPhoneme
                      }
                    }]);
                    setCurrentExerciseIndex(0);
                    setView('practice');
                  }}
                  className="w-full bg-white border-2 border-slate-100 p-6 rounded-2xl mb-4 flex justify-between items-center hover:border-indigo-200 hover:shadow-lg transition group"
                >
                  <span className="font-bold text-xl uppercase text-slate-700 group-hover:text-indigo-600 transition">{id} Set</span>
                  <span className="bg-indigo-50 text-indigo-600 px-6 py-3 rounded-full font-bold group-hover:bg-indigo-600 group-hover:text-white transition">Play →</span>
                </button>
              ))}
            </div>
          )}

          {view === 'practice' && currentExercise && (
            <div className="max-w-xl mx-auto w-full animate-in fade-in flex flex-col items-center text-center">
              
              <div className="bg-indigo-50/50 border border-indigo-100 p-8 rounded-3xl w-full mb-8 shadow-sm">
                  <p className="text-slate-600 mb-4 text-lg font-medium">{currentExercise.frontend_display.visual_audio_instructions}</p>
                  <p className="text-5xl font-black text-indigo-900 tracking-tight">{currentExercise.frontend_display.system_question}</p>
              </div>

              <div className="relative w-80 h-80 bg-slate-900 rounded-[2.5rem] overflow-hidden mb-10 shadow-2xl border-8 border-white ring-4 ring-indigo-50">
                  <video
                      ref={videoRef}
                      className={`absolute inset-0 w-full h-full object-cover transform scale-x-[-1] transition-opacity duration-500 ${isRecording ? 'opacity-100' : 'opacity-50 blur-sm'}`}
                      muted
                      playsInline
                  />
                  {!isRecording && !isProcessing && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm">
                          <span className="text-white font-bold text-lg tracking-widest uppercase">Ready</span>
                      </div>
                  )}
                  {isProcessing && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-indigo-900/80 backdrop-blur-md animate-pulse">
                          <div className="w-12 h-12 border-4 border-white/30 border-t-white rounded-full animate-spin mb-4"></div>
                          <span className="text-white font-bold tracking-widest uppercase text-sm">Analyzing Audio & Video</span>
                      </div>
                  )}
              </div>

              <button
                  onClick={handleRecordToggle}
                  disabled={isProcessing}
                  className={`w-full max-w-xs py-5 rounded-full font-black text-xl text-white shadow-2xl transition-all duration-300 transform active:scale-95 ${
                      isRecording ? 'bg-rose-500 hover:bg-rose-600 animate-pulse ring-4 ring-rose-200' :
                      isProcessing ? 'bg-slate-300 cursor-not-allowed text-slate-500 shadow-none' : 'bg-indigo-600 hover:bg-indigo-700 hover:-translate-y-1'
                  }`}
              >
                  {isProcessing ? 'Processing...' : isRecording ? 'Stop Recording' : 'Start Recording'}
              </button>
            </div>
          )}

          {view === 'feedback' && lastResult && (
            <div className="max-w-xl mx-auto w-full animate-in fade-in flex flex-col items-center text-center">
              
              <div className={`p-8 rounded-3xl w-full mb-10 shadow-sm border-4 ${
                  lastResult.status === 'success' ? 'bg-emerald-50 border-emerald-100' : 'bg-orange-50 border-orange-100'
                }`}
              >
                  <h2 className={`text-4xl font-black mb-6 ${
                    lastResult.status === 'success' ? 'text-emerald-600' : 'text-orange-600'
                  }`}>
                      {lastResult.status === 'success' ? 'Awesome Job!' : 'Good Effort!'}
                  </h2>
                  
                  {/* PULLING YOUR CUSTOM CONVERSATIONAL FEEDBACK HERE */}
                  <p className="text-slate-700 text-xl font-medium leading-relaxed mb-6">
                      {lastResult.conversational_feedback}
                  </p>

                  <div className="flex gap-4 justify-center text-xs font-bold text-slate-500 uppercase tracking-wider">
                      <span className="bg-white px-4 py-2 rounded-full shadow-sm border">
                        Audio: {lastResult.diagnostic_data?.audio_analysis?.audio_insight}
                      </span>
                      <span className="bg-white px-4 py-2 rounded-full shadow-sm border">
                        Visual: {lastResult.diagnostic_data?.visual_analysis?.geometric_flaw || "Perfect Form"}
                      </span>
                  </div>
              </div>

              <div className="flex flex-col gap-4 w-full max-w-xs">
                  <button 
                      onClick={() => {
                          if (lastResult.next_turn) {
                              setExercises(prev => [...prev, lastResult.next_turn]);
                              setCurrentExerciseIndex(prev => prev + 1);
                          }
                          setView('practice');
                      }}
                      className="w-full py-4 rounded-full font-bold text-lg bg-indigo-600 text-white hover:bg-indigo-700 transition shadow-xl hover:-translate-y-1"
                  >
                      Move On to Next Word
                  </button>
                  
                  <button 
                      onClick={() => setView('practice')}
                      className="w-full py-4 rounded-full font-bold text-lg bg-slate-100 text-slate-700 hover:bg-slate-200 transition"
                  >
                      Redo This Word
                  </button>
                  
                  <button 
                      onClick={handleReturnToDashboard}
                      className="w-full py-4 rounded-full font-bold text-lg text-slate-400 hover:text-slate-600 transition"
                  >
                      Quit for the Day
                  </button>
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}