import React, { useState, useRef, useEffect } from 'react';

// Lumi - Clinical Speech Engine
export default function LumiApp() {
  const [view, setView] = useState('home');
  const [profile, setProfile] = useState({
    userId: "user_123",
    age: 6,
    selectedDisorders: [] // Now an array for multi-select
  });

  const toggleDisorder = (id) => {
    setProfile(prev => ({
      ...prev,
      selectedDisorders: prev.selectedDisorders.includes(id)
        ? prev.selectedDisorders.filter(item => item !== id)
        : [...prev.selectedDisorders, id]
    }));
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 p-4 flex items-center justify-center">
      <div className="bg-white w-full max-w-4xl min-h-[700px] rounded-3xl shadow-2xl border border-slate-100 overflow-hidden flex flex-col">
        
        {/* Navigation / Header */}
        {view !== 'home' && (
          <header className="px-8 py-6 border-b flex justify-between items-center">
            <h1 className="text-2xl font-black text-indigo-600 tracking-tighter">LUMI</h1>
            <div className="flex gap-4 items-center">
              <span className="text-sm font-medium text-slate-500">Child: {profile.userId}</span>
              <div className="w-8 h-8 bg-indigo-100 rounded-full border-2 border-indigo-200"></div>
            </div>
          </header>
        )}

        <main className="flex-1 p-8 overflow-y-auto">
          
          {/* VIEW: HOME */}
          {view === 'home' && (
            <div className="text-center py-20 animate-in fade-in slide-in-from-bottom-4">
              <h1 className="text-7xl font-black text-indigo-600 mb-6">Lumi</h1>
              <p className="text-xl text-slate-500 max-w-md mx-auto mb-12">
                The world's first multimodal AI speech therapist for children.
              </p>
              <button 
                onClick={() => setView('register')}
                className="bg-indigo-600 text-white px-12 py-4 rounded-full font-bold text-lg hover:bg-indigo-700 transition shadow-xl shadow-indigo-200"
              >
                Get Started
              </button>
            </div>
          )}

          {/* VIEW: REGISTRATION / MULTI-SELECT QUIZ */}
          {view === 'register' && (
            <div className="max-w-md mx-auto">
              <h2 className="text-3xl font-bold mb-2">Help us help them.</h2>
              <p className="text-slate-500 mb-8">Select all areas where your child needs support:</p>
              
              <div className="space-y-4">
                {[
                  { id: 'articulation', label: "Articulation", desc: "Trouble with specific sounds like R, L, S" },
                  { id: 'stuttering', label: "Fluency", desc: "Repeating sounds or hesitations" },
                  { id: 'pragmatics', label: "Social", desc: "Tone of voice and facial expressions" }
                ].map(item => (
                  <div 
                    key={item.id}
                    onClick={() => toggleDisorder(item.id)}
                    className={`p-5 border-2 rounded-2xl cursor-pointer transition ${
                      profile.selectedDisorders.includes(item.id) 
                        ? 'border-indigo-600 bg-indigo-50/50' 
                        : 'border-slate-100 hover:border-slate-200'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-bold text-lg">{item.label}</p>
                        <p className="text-sm text-slate-500">{item.desc}</p>
                      </div>
                      <div className={`w-6 h-6 rounded-full border-2 ${
                        profile.selectedDisorders.includes(item.id) ? 'bg-indigo-600 border-indigo-600' : 'border-slate-200'
                      }`}></div>
                    </div>
                  </div>
                ))}
              </div>

              <button 
                onClick={() => setView('dashboard')}
                disabled={profile.selectedDisorders.length === 0}
                className="w-full mt-10 bg-indigo-600 text-white py-4 rounded-xl font-bold disabled:opacity-50"
              >
                Continue to Dashboard
              </button>
            </div>
          )}

          {/* VIEW: DASHBOARD (The Polish) */}
          {view === 'dashboard' && (
            <div className="animate-in fade-in duration-500">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                <div className="col-span-2 bg-indigo-600 rounded-3xl p-8 text-white shadow-lg shadow-indigo-100 relative overflow-hidden">
                   <div className="relative z-10">
                     <h3 className="text-3xl font-bold mb-2">Welcome back!</h3>
                     <p className="opacity-90 max-w-xs">Today's goal: Practice 3 words to unlock the next level.</p>
                   </div>
                   <div className="absolute -right-4 -bottom-4 w-32 h-32 bg-white opacity-10 rounded-full"></div>
                </div>
                <div className="bg-white border-2 border-slate-50 rounded-3xl p-6 flex flex-col justify-center items-center text-center">
                   <p className="text-4xl font-black text-indigo-600">82%</p>
                   <p className="text-sm font-bold text-slate-400 uppercase tracking-widest mt-2">Accuracy Score</p>
                </div>
              </div>

              <h3 className="text-xl font-bold mb-6">Today's Practice Path</h3>
              <div className="grid grid-cols-1 gap-4">
                {profile.selectedDisorders.map(id => (
                  <button 
                    key={id}
                    onClick={() => setView('practice')}
                    className="group bg-white border border-slate-100 p-6 rounded-2xl flex justify-between items-center hover:shadow-md transition text-left"
                  >
                    <div>
                      <p className="text-xs font-black text-indigo-400 uppercase mb-1">{id}</p>
                      <p className="text-lg font-bold text-slate-800">Introduction to /{id === 'articulation' ? 'r' : 's'}/</p>
                    </div>
                    <span className="bg-slate-100 group-hover:bg-indigo-600 group-hover:text-white p-3 rounded-full transition">→</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* VIEW: PRACTICE (Placeholders for Teammates) */}
          {view === 'practice' && (
            <div className="max-w-2xl mx-auto">
               <div className="flex justify-between items-center mb-8">
                  <button onClick={() => setView('dashboard')} className="text-slate-400 font-bold">← Back</button>
                  <span className="bg-amber-100 text-amber-700 px-4 py-1 rounded-full text-sm font-bold">Session Active</span>
               </div>
               
               <div id="ai-exercise-target" className="mb-10 text-center">
                  <p className="text-slate-400 uppercase font-black text-xs tracking-widest mb-2">Say the Word</p>
                  <h2 className="text-6xl font-black text-slate-900 tracking-tight">RABBIT</h2>
               </div>

               <div className="aspect-video bg-slate-900 rounded-[2rem] mb-8 shadow-2xl overflow-hidden">
                  {/* MediaPipe Video will be injected here */}
               </div>

               <div className="flex justify-center gap-4 mb-10">
                  <button className="bg-emerald-500 text-white w-20 h-20 rounded-full flex items-center justify-center shadow-lg hover:scale-105 transition">REC</button>
               </div>

               <div id="visualizer-target" className="bg-indigo-50 p-8 rounded-[2rem] border border-indigo-100 flex gap-6">
                  <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center text-4xl shadow-sm">🤖</div>
                  <div>
                    <h4 className="font-bold text-indigo-600 mb-2">Lumi's Feedback</h4>
                    <p className="text-slate-600 leading-relaxed">Wait for recording to see feedback...</p>
                  </div>
               </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}