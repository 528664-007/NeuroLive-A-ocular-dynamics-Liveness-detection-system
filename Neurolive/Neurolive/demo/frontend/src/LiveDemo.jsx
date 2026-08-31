import { motion } from "framer-motion";
import { Activity, Cpu, Zap, CheckCircle, AlertCircle, ArrowLeft, Terminal, Shield } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const CHALLENGE_NAMES = {
  blink_twice: "BLINK TWICE",
  saccade_left_right: "LOOK LEFT → RIGHT",
  saccade_up_down: "LOOK UP → DOWN",
};

export default function LiveDemo({
  videoRef,
  canvasRef,
  cameraReady,
  modelReady,
  session,
  challenge,
  challengeStatus,
  challengeMessage,
  blinkCount,
  gazeState,
  eventCount,
  backendEventCount,
  decision,
  error,
  startChallenge,
  submitResponse,
  restart
}) {
  const chartData = decision?.activity_profile?.map((value, index) => ({
    bin: index + 1,
    activity: value,
  })) || [];

  return (
    <div className="min-h-screen bg-[#050012] text-white p-4 md:p-8 font-sans overflow-x-hidden relative selection:bg-purple-500/30">
      {/* Dynamic Animated Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCI+CjxwYXRoIGQ9Ik0wIDQwaDQwVjBIMHoiIGZpbGw9Im5vbmUiLz4KPHBhdGggZD0iTTAgMGg0MHY0MEgweiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjEiIG9wYWNpdHk9IjAuMDIiLz4KPC9zdmc+')] mix-blend-screen"></div>
        <div className="absolute top-0 inset-x-0 h-[60vh] bg-gradient-to-b from-purple-900/10 via-cyan-900/5 to-transparent"></div>
      </div>

      <div className="max-w-[1600px] mx-auto relative z-10 flex flex-col gap-6">
        
        {/* Header Area */}
        <header className="flex justify-between items-center bg-white/5 backdrop-blur-xl border border-white/10 p-4 rounded-xl shadow-lg">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => window.location.reload()}
              className="p-2 hover:bg-white/10 rounded-full transition-colors border border-white/5"
            >
              <ArrowLeft className="w-5 h-5 text-white/70" />
            </button>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-purple-500 flex items-center justify-center shadow-[0_0_15px_rgba(124,58,237,0.5)]">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold tracking-widest text-lg bg-clip-text text-transparent bg-gradient-to-r from-white to-white/70">CANTHUSCORE LIVE</h1>
              <p className="font-mono text-[10px] text-cyan-400 tracking-[0.2em] uppercase">Intelligence Terminal v2.1</p>
            </div>
          </div>
          
          {/* Status Indicators */}
          <div className="flex gap-6 font-mono text-xs bg-black/40 px-4 py-2 rounded-full border border-white/5">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${cameraReady ? 'bg-cyan-400 shadow-[0_0_8px_rgba(0,229,255,0.8)] animate-pulse' : 'bg-red-500'}`}></span>
              <span className="text-white/80 font-bold uppercase tracking-widest">Sensor</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${modelReady ? 'bg-purple-400 shadow-[0_0_8px_rgba(168,85,247,0.8)] animate-pulse' : 'bg-amber-400'}`}></span>
              <span className="text-white/80 font-bold uppercase tracking-widest">Network</span>
            </div>
            <div className="flex items-center gap-2">
              <Shield className="w-3 h-3 text-emerald-400" />
              <span className="text-emerald-400 font-bold uppercase tracking-widest">Secure</span>
            </div>
          </div>
        </header>

        {error && (
          <div className="bg-red-500/20 border border-red-500/30 p-4 rounded-xl flex items-start gap-3 backdrop-blur-md">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <p className="font-mono text-sm text-red-100">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* LEFT: Camera Viewport */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white/5 border border-white/10 backdrop-blur-md p-1.5 rounded-2xl relative group shadow-xl">
              <div className="absolute top-4 left-4 z-20 flex gap-2">
                <span className="bg-black/60 backdrop-blur-xl text-[10px] font-bold tracking-widest uppercase px-3 py-1 border border-white/10 rounded-full text-cyan-400 shadow-[0_0_10px_rgba(0,229,255,0.3)] flex items-center">
                  <span className="inline-block w-1.5 h-1.5 bg-cyan-400 rounded-full mr-2 animate-pulse"></span>
                  Live Feed
                </span>
                {session && (
                  <span className="bg-purple-500/20 backdrop-blur-xl text-[10px] font-bold tracking-widest uppercase px-3 py-1 border border-purple-500/30 rounded-full text-purple-200">
                    Session Linked
                  </span>
                )}
              </div>
              
              <div className="relative rounded-xl overflow-hidden bg-black aspect-[4/3] border border-white/5">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="w-full h-full object-cover scale-x-[-1]"
                />
                <canvas
                  ref={canvasRef}
                  width="96"
                  height="64"
                  className="hidden"
                />
                
                {/* Advanced Targeting HUD */}
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                   <div className="w-[50%] h-[60%] border border-cyan-400/20 rounded-[30%] relative group-hover:border-cyan-400/40 transition-colors duration-700">
                     <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-cyan-400 rounded-tl-xl transition-all duration-300 group-hover:w-10 group-hover:h-10"></div>
                     <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-cyan-400 rounded-tr-xl transition-all duration-300 group-hover:w-10 group-hover:h-10"></div>
                     <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-cyan-400 rounded-bl-xl transition-all duration-300 group-hover:w-10 group-hover:h-10"></div>
                     <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-cyan-400 rounded-br-xl transition-all duration-300 group-hover:w-10 group-hover:h-10"></div>
                   </div>
                </div>

                {/* Simulated Console Logs */}
                <div className="absolute bottom-10 left-4 right-4 h-12 overflow-hidden pointer-events-none mask-image-gradient-t flex flex-col justify-end">
                   <motion.div 
                     animate={{ y: [20, -10, 20] }} 
                     transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                     className="font-mono text-[8px] text-cyan-400/80 tracking-widest uppercase leading-tight"
                   >
                     &gt; Initializing biometric scan...<br/>
                     &gt; Voxel buffer mapping complete.<br/>
                     &gt; Deep network ready.<br/>
                     &gt; Awaiting protocol signal...
                   </motion.div>
                </div>
                
                {/* HUD Overlay Details */}
                <div className="absolute bottom-4 left-4 right-4 flex justify-between text-[10px] font-mono text-white/50 tracking-widest px-2 bg-black/40 backdrop-blur-md py-1 rounded">
                  <span>LATENCY: 12ms</span>
                  <span>FPS: 60.0</span>
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="bg-white/5 border border-white/10 backdrop-blur-md p-6 rounded-2xl flex flex-col gap-4 shadow-xl">
              <h3 className="font-bold text-sm text-white/50 tracking-widest uppercase mb-2 flex items-center justify-between">
                <span>System Operations</span>
                <Terminal className="w-4 h-4 text-cyan-400" />
              </h3>
              <div className="flex gap-4">
                <button
                  onClick={startChallenge}
                  disabled={!cameraReady || !modelReady || challengeStatus === "capturing" || challengeStatus === "processing"}
                  className="flex-1 py-4 px-4 bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-bold text-sm tracking-widest uppercase rounded-xl hover:shadow-[0_0_20px_rgba(124,58,237,0.5)] transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
                >
                  <span className="group-hover:tracking-[0.25em] transition-all">Start Challenge</span>
                </button>
                <button
                  onClick={() => submitResponse(false)}
                  disabled={!session || challengeStatus === "processing" || challengeStatus === "completed" || challengeStatus === "idle"}
                  className="flex-1 py-4 px-4 bg-white/5 border border-white/10 font-bold tracking-widest uppercase text-sm rounded-xl hover:bg-white/10 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {challengeStatus === "processing" ? "Processing..." : "Force Submit"}
                </button>
              </div>
              <button
                onClick={restart}
                className="w-full py-4 px-4 border border-red-500/30 text-red-400 font-bold tracking-widest uppercase text-sm rounded-xl hover:bg-red-500/10 transition-colors"
              >
                Terminate Session
              </button>
            </div>
          </div>

          {/* CENTER: Challenge & Analysis */}
          <div className="lg:col-span-4 flex flex-col gap-6 h-full">
            <div className="bg-white/5 border border-white/10 backdrop-blur-md p-6 rounded-2xl flex-1 flex flex-col relative overflow-hidden shadow-xl">
              <div className="absolute -right-10 -top-10 text-white/5 pointer-events-none">
                <Activity className="w-48 h-48" />
              </div>
              
              <h2 className="font-bold tracking-widest text-sm text-white/50 uppercase mb-6 relative z-10 flex justify-between">
                Active Challenge
                <span className="text-[10px] text-purple-400 tracking-widest px-2 py-0.5 border border-purple-500/30 rounded-full bg-purple-500/10">STAGE 1</span>
              </h2>
              
              <div className="flex-1 flex flex-col items-center justify-center text-center gap-6 z-10">
                <div className="font-sans text-3xl font-bold tracking-wide text-white drop-shadow-md">
                  {challenge ? CHALLENGE_NAMES[challenge] || challenge : "AWAITING INIT"}
                </div>
                
                <div className={`font-bold text-sm tracking-widest uppercase px-6 py-3 rounded-full border-2 transition-colors duration-300 ${
                  challengeStatus === 'passed' ? 'border-emerald-500 text-emerald-400 bg-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.3)]' :
                  challengeStatus === 'failed' ? 'border-red-500 text-red-400 bg-red-500/20 shadow-[0_0_15px_rgba(239,68,68,0.3)]' :
                  challengeStatus === 'processing' ? 'border-cyan-400 text-cyan-400 bg-cyan-400/20 shadow-[0_0_15px_rgba(34,211,238,0.3)] animate-pulse' :
                  'border-white/20 text-white/70 bg-white/10'
                }`}>
                  {challengeMessage}
                </div>

                <div className="w-full bg-black/40 rounded-xl border border-white/5 p-4 mt-2">
                  {challenge === "blink_twice" && (
                    <div className="font-mono text-sm tracking-widest text-white/70 flex justify-between items-center">
                      <span>BLINKS DETECTED</span>
                      <span className="text-white font-bold text-lg">{blinkCount} <span className="text-white/30 text-sm">/ 2</span></span>
                    </div>
                  )}
                  
                  {challenge !== "blink_twice" && challenge && (
                    <div className="font-mono text-sm tracking-widest text-white/70 flex justify-between items-center">
                      <span>GAZE VECTOR</span>
                      <span className="text-cyan-400 font-bold bg-cyan-400/10 px-2 py-1 rounded">{gazeState}</span>
                    </div>
                  )}
                  {!challenge && (
                    <div className="font-mono text-[10px] tracking-widest text-white/30 uppercase text-center">
                      Protocol Unassigned
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Event Stream Stats */}
            <div className="bg-white/5 border border-white/10 backdrop-blur-md p-6 rounded-2xl shadow-xl">
              <h3 className="font-bold tracking-widest text-sm text-white/50 uppercase mb-4">Event Stream Telemetry</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-black/40 p-4 rounded-xl border border-white/5 relative overflow-hidden group">
                  <div className="absolute top-0 left-0 w-1 h-full bg-cyan-400"></div>
                  <div className="font-mono text-[10px] tracking-widest uppercase text-cyan-400 mb-1 pl-2">Local Events</div>
                  <div className="font-sans font-light text-3xl text-white pl-2">{eventCount.toLocaleString()}</div>
                </div>
                <div className="bg-black/40 p-4 rounded-xl border border-white/5 relative overflow-hidden group">
                  <div className={`absolute top-0 left-0 w-1 h-full ${backendEventCount >= 100 ? 'bg-emerald-400' : 'bg-purple-400'}`}></div>
                  <div className="font-mono text-[10px] tracking-widest uppercase text-purple-400 mb-1 pl-2">Backend Buffer</div>
                  <div className={`font-sans font-light text-3xl flex items-center gap-2 pl-2 ${backendEventCount >= 100 ? 'text-emerald-400' : 'text-white'}`}>
                    {backendEventCount.toLocaleString()}
                    {backendEventCount >= 100 && <CheckCircle className="w-4 h-4 text-emerald-400" />}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: Inference & Results */}
          <div className="lg:col-span-3 flex flex-col gap-6 h-full">
            <div className="bg-white/5 border border-white/10 backdrop-blur-md p-6 rounded-2xl flex-1 flex flex-col shadow-xl overflow-hidden relative">
              <h2 className="font-bold tracking-widest text-sm text-white/50 uppercase mb-6 flex items-center gap-2 relative z-10">
                <Cpu className="w-4 h-4 text-white/50" /> Neural Core
              </h2>

              {!decision ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center text-white/30 font-mono text-sm border-2 border-dashed border-white/10 rounded-xl p-6 relative z-10">
                  {challengeStatus === "processing" ? (
                    <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }}>
                      <Cpu className="w-10 h-10 text-cyan-400 mb-4 drop-shadow-[0_0_15px_rgba(0,229,255,0.5)]" />
                    </motion.div>
                  ) : (
                    <div className="mb-4 font-bold tracking-widest text-white/20">AWAITING INPUT</div>
                  )}
                  <span className="tracking-widest uppercase text-[10px]">
                    {challengeStatus === "processing" ? "ANALYZING VOXEL GRID..." : "STANDBY MODE"}
                  </span>
                </div>
              ) : (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex-1 rounded-xl p-4 flex flex-col items-center justify-center text-center relative z-10"
                >
                  <div className={`absolute inset-0 rounded-full blur-[60px] -z-10 ${
                    decision.decision === "genuine" ? "bg-emerald-500/20" :
                    decision.decision === "attack" ? "bg-red-500/20" :
                    "bg-amber-500/20"
                  }`}></div>
                  
                  <h3 className={`font-black ${
                    decision.decision.length > 8 ? "text-3xl lg:text-4xl" : "text-4xl lg:text-5xl"
                  } tracking-tight mb-4 drop-shadow-md break-all sm:break-normal ${
                    decision.decision === "genuine" ? "text-emerald-400" :
                    decision.decision === "attack" ? "text-red-500" :
                    "text-amber-400"
                  }`}>
                    {decision.decision.toUpperCase()}
                  </h3>
                  
                  {decision.confidence && (
                    <div className="font-bold tracking-widest text-xs text-white/80 bg-black/60 px-4 py-2 rounded-xl border border-white/10 shadow-lg">
                      CONFIDENCE: <span className="text-cyan-400">{(decision.confidence * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  
                  <div className="mt-6 font-mono text-[9px] text-white/40 tracking-widest uppercase opacity-70">
                    Spatio-Temporal Analysis Complete.
                  </div>
                </motion.div>
              )}
            </div>

            {/* Activity Profile */}
            <div className="bg-white/5 border border-white/10 backdrop-blur-md p-6 rounded-2xl h-56 flex flex-col shadow-xl">
              <h3 className="font-bold tracking-widest text-xs text-white/50 uppercase mb-4">Temporal Activity Profile</h3>
              <div className="flex-1 w-full bg-black/30 rounded-xl border border-white/5 p-3">
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                      <XAxis dataKey="bin" stroke="rgba(255,255,255,0.2)" tick={{fill: 'rgba(255,255,255,0.5)', fontSize: 10}} />
                      <YAxis stroke="rgba(255,255,255,0.2)" tick={{fill: 'rgba(255,255,255,0.5)', fontSize: 10}} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'rgba(5,0,18,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                        itemStyle={{ color: '#22d3ee' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="activity" 
                        stroke="url(#colorUv)" 
                        strokeWidth={2}
                        dot={false}
                        animationDuration={1500}
                      />
                      <defs>
                        <linearGradient id="colorUv" x1="0" y1="0" x2="1" y2="0">
                          <stop offset="0%" stopColor="#c084fc" />
                          <stop offset="100%" stopColor="#22d3ee" />
                        </linearGradient>
                      </defs>
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="w-full h-full flex items-center justify-center font-bold tracking-widest text-[10px] text-white/30 uppercase">
                    NO ACTIVITY DATA
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
