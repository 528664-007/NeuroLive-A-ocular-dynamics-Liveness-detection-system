import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, Hexagon, ShieldCheck, Zap, Layers, Network, ChevronDown, Lock, Server, BarChart3, Database } from "lucide-react";

export default function Landing({ onEnter }) {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  const yHero = useTransform(scrollYProgress, [0, 1], ["0%", "40%"]);
  const opacityHero = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  
  return (
    <div ref={containerRef} className="bg-[#020008] text-white overflow-hidden relative selection:bg-purple-500/30 font-sans">
      
      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPgo8cmVjdCB3aWR0aD0iNCIgaGVpZ2h0PSI0IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDMiLz4KPC9zdmc+')] mix-blend-overlay"></div>
        <motion.div 
          animate={{ scale: [1, 1.2, 1], rotate: [0, 90, 0] }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          className="absolute top-[-20%] left-[-10%] w-[60vw] h-[60vw] rounded-full bg-[radial-gradient(circle_at_center,rgba(124,58,237,0.15)_0%,transparent_60%)] blur-[100px]" 
        />
        <motion.div 
          animate={{ scale: [1, 1.5, 1], rotate: [0, -90, 0] }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
          className="absolute bottom-[-20%] right-[-10%] w-[70vw] h-[70vw] rounded-full bg-[radial-gradient(circle_at_center,rgba(0,229,255,0.15)_0%,transparent_60%)] blur-[120px]" 
        />
      </div>

      {/* Floating Navbar */}
      <motion.nav 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="fixed top-6 left-1/2 -translate-x-1/2 w-[90%] max-w-7xl z-50 bg-white/5 backdrop-blur-2xl border border-white/10 rounded-full flex justify-between items-center px-6 md:px-8 py-4 shadow-[0_8px_32px_rgba(0,0,0,0.3)]"
      >
        <div className="flex items-center gap-3 cursor-pointer group">
          <motion.div whileHover={{ rotate: 90 }} transition={{ type: "spring" }}>
            <Hexagon className="w-8 h-8 text-cyan-400 drop-shadow-[0_0_15px_rgba(0,229,255,0.8)]" fill="currentColor" fillOpacity={0.2} strokeWidth={1.5} />
          </motion.div>
          <span className="font-bold tracking-wider text-lg bg-clip-text text-transparent bg-gradient-to-r from-white to-white/70">CanthusCore</span>
        </div>
        <div className="hidden md:flex gap-8 text-sm font-medium text-white/70">
          <a href="#" className="hover:text-cyan-400 transition-colors">Platform</a>
          <a href="#" className="hover:text-purple-400 transition-colors">Features</a>
          <a href="#" className="hover:text-pink-400 transition-colors">Security</a>
        </div>
        <motion.button 
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onEnter} 
          className="px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-purple-600 text-white text-sm font-bold rounded-full flex items-center gap-2 group shadow-[0_0_20px_rgba(124,58,237,0.4)]"
        >
          Launch Demo
          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </motion.button>
      </motion.nav>

      {/* Hero */}
      <motion.div style={{ y: yHero, opacity: opacityHero }} className="relative z-10 pt-48 pb-20 px-8 min-h-[90vh] flex flex-col justify-center items-center text-center">
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, type: "spring" }}
          className="inline-flex items-center gap-3 bg-white/5 border border-white/10 px-5 py-2 rounded-full backdrop-blur-md mb-8 hover:bg-white/10 transition-colors"
        >
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
          </span>
          <span className="text-xs font-bold tracking-[0.2em] uppercase text-cyan-50">CanthusCore V2 Engine Live</span>
        </motion.div>
        
        <motion.h1 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
          className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tight leading-[1.1] mb-6 max-w-5xl"
        >
          The Future of <br/>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 drop-shadow-[0_0_30px_rgba(124,58,237,0.5)]">
            Liveness Intelligence.
          </span>
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.4 }}
          className="text-lg md:text-2xl text-white/60 max-w-3xl leading-relaxed mb-12 font-light"
        >
          Transform visual motion into real-time temporal event representations. Advanced neural inference for absolute, impenetrable verification certainty.
        </motion.p>
        
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.6 }}
          className="flex flex-col sm:flex-row gap-6"
        >
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onEnter} 
            className="px-10 py-5 bg-white text-black rounded-full font-bold text-lg shadow-[0_0_40px_rgba(255,255,255,0.2)] hover:shadow-[0_0_60px_rgba(255,255,255,0.4)] transition-all flex items-center justify-center gap-3 group"
          >
            Experience Live Demo
            <div className="bg-black text-white p-1 rounded-full group-hover:rotate-45 transition-transform duration-300">
              <ArrowRight className="w-5 h-5" />
            </div>
          </motion.button>
        </motion.div>
      </motion.div>

      {/* New Live Statistics Section */}
      <motion.div 
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        className="relative z-10 py-12 px-8 max-w-7xl mx-auto border-t border-b border-white/5 bg-white/[0.01]"
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-4 text-center">
          <div>
            <div className="text-4xl md:text-5xl font-black text-cyan-400 mb-2">99.9%</div>
            <div className="text-xs font-bold tracking-widest text-white/50 uppercase">Attack Prevention</div>
          </div>
          <div>
            <div className="text-4xl md:text-5xl font-black text-purple-400 mb-2">&lt;35ms</div>
            <div className="text-xs font-bold tracking-widest text-white/50 uppercase">Network Latency</div>
          </div>
          <div>
            <div className="text-4xl md:text-5xl font-black text-pink-400 mb-2">3.4M+</div>
            <div className="text-xs font-bold tracking-widest text-white/50 uppercase">Daily Verifications</div>
          </div>
          <div>
            <div className="text-4xl md:text-5xl font-black text-white mb-2 flex justify-center items-center gap-2">
              <Lock className="w-8 h-8 text-emerald-400" /> AES-256
            </div>
            <div className="text-xs font-bold tracking-widest text-white/50 uppercase">Encryption Tier</div>
          </div>
        </div>
      </motion.div>

      {/* Architecture Showcase */}
      <div className="relative z-10 py-32 px-8 max-w-7xl mx-auto">
        <div className="text-center mb-24">
          <h2 className="text-4xl md:text-6xl font-bold mb-6">Engineered for <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400">Precision.</span></h2>
          <p className="text-xl text-white/50 max-w-2xl mx-auto font-light">CanthusCore completely bypasses legacy facial recognition limits by streaming true temporal motion data.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <motion.div
            whileHover={{ y: -5 }}
            className="bg-white/5 border border-white/10 rounded-3xl p-10 relative overflow-hidden group backdrop-blur-xl hover:shadow-[0_0_40px_rgba(0,229,255,0.15)] transition-all duration-300"
          >
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center mb-8 shadow-lg">
              <Network className="w-8 h-8 text-white" strokeWidth={1.5} />
            </div>
            <h3 className="text-3xl font-bold mb-4">Event-Based Vision</h3>
            <p className="text-white/60 leading-relaxed text-lg">
              Asynchronous event processing captures micro-expressions and vital signs absolutely invisible to traditional optical sensors.
            </p>
          </motion.div>
          
          <motion.div
            whileHover={{ y: -5 }}
            className="bg-white/5 border border-white/10 rounded-3xl p-10 relative overflow-hidden group backdrop-blur-xl hover:shadow-[0_0_40px_rgba(192,38,211,0.15)] transition-all duration-300"
          >
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-400 to-pink-500 flex items-center justify-center mb-8 shadow-lg">
              <Layers className="w-8 h-8 text-white" strokeWidth={1.5} />
            </div>
            <h3 className="text-3xl font-bold mb-4">Spatial Voxelization</h3>
            <p className="text-white/60 leading-relaxed text-lg">
              High-dimensional tensors map temporal changes into structural geometry, ensuring flawless deep contextual awareness.
            </p>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <motion.div
            whileHover={{ y: -5 }}
            className="bg-white/5 border border-white/10 rounded-3xl p-8 backdrop-blur-xl hover:shadow-[0_0_30px_rgba(52,211,153,0.15)] transition-all"
          >
            <ShieldCheck className="w-10 h-10 text-emerald-400 mb-6" />
            <h3 className="text-xl font-bold mb-3">Neural Certainty</h3>
            <p className="text-white/50 text-sm">Bespoke Joint-Core neural architecture synthesizes behavioral challenges.</p>
          </motion.div>
          
          <motion.div
            whileHover={{ y: -5 }}
            className="bg-white/5 border border-white/10 rounded-3xl p-8 backdrop-blur-xl hover:shadow-[0_0_30px_rgba(251,191,36,0.15)] transition-all"
          >
            <Server className="w-10 h-10 text-amber-400 mb-6" />
            <h3 className="text-xl font-bold mb-3">Edge Compute</h3>
            <p className="text-white/50 text-sm">Deployed lightweight models that run locally with zero cloud dependencies.</p>
          </motion.div>
          
          <motion.div
            whileHover={{ y: -5 }}
            className="bg-white/5 border border-white/10 rounded-3xl p-8 backdrop-blur-xl hover:shadow-[0_0_30px_rgba(56,189,248,0.15)] transition-all"
          >
            <Database className="w-10 h-10 text-sky-400 mb-6" />
            <h3 className="text-xl font-bold mb-3">Immutable Logs</h3>
            <p className="text-white/50 text-sm">Every verification generates a cryptographic token for audit compliance.</p>
          </motion.div>
        </div>
      </div>

      {/* Massive Call to Action */}
      <div className="relative z-10 py-24 px-8">
        <div className="max-w-5xl mx-auto bg-gradient-to-br from-white/10 to-white/5 border border-white/20 rounded-[3rem] p-16 text-center relative overflow-hidden backdrop-blur-2xl">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 50, repeat: Infinity, ease: "linear" }}
            className="absolute -top-1/2 -right-1/2 w-full h-full bg-gradient-to-b from-cyan-500/20 to-purple-500/20 blur-[100px] rounded-full pointer-events-none"
          ></motion.div>
          
          <div className="relative z-10">
            <h2 className="text-4xl md:text-6xl font-bold mb-8">Ready to upgrade?</h2>
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onEnter} 
              className="px-12 py-5 bg-white text-black rounded-full font-bold text-xl hover:shadow-[0_0_40px_rgba(255,255,255,0.4)] transition-all inline-flex items-center gap-4"
            >
              Enter CanthusCore
              <Zap className="w-6 h-6 text-purple-600" fill="currentColor" />
            </motion.button>
          </div>
        </div>
      </div>
      
      <footer className="relative z-10 border-t border-white/5 bg-[#020008] py-8 px-12 text-center text-white/30 font-bold text-xs tracking-widest uppercase">
        CanthusCore Enterprise Edition © 2026
      </footer>
    </div>
  );
}
