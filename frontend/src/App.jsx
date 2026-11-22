import React, { useEffect, useRef, useState } from "react";
import {
  Activity, AlertTriangle, Bell, BellOff, Cpu, Play, Pause, RotateCcw,
  Square, Users, Video, ShieldCheck, Upload, FolderOpen, RefreshCw,
  Thermometer, Gauge, Clock3, CheckCircle2
} from "lucide-react";

const API = "http://127.0.0.1:8000";
const initial = {
  fps: 0, inference_ms: 0, people: 0, active_falls: 0, cuda: false,
  gpu_name: "Connecting…", device: "—", fp16: false, video: "—",
  status: "stopped", total_falls: 0
};

// Optional telemetry fields are read only if the existing backend supplies them.
// The UI deliberately does not invent CPU/GPU usage or temperature values.
function firstValue(source, keys) {
  for (const key of keys) {
    const value = source?.[key];
    if (value !== undefined && value !== null && value !== "") return Number(value);
  }
  return null;
}

function formatMetric(value, suffix = "") {
  return value === null || !Number.isFinite(value) ? "—" : `${Math.round(value)}${suffix}`;
}

export default function App() {
  const [stats, setStats] = useState(initial);
  const [system, setSystem] = useState({ cpu_percent: null, cpu_temperature: null, gpu_percent: null, gpu_temperature: null, gpu_name: "—" });
  const [history, setHistory] = useState([]);
  const [videos, setVideos] = useState([]);
  const [videoFolder, setVideoFolder] = useState("");
  const [alarmEnabled, setAlarmEnabled] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [uploading, setUploading] = useState(false);
  const audioCtx = useRef(null);
  const alarmTimer = useRef(null);
  const fileInput = useRef(null);

  async function refreshVideos() {
    try {
      const response = await fetch(`${API}/api/videos`);
      const data = await response.json();
      setVideos(data.videos || []);
      setVideoFolder(data.folder || "");
    } catch {}
  }

  async function refresh() {
    try {
      const [statsResponse, historyResponse, systemResponse] = await Promise.all([
        fetch(`${API}/api/monitor/stats`).then((response) => response.json()),
        fetch(`${API}/api/monitor/history`).then((response) => response.json()),
        fetch(`${API}/api/system/stats`).then((response) => response.json()).catch(() => ({}))
      ]);
      setStats(statsResponse);
      setHistory(historyResponse.events || []);
      setSystem(systemResponse || {});
    } catch {}
  }

  useEffect(() => {
    refresh();
    refreshVideos();
    const statsTimer = setInterval(refresh, 650);
    const videosTimer = setInterval(refreshVideos, 3000);
    return () => {
      clearInterval(statsTimer);
      clearInterval(videosTimer);
    };
  }, []);

  async function unlockAudio() {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass && !audioCtx.current) audioCtx.current = new AudioContextClass();
      if (audioCtx.current?.state === "suspended") await audioCtx.current.resume();
    } catch {}
  }

  // Industrial two-pulse warning: clear and audible at moderate volume,
  // with a short pause between cycles to avoid an irritating continuous siren.
  function playIndustrialAlert() {
    const context = audioCtx.current;
    if (!context || context.state !== "running") return;
    const sequence = [
      { frequency: 880, offset: 0.00, duration: 0.20 },
      { frequency: 660, offset: 0.34, duration: 0.24 }
    ];
    const master = context.createGain();
    master.gain.setValueAtTime(0.72, context.currentTime);
    master.connect(context.destination);
    sequence.forEach(({ frequency, offset, duration }) => {
      const oscillator = context.createOscillator();
      const envelope = context.createGain();
      const begin = context.currentTime + 0.025 + offset;
      oscillator.type = "triangle";
      oscillator.frequency.setValueAtTime(frequency, begin);
      envelope.gain.setValueAtTime(0.0001, begin);
      envelope.gain.exponentialRampToValueAtTime(0.105, begin + 0.025);
      envelope.gain.setValueAtTime(0.105, begin + duration - 0.045);
      envelope.gain.exponentialRampToValueAtTime(0.0001, begin + duration);
      oscillator.connect(envelope);
      envelope.connect(master);
      oscillator.start(begin);
      oscillator.stop(begin + duration + 0.01);
    });
    const endAt = context.currentTime + 0.025 + 0.34 + 0.25 + 0.08;
    master.gain.setValueAtTime(0.72, endAt - 0.02);
    master.gain.linearRampToValueAtTime(0.0001, endAt);
    setTimeout(() => master.disconnect(), 900);
  }

  const fallActive = (stats.active_falls || 0) > 0;
  useEffect(() => {
    if (alarmTimer.current) clearInterval(alarmTimer.current);
    alarmTimer.current = null;
    if (fallActive && alarmEnabled) {
      playIndustrialAlert();
      alarmTimer.current = setInterval(playIndustrialAlert, 4200);
    }
    return () => {
      if (alarmTimer.current) clearInterval(alarmTimer.current);
    };
  }, [fallActive, alarmEnabled]);

  async function action(path) {
    setBusy(true);
    setMessage("");
    try {
      if (path.includes("start") || path.includes("restart")) await unlockAudio();
      const response = await fetch(`${API}${path}`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The request could not be completed.");
      await refresh();
      if (path.includes("start") || path.includes("restart")) {
        setMessage("Monitoring is running. Your video playlist is being processed.");
      }
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadSelected(event) {
    const selectedFiles = Array.from(event.target.files || []);
    if (!selectedFiles.length) return;
    setUploading(true);
    setMessage("");
    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append("files", file));
      const response = await fetch(`${API}/api/videos/upload`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed.");
      setMessage(`Added ${data.saved.length} video${data.saved.length === 1 ? "" : "s"} to your playlist.`);
      await refreshVideos();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  const running = ["running", "paused", "starting"].includes(stats.status);
  const cpuUsage = firstValue(system, ["cpu_percent"]);
  const gpuUsage = firstValue(system, ["gpu_percent"]);
  const cpuTemp = firstValue(system, ["cpu_temperature"]);
  const gpuTemp = firstValue(system, ["gpu_temperature"]);

  return (
    <main className="appShell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Fallguard home">
          <span className="brandMark"><ShieldCheck size={21} strokeWidth={1.8} /></span>
          <span className="brandCopy"><strong>fallguard<span>.</span></strong><small>WORKPLACE SAFETY MONITOR</small></span>
        </a>
        <div className="topStatus">
          <span className={`connectionDot ${running ? "isLive" : ""}`} />
          <span>{running ? "Monitoring" : (stats.status || "Connecting")}</span>
          <span className="statusDivider" />
          <span className="modelLabel">VISION SYSTEM · V2</span>
        </div>
      </header>

      <section className="welcomeRow projectTitleRow" id="top">
        <div>
          <div className="overline">WORKPLACE SAFETY</div>
          <h1>FallGuard</h1>
        </div>
        <div className={`overallStatus ${fallActive ? "statusAlert" : "statusClear"}`}>
          <span className="statusPulse" />
          <span><strong>{fallActive ? "Attention needed" : "All clear"}</strong><small>{fallActive ? "A possible fall has been detected" : "No fall event currently detected"}</small></span>
        </div>
      </section>

      <section className="dashboardGrid">
        <section className="panel livePanel">
          <div className="sectionHeading">
            <div><span className="sectionEyebrow">LIVE VIEW</span><h2>Camera feed</h2></div>
            <span className={`liveLabel ${running ? "liveNow" : ""}`}><i />{running ? "LIVE" : "STANDBY"}</span>
          </div>
          <div className="videoViewport">
            {stats.frames > 0 || running ? (
              <img src={`${API}/api/monitor/video`} alt="Live fall detection video feed" />
            ) : (
              <div className="videoEmpty"><span className="emptyCamera"><Video size={25} /></span><strong>Your camera view will appear here</strong><p>Add a reference video, then start monitoring.</p></div>
            )}
            <div className="videoMeta"><span><i /> LIVE ANALYSIS</span><span>{stats.video || "Waiting for video"}</span></div>
            {fallActive && <div className="fallNotice"><AlertTriangle size={16} /> Fall event detected · Please check the footage</div>}
          </div>
          <div className="videoCaption"><span><CheckCircle2 size={14} /> Person tracking is active when monitoring runs</span><span>{stats.people || 0} people in view</span></div>
          <div className="controlBar">
            <button className="button buttonPrimary" disabled={busy} onClick={() => action("/api/monitor/start")}><Play size={15} fill="currentColor" /> Start monitoring</button>
            <button className="button" disabled={busy || !running} onClick={() => action("/api/monitor/pause")}><Pause size={15} /> Pause</button>
            <button className="button" disabled={busy} onClick={() => action("/api/monitor/restart")}><RotateCcw size={15} /> Restart</button>
            <button className="button buttonQuietStop" disabled={busy || !running} onClick={() => action("/api/monitor/stop")}><Square size={13} /> Stop</button>
          </div>
        </section>

        <aside className="sideColumn">
          <section className={`safetyCard ${fallActive ? "safetyCardAlert" : ""}`}>
            <div className="safetyCardTop"><span className="safetyIcon"><ShieldCheck size={18} /></span><span className="sectionEyebrow">SAFETY STATUS</span></div>
            <h2>{fallActive ? "Fall detected" : "All clear"}</h2>
            <button className={`soundToggle ${alarmEnabled ? "soundOn" : "soundOff"}`} onClick={async () => { await unlockAudio(); setAlarmEnabled((value) => !value); }} aria-label={`Alarm ${alarmEnabled ? "on" : "off"}`}>
              {alarmEnabled ? <Bell size={15} /> : <BellOff size={15} />}
              <span><strong>Alarm</strong><small>{alarmEnabled ? "ON" : "OFF"}</small></span>
              <span className={`toggleTrack ${alarmEnabled ? "toggleActive" : ""}`}><i /></span>
            </button>
          </section>

          <section className="panel systemPanel">
            <div className="systemHeading"><div><span className="sectionEyebrow">SYSTEM HEALTH</span><h2>Device status</h2></div><Activity size={17} /></div>
            <div className="systemGroup">
              <div className="systemRow"><span className="systemName"><Cpu size={15} /> CPU</span><strong>{formatMetric(cpuUsage, "%")}</strong></div>
              <div className="meterTrack"><i style={{ width: `${cpuUsage === null ? 0 : Math.max(0, Math.min(100, cpuUsage))}%` }} /></div>
              <div className="systemSub"><span><Thermometer size={12} /> Temperature</span><b>{formatMetric(cpuTemp, "°C")}</b></div>
            </div>
            <div className="systemGroup">
              <div className="systemRow"><span className="systemName"><Gauge size={15} /> GPU</span><strong>{formatMetric(gpuUsage, "%")}</strong></div>
              <div className="meterTrack gpuMeter"><i style={{ width: `${gpuUsage === null ? 0 : Math.max(0, Math.min(100, gpuUsage))}%` }} /></div>
              <div className="systemSub"><span><Thermometer size={12} /> Temperature</span><b>{formatMetric(gpuTemp, "°C")}</b></div>
            </div>
            <div className="deviceFoot"><span className={stats.cuda ? "deviceReady" : "deviceWait"} />{stats.cuda ? "GPU acceleration connected" : "Waiting for compute connection"}</div>
          </section>

          <section className="panel quickFacts">
            <div className="quickFact"><span>Processing speed</span><strong>{stats.fps || "—"}<small> FPS</small></strong></div>
            <div className="quickFact"><span>People in view</span><strong>{stats.people ?? "—"}</strong></div>
            <div className="quickFact"><span>Recorded events</span><strong>{stats.total_falls ?? 0}</strong></div>
          </section>
        </aside>
      </section>

      <section className="lowerGrid">
        <section className="panel historyPanel">
          <div className="sectionHeading compactHeading"><div><span className="sectionEyebrow">RECENT ACTIVITY</span><h2>Fall event history</h2></div><span className="softCount">{history.length} recorded</span></div>
          {!history.length ? (
            <div className="historyEmpty"><span><Clock3 size={18} /></span><div><strong>No events to review</strong><p>Confirmed fall events will appear here as they happen.</p></div></div>
          ) : (
            <div className="historyList">
              {history.slice(0, 7).map((event, index) => (
                <div className="historyItem" key={`${event.timestamp}-${index}`}>
                  <span className="historyAlertIcon"><AlertTriangle size={14} /></span>
                  <div className="historyMain"><strong>Fall detected</strong><small>Person #{event.track_id} · {Math.round((event.confidence || 0) * 100)}% confidence</small></div>
                  <time>{event.timestamp}</time>
                  <span className="historyVideo">{event.video}</span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="panel playlistPanel">
          <div className="sectionHeading compactHeading"><div><span className="sectionEyebrow">YOUR FOOTAGE</span><h2>Video playlist</h2></div><button className="refreshButton" onClick={refreshVideos} title="Refresh playlist" aria-label="Refresh playlist"><RefreshCw size={15} /></button></div>
          <div className="folderLocation"><FolderOpen size={13} /><span>{videoFolder || "Video folder is loading…"}</span></div>
          <div className="playlistList">
            {!videos.length ? <div className="playlistEmpty">No videos added yet.</div> : videos.map((video, index) => (
              <div className="playlistItem" key={video}>
                <span className="playlistPlay"><Play size={11} fill="currentColor" /></span>
                <div className="playlistInfo"><strong title={video}>{video}</strong><small>Video {index + 1}</small></div>
                <span className={stats.video === video ? "playlistPlaying" : "playlistQueued"}>{stats.video === video ? "NOW PLAYING" : "QUEUED"}</span>
              </div>
            ))}
          </div>
          <input ref={fileInput} type="file" accept="video/*,.mkv,.avi,.m4v" multiple hidden onChange={uploadSelected} />
          <button className="uploadButton" disabled={uploading} onClick={() => fileInput.current?.click()}><Upload size={14} />{uploading ? "Adding videos…" : "Add videos"}</button>
        </section>
      </section>

      {message && <div className="messageToast" role="status"><span>{message}</span><button onClick={() => setMessage("")} aria-label="Dismiss message">×</button></div>}
      {stats.error && <div className="errorToast"><AlertTriangle size={15} />{stats.error}</div>}
      <footer className="footer"><span>FALLGUARD <i>·</i> WORKPLACE SAFETY</span><span>Live monitoring</span></footer>
    </main>
  );
}
