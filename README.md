import { useState, useEffect, useRef } from "react";

// ============================================================
// PIPELINE MODULES (merged from Python → JS)
// ============================================================

// ── Segmenter ────────────────────────────────────────────────
const WORDS_PER_SECOND = 2.5;
const STOPWORDS = new Set(["the","a","an","is","are","was","were","be","been","have","has","had","do","does","did","will","would","could","should","may","might","can","to","of","in","on","at","for","and","or","but","if","then","so","yet","it","its","this","that","you","your","we","our","i","my","me","they","their","with","from","by","about","up","not","no","what","when","how","all","just","like","get","into","every","one","s"]);

function extractKeywords(text) {
  const words = text.toLowerCase().match(/\b[a-z]{4,}\b/g) || [];
  const seen = new Set();
  return words.filter(w => !STOPWORDS.has(w) && !seen.has(w) && seen.add(w)).slice(0, 5);
}

function estimateDuration(text) {
  return parseFloat((text.split(" ").length / WORDS_PER_SECOND).toFixed(2));
}

function classifyScene(index, total) {
  if (index === 0) return "hook";
  if (index === 1) return "setup";
  if (index >= total - 2) return "cta";
  return "body";
}

function segmentScript(script) {
  const sentences = script.split(/(?<=[.!?])\s+/).filter(s => s.trim());
  const scenes = [];
  let buffer = [], bufferDur = 0, sceneIdx = 0;
  const TARGET = 5.0;

  for (let i = 0; i < sentences.length; i++) {
    const sent = sentences[i];
    const dur = estimateDuration(sent);
    buffer.push(sent);
    bufferDur += dur;
    if (bufferDur >= TARGET || i === sentences.length - 1) {
      const text = buffer.join(" ");
      scenes.push({
        index: sceneIdx,
        type: classifyScene(sceneIdx, sentences.length),
        text,
        keywords: extractKeywords(text),
        duration_s: parseFloat(bufferDur.toFixed(2)),
        media: null, effect: null, transition: null,
      });
      buffer = []; bufferDur = 0; sceneIdx++;
    }
  }
  return scenes;
}

// ── Media Matcher ─────────────────────────────────────────────
const STOCK_LIBRARY = [
  { clip_id:"clip_001", title:"Sunrise over mountains",      tags:["sunrise","mountains","nature","morning","light"],    duration_s:8, source:"stock" },
  { clip_id:"clip_002", title:"Person running on road",       tags:["running","person","road","exercise","motivation"],   duration_s:5, source:"stock" },
  { clip_id:"clip_003", title:"City skyline timelapse",       tags:["city","skyline","urban","timelapse","night"],        duration_s:6, source:"stock" },
  { clip_id:"clip_004", title:"Hands typing on keyboard",     tags:["typing","keyboard","work","computer","tech"],        duration_s:4, source:"stock" },
  { clip_id:"clip_005", title:"Ocean waves at sunset",        tags:["ocean","waves","sunset","water","calm"],             duration_s:7, source:"stock" },
  { clip_id:"clip_006", title:"Crowd cheering at concert",    tags:["crowd","concert","cheering","energy","music"],       duration_s:5, source:"stock" },
  { clip_id:"clip_007", title:"Person looking at horizon",    tags:["person","horizon","dream","future","hope"],          duration_s:6, source:"stock" },
  { clip_id:"clip_008", title:"Coffee brewing close-up",      tags:["coffee","morning","close","warm","cozy"],            duration_s:4, source:"stock" },
  { clip_id:"clip_009", title:"Athlete crossing finish line", tags:["athlete","finish","winning","success","sports"],     duration_s:5, source:"stock" },
  { clip_id:"clip_010", title:"Notebook and pen on desk",     tags:["notebook","writing","goals","planning","desk"],      duration_s:4, source:"stock" },
  { clip_id:"clip_011", title:"Friends laughing together",    tags:["friends","laughing","happy","social","joy"],         duration_s:6, source:"stock" },
  { clip_id:"clip_012", title:"Road stretching to horizon",   tags:["road","journey","travel","direction","path"],        duration_s:7, source:"stock" },
  { clip_id:"clip_013", title:"Motivational text on wall",    tags:["motivation","text","inspiration","quote","wall"],    duration_s:3, source:"stock" },
  { clip_id:"clip_014", title:"Slow motion rain drops",       tags:["rain","water","slow","motion","drops"],              duration_s:5, source:"stock" },
  { clip_id:"clip_015", title:"Person meditating outdoors",   tags:["meditation","peace","mindfulness","calm","focus"],   duration_s:6, source:"stock" },
  { clip_id:"clip_016", title:"Startup office team",          tags:["startup","office","team","work","business"],         duration_s:5, source:"stock" },
  { clip_id:"clip_017", title:"Fire burning close-up",        tags:["fire","energy","passion","burning","intensity"],     duration_s:4, source:"stock" },
  { clip_id:"clip_018", title:"Mountain climber at peak",     tags:["climbing","mountain","peak","achievement","hard"],   duration_s:7, source:"stock" },
  { clip_id:"clip_019", title:"Fast car on highway",          tags:["car","speed","highway","fast","drive"],              duration_s:4, source:"stock" },
  { clip_id:"clip_020", title:"Night sky stars milky way",    tags:["stars","night","sky","universe","wonder"],           duration_s:8, source:"stock" },
];

const ALL_TAGS = [...new Set(STOCK_LIBRARY.flatMap(c => c.tags))].sort();
const VOCAB_INDEX = Object.fromEntries(ALL_TAGS.map((t, i) => [t, i]));

function embed(words) {
  const vec = new Float32Array(ALL_TAGS.length);
  words.forEach(w => { if (VOCAB_INDEX[w.toLowerCase()] !== undefined) vec[VOCAB_INDEX[w.toLowerCase()]]++; });
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

const CLIP_EMBEDDINGS = STOCK_LIBRARY.map(c => embed(c.tags));

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

const SCENE_HINTS = { hook:["energy","attention","striking"], setup:["context","background","scene"], body:["action","detail","process"], climax:["intensity","peak","achievement"], cta:["motivation","call","action"] };

function matchMedia(scene) {
  const boosted = [...(scene.keywords || []), ...(SCENE_HINTS[scene.type] || [])];
  const qvec = embed(boosted);
  let bestIdx = 0, bestScore = -1;
  CLIP_EMBEDDINGS.forEach((cv, i) => {
    const s = cosineSim(qvec, cv);
    if (s > bestScore) { bestScore = s; bestIdx = i; }
  });
  return { ...STOCK_LIBRARY[bestIdx], score: parseFloat(bestScore.toFixed(4)) };
}

// ── TTS Engine ────────────────────────────────────────────────
const VOICE_PROFILES = {
  en_female:   { name:"Nova",  pitch_hz:220, speed_wpm:150, sample_rate:44100 },
  en_male:     { name:"Atlas", pitch_hz:110, speed_wpm:140, sample_rate:44100 },
  en_narrator: { name:"Sage",  pitch_hz:160, speed_wpm:130, sample_rate:44100 },
};

function generateTTS(script, voice = "en_female") {
  const profile = VOICE_PROFILES[voice] || VOICE_PROFILES.en_female;
  const words = script.split(" ").length;
  const pauses = (script.match(/[.!?]/g) || []).length * 0.4;
  const duration_s = parseFloat(((words / profile.speed_wpm) * 60 + pauses).toFixed(2));
  const sentences = script.split(/(?<=[.!?])\s+/).filter(s => s.trim());
  let cursor = 0;
  const segments = sentences.map((sent, i) => {
    const dur = parseFloat(((sent.split(" ").length / profile.speed_wpm) * 60).toFixed(3));
    const seg = { segment_id: i, text: sent, start_s: parseFloat(cursor.toFixed(3)), end_s: parseFloat((cursor + dur).toFixed(3)), duration_s: dur };
    cursor += dur + 0.4;
    return seg;
  });
  const hash = [...script].reduce((h, c) => (h * 31 + c.charCodeAt(0)) & 0xffffffff, 0).toString(16).slice(0,12);
  return { audio_id:`tts_${hash}`, voice: profile.name, voice_id: voice, duration_s, word_count: words, sample_rate: profile.sample_rate, format:"mp3", segments, file_path:`output/audio/tts_${hash}.mp3` };
}

// ── Caption Sync ──────────────────────────────────────────────
function alignCaptions(script, audioManifest) {
  const WORDS_PER_CARD = 5;
  const allWords = [];
  audioManifest.segments.forEach(seg => {
    const words = seg.text.split(" ");
    const totalW = words.reduce((s, w) => s + Math.max(1, w.length * 0.8), 0);
    let cursor = seg.start_s;
    words.forEach(word => {
      const w = Math.max(1, word.length * 0.8);
      const dur = parseFloat(((w / totalW) * seg.duration_s).toFixed(3));
      allWords.push({ word: word.replace(/[.,!?;:]/g, ""), start_s: parseFloat(cursor.toFixed(3)), end_s: parseFloat((cursor + dur).toFixed(3)) });
      cursor += dur;
    });
  });

  const styles = ["bold_white","bold_yellow","outline_white","highlight_box"];
  const captions = [];
  for (let i = 0; i < allWords.length; i += WORDS_PER_CARD) {
    const chunk = allWords.slice(i, i + WORDS_PER_CARD);
    if (!chunk.length) break;
    captions.push({
      caption_id: captions.length,
      text: chunk.map(w => w.word).join(" "),
      start_s: chunk[0].start_s,
      end_s: chunk[chunk.length - 1].end_s,
      duration_s: parseFloat((chunk[chunk.length-1].end_s - chunk[0].start_s).toFixed(3)),
      style: styles[captions.length % styles.length],
      position: "bottom_center",
    });
  }
  return captions;
}

// ── Effects Engine ────────────────────────────────────────────
const EFFECTS_MAP = { hook:["zoom_punch","glitch_flash","velocity_ramp"], setup:["slow_zoom","fade_in","parallax"], body:["steady","ken_burns","slow_zoom"], climax:["velocity_ramp","zoom_punch","shake"], cta:["fade_in","glow_pulse","slow_zoom"] };
const TRANSITIONS_MAP = { hook:"hard_cut", setup:"cross_dissolve", body:"smooth_slide", climax:"glitch", cta:"fade_to_black" };
const COLOR_GRADES = { "9:16":"moody_contrast", "16:9":"cinematic_teal", "1:1":"warm_feed" };

function applyEffects(scenes, aspectRatio = "9:16") {
  const grade = COLOR_GRADES[aspectRatio] || "moody_contrast";
  return scenes.map((scene, i) => {
    const opts = EFFECTS_MAP[scene.type] || ["steady"];
    return {
      ...scene,
      effect: opts[i % opts.length],
      transition: i === scenes.length - 1 ? "fade_to_black" : (TRANSITIONS_MAP[scene.type] || "smooth_slide"),
      color_grade: grade,
    };
  });
}

// ── Timeline Builder ──────────────────────────────────────────
function buildTimeline(scenes, audioManifest, captions, aspectRatio = "9:16") {
  const RES = { "9:16":[1080,1920], "16:9":[1920,1080], "1:1":[1080,1080] };
  const [w, h] = RES[aspectRatio] || [1080, 1920];
  const PLATFORMS = { "9:16":"tiktok_reels", "16:9":"youtube", "1:1":"instagram_feed" };

  let cursor = 0;
  const videoTrack = scenes.map((scene, i) => {
    const dur = Math.min(scene.duration_s, scene.media?.duration_s || scene.duration_s);
    const item = { track_item_id:`vi_${String(i).padStart(3,"0")}`, scene_index:i, scene_type:scene.type, clip_id:scene.media?.clip_id||"unknown", clip_title:scene.media?.title||"", match_score:scene.media?.score||0, timeline_start_s:parseFloat(cursor.toFixed(3)), timeline_end_s:parseFloat((cursor+dur).toFixed(3)), duration_s:parseFloat(dur.toFixed(3)), effect:scene.effect, transition:scene.transition, color_grade:scene.color_grade };
    cursor += dur;
    return item;
  });

  const totalDur = Math.max(cursor, audioManifest.duration_s);
  const dominant = scenes.reduce((acc, s) => { acc[s.type] = (acc[s.type]||0)+1; return acc; }, {});
  const domType = Object.entries(dominant).sort((a,b)=>b[1]-a[1])[0][0];
  const BGM = { hook:{track_id:"bgm_energy_01",genre:"upbeat",bpm:128}, body:{track_id:"bgm_ambient_02",genre:"ambient",bpm:90}, cta:{track_id:"bgm_uplifting_04",genre:"uplifting",bpm:100} };
  const bgm = BGM[domType] || BGM.body;

  return {
    schema_version: "1.0.0",
    meta: { aspect_ratio:aspectRatio, resolution:`${w}x${h}`, width:w, height:h, fps:30, total_duration_s:parseFloat(totalDur.toFixed(2)), scene_count:scenes.length, caption_count:captions.length },
    video_track: videoTrack,
    audio_track: { tts:{ audio_id:audioManifest.audio_id, file_path:audioManifest.file_path, voice:audioManifest.voice, duration_s:audioManifest.duration_s, volume:1.0 }, bgm:{ ...bgm, volume:0.25, loop:true }, mix_config:{ ducking:true, duck_ratio:0.3, normalize:true, target_lufs:-14 } },
    caption_track: captions,
    export_config: { codec:"h264", bitrate_kbps:aspectRatio==="16:9"?12000:8000, profile:"high", fps:30, audio_codec:"aac", audio_bitrate_kbps:192, format:"mp4", color_space:"yuv420p", platform:PLATFORMS[aspectRatio]||"tiktok_reels" },
  };
}

// ── LLM (Anthropic API) ───────────────────────────────────────
async function callAnthropic(systemPrompt, userPrompt, maxTokens = 1000) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model:"claude-sonnet-4-20250514", max_tokens:maxTokens, system:systemPrompt, messages:[{ role:"user", content:userPrompt }] }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.content[0].text;
}

async function generateScript(prompt) {
  return callAnthropic(
    "You are a video script writer. Write a punchy, engaging script for a 30-60 second short-form video in a natural spoken voice. No stage directions, just spoken words. Under 150 words.",
    `Write a video script about: ${prompt}`
  );
}

async function segmentWithLLM(script) {
  const raw = await callAnthropic(
    "You are a video editor AI. Split the script into scenes for a short-form video. Each scene is 3-8 seconds. Label each: hook|setup|body|climax|cta. Return ONLY valid JSON array, no markdown. Format: [{\"type\":\"hook\",\"text\":\"...\",\"keywords\":[\"w1\",\"w2\"],\"duration_s\":4.0}]",
    `Split into scenes:\n\n${script}`,
    1000
  );
  const clean = raw.trim().replace(/^```json|^```|```$/g, "").trim();
  return JSON.parse(clean);
}

// ── Full Pipeline ─────────────────────────────────────────────
async function runPipeline(prompt, aspectRatio, voice, onStep) {
  const steps = [];
  const log = (step, status, detail) => { steps.push({ step, status, detail }); onStep([...steps]); };

  // Step 1: Script
  log("Script Generation", "running", "Expanding prompt...");
  let script;
  try {
    script = prompt.split(" ").length < 30 ? await generateScript(prompt) : prompt;
    log("Script Generation", "done", `${script.split(" ").length} words generated`);
  } catch(e) {
    script = `Every great journey starts with a single decision. The decision to try. Right now, wherever you are, you have the power to change everything. Stop waiting for the perfect moment — it doesn't exist. The people who changed the world didn't have more talent. They simply refused to quit. Every setback was fuel. Every failure was data. You were built for this. Take the first step. Do it today.`;
    log("Script Generation", "fallback", `API unavailable — using demo script`);
  }

  // Step 2: Segmentation
  log("Scene Segmentation", "running", "Analyzing script structure...");
  let scenes;
  try {
    const raw = await segmentWithLLM(script);
    scenes = raw.map((s, i) => ({ index:i, type:s.type||"body", text:s.text, keywords:s.keywords||extractKeywords(s.text), duration_s:s.duration_s||estimateDuration(s.text), media:null, effect:null, transition:null }));
    log("Scene Segmentation", "done", `${scenes.length} scenes (LLM)`);
  } catch(e) {
    scenes = segmentScript(script);
    log("Scene Segmentation", "done", `${scenes.length} scenes (heuristic)`);
  }

  // Step 3: Media Match
  log("Media Matching", "running", "Finding matching clips...");
  scenes = scenes.map(s => ({ ...s, media: matchMedia(s) }));
  log("Media Matching", "done", `${scenes.length} clips matched`);

  // Step 4: TTS
  log("TTS Voiceover", "running", "Generating audio manifest...");
  const audioManifest = generateTTS(script, voice);
  log("TTS Voiceover", "done", `${audioManifest.duration_s}s audio @ ${audioManifest.sample_rate}Hz`);

  // Step 5: Captions
  log("Caption Sync", "running", "Aligning captions...");
  const captions = alignCaptions(script, audioManifest);
  log("Caption Sync", "done", `${captions.length} caption cards`);

  // Step 6: Effects
  log("Effects & Transitions", "running", "Applying effects stack...");
  scenes = applyEffects(scenes, aspectRatio);
  log("Effects & Transitions", "done", "Effects applied");

  // Step 7: Timeline
  log("Timeline Assembly", "running", "Building master timeline...");
  const timeline = buildTimeline(scenes, audioManifest, captions, aspectRatio);
  log("Timeline Assembly", "done", `${timeline.meta.total_duration_s}s total`);

  // Step 8: Export
  log("Export", "running", "Generating output...");
  await new Promise(r => setTimeout(r, 400));
  log("Export", "done", `${timeline.meta.resolution} · ${timeline.export_config.platform}`);

  return { timeline, script, scenes, captions, audioManifest };
}

// ============================================================
// UI COMPONENTS
// ============================================================

const STEP_ICONS = ["✦","⊹","◈","◎","⊞","✧","⊛","⊡"];
const STATUS_COLORS = { running:"#f59e0b", done:"#22d3a5", fallback:"#a78bfa", idle:"#334155" };

function StepRow({ step, status, detail, icon }) {
  return (
    <div style={{ display:"flex", alignItems:"flex-start", gap:12, padding:"10px 0", borderBottom:"1px solid #0f172a", opacity: status==="idle"?0.3:1, transition:"opacity 0.3s" }}>
      <div style={{ width:28, height:28, borderRadius:6, background: status==="running" ? "#f59e0b22" : status==="done"?"#22d3a522":status==="fallback"?"#a78bfa22":"#1e293b", border:`1px solid ${STATUS_COLORS[status]||"#1e293b"}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, color:STATUS_COLORS[status]||"#475569", flexShrink:0, transition:"all 0.3s" }}>
        {status === "running" ? <SpinIcon /> : icon}
      </div>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontSize:13, fontWeight:600, color: status==="idle"?"#334155":"#e2e8f0", letterSpacing:"0.02em", fontFamily:"'DM Mono', monospace" }}>{step}</div>
        {detail && <div style={{ fontSize:11, color:"#64748b", marginTop:2, fontFamily:"'DM Mono', monospace" }}>{detail}</div>}
      </div>
      {status==="done" && <div style={{ fontSize:11, color:"#22d3a5", fontFamily:"'DM Mono', monospace", flexShrink:0 }}>✓ done</div>}
      {status==="fallback" && <div style={{ fontSize:11, color:"#a78bfa", fontFamily:"'DM Mono', monospace", flexShrink:0 }}>⚑ fallback</div>}
    </div>
  );
}

function SpinIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" style={{ animation:"spin 0.8s linear infinite" }}>
      <circle cx="6" cy="6" r="4.5" fill="none" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="14 8" />
    </svg>
  );
}

function TimelineViz({ timeline }) {
  if (!timeline) return null;
  const { video_track, meta, caption_track } = timeline;
  const total = meta.total_duration_s;
  const COLORS = { hook:"#f43f5e", setup:"#f59e0b", body:"#22d3a5", climax:"#818cf8", cta:"#fb923c" };
  const effectColors = { zoom_punch:"#f43f5e", velocity_ramp:"#fb923c", glitch_flash:"#a78bfa", slow_zoom:"#22d3a5", ken_burns:"#38bdf8", steady:"#64748b", fade_in:"#818cf8", parallax:"#4ade80", shake:"#f472b6", glow_pulse:"#fbbf24" };

  return (
    <div style={{ background:"#080f1a", borderRadius:12, padding:20, border:"1px solid #1e293b" }}>
      <div style={{ fontSize:11, color:"#475569", fontFamily:"'DM Mono', monospace", marginBottom:12, letterSpacing:"0.08em" }}>TIMELINE · {total}s · {meta.fps}fps · {meta.resolution}</div>

      {/* Video track */}
      <div style={{ marginBottom:8 }}>
        <div style={{ fontSize:10, color:"#334155", fontFamily:"'DM Mono', monospace", marginBottom:4, letterSpacing:"0.06em" }}>VIDEO TRACK</div>
        <div style={{ display:"flex", height:32, borderRadius:4, overflow:"hidden", gap:1, background:"#0a1628" }}>
          {video_track
