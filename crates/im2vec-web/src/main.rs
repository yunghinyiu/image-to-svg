use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart},
    http::StatusCode,
    response::Html,
    routing::{get, post},
};
use im2vec_core::{ConvertOptions, ImPreset, convert_bytes};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct ConvertResponse {
    svg: String,
    width: u32,
    height: u32,
    path_count: usize,
    svg_bytes: usize,
    elapsed_ms: u128,
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn healthz() -> &'static str {
    "ok"
}

fn parse_preset(s: &str) -> ImPreset {
    match s {
        "illustration" => ImPreset::Illustration,
        "photo" => ImPreset::Photo,
        "mono" => ImPreset::Mono,
        _ => ImPreset::Logo,
    }
}

async fn api_convert(mut mp: Multipart) -> Result<Json<ConvertResponse>, (StatusCode, String)> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut opts = ConvertOptions::default();

    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        let data = field
            .bytes()
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("read field: {e}")))?;
        let text = String::from_utf8_lossy(&data).to_string();
        match name.as_str() {
            "file" => file_bytes = Some(data.to_vec()),
            "preset" => opts = ConvertOptions::for_preset(parse_preset(text.trim())),
            "mode" => opts.mode = text.trim().to_string(),
            "hierarchical" => opts.hierarchical = text.trim().to_string(),
            "filter_speckle" => {
                if let Ok(v) = text.trim().parse() {
                    opts.filter_speckle = v;
                }
            }
            "color_precision" => {
                if let Ok(v) = text.trim().parse() {
                    opts.color_precision = v;
                }
            }
            "gradient_step" => {
                if let Ok(v) = text.trim().parse() {
                    opts.gradient_step = v;
                }
            }
            "max_colors" => {
                let t = text.trim();
                opts.max_colors = if t.is_empty() || t == "0" {
                    None
                } else {
                    t.parse().ok()
                };
            }
            "simplify" => {
                let t = text.trim();
                opts.simplify = if t.is_empty() || t == "off" {
                    None
                } else {
                    t.parse().ok()
                };
            }
            "path_precision" => {
                if let Ok(v) = text.trim().parse() {
                    opts.path_precision = v;
                }
            }
            "clustering" => opts.clustering = text.trim().to_string(),
            "watershed_detail" => {
                let t = text.trim();
                opts.watershed_detail = if t.is_empty() || t == "auto" {
                    None
                } else {
                    t.parse().ok()
                };
            }
            _ => {}
        }
    }

    let bytes = file_bytes.ok_or((StatusCode::BAD_REQUEST, "missing 'file'".into()))?;
    if bytes.len() > 25 * 1024 * 1024 {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, "file > 25MB".into()));
    }

    let t = Instant::now();
    let out = tokio::task::spawn_blocking(move || convert_bytes(&bytes, &opts))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("task: {e}")))?
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("convert: {e:#}")))?;

    Ok(Json(ConvertResponse {
        svg: out.svg,
        width: out.width,
        height: out.height,
        path_count: out.path_count,
        svg_bytes: out.svg_bytes,
        elapsed_ms: t.elapsed().as_millis(),
    }))
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/api/convert", post(api_convert))
        .layer(DefaultBodyLimit::max(30 * 1024 * 1024));

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5173);
    let addr = format!("127.0.0.1:{port}");
    println!("im2vec-web dev server: http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>im2vec — image to SVG</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #0f1115; color: #e8eaf0; }
  header { padding: 20px 24px; border-bottom: 1px solid #262b36; }
  header h1 { margin: 0; font-size: 20px; } header p { margin: 4px 0 0; color: #9aa3b2; font-size: 13px;}
  main { display: grid; grid-template-columns: 320px 1fr; gap: 16px; padding: 16px 24px; }
  .panel { background: #171b22; border: 1px solid #262b36; border-radius: 12px; padding: 16px; }
  label { display: block; font-size: 12px; color: #9aa3b2; margin: 12px 0 4px; }
  select, input[type=number], input[type=text] { width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #2e3542; background: #0f1319; color: inherit; }
  input[type=file] { width: 100%; font-size: 13px; }
  input[type=range] { width: 100%; }
  button { margin-top: 14px; width: 100%; padding: 10px; border: 0; border-radius: 10px; background: #5b8cff; color: white; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
  .views { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .view { min-height: 420px; display: flex; align-items: center; justify-content: center; background: repeating-conic-gradient(#1c212b 0 25%, #171b22 0 50%) 0 0/24px 24px; border-radius: 8px; overflow: auto; padding: 12px;}
  .view img, .view svg { max-width: 100%; max-height: 60vh; background: white; border-radius: 6px;}
  .stats { font-size: 12px; color: #9aa3b2; margin-top: 8px; white-space: pre-wrap;}
  .row { display: flex; gap: 8px; } .row > * { flex: 1; }
  a.dl { display: none; margin-top: 8px; font-size: 13px; color: #8fb4ff; }
  .hint { font-size: 12px; color: #9aa3b2; margin-top: 6px; min-height: 16px; word-break: break-all;}
  body.dragover::after { content: 'Drop image to convert'; position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 28px; font-weight: 700; background: rgba(91,140,255,.18); backdrop-filter: blur(2px); border: 4px dashed #5b8cff; z-index: 99; pointer-events: none; }
  @media (max-width: 900px){ main{grid-template-columns:1fr;} .views{grid-template-columns:1fr;} }
</style>
</head>
<body>
<header><h1>im2vec — image → SVG</h1><p>Logo-first tracer (Rust + vtracer). <b>Paste</b> (⌘V / Ctrl+V), drag &amp; drop, or pick a file — it converts instantly. Download only if you like the result.</p></header>
<main>
  <div class="panel">
    <label>Image (png / jpg / webp) — paste / drop / browse</label>
    <input id="file" type="file" accept="image/*"/>
    <div class="hint" id="fname">No image yet — paste one anywhere on this page.</div>
    <label>Preset</label>
    <select id="preset">
      <option value="logo" selected>logo — flat, few colors</option>
      <option value="illustration">illustration — more colors</option>
      <option value="photo">photo — keeps gradients/shadows</option>
      <option value="mono">mono — black &amp; white line art</option>
    </select>
    <div class="row">
      <div><label>Mode</label><select id="mode"><option value="spline" selected>spline</option><option value="polygon">polygon</option><option value="pixel">pixel</option></select></div>
      <div><label>Layers</label><select id="hierarchical"><option value="stacked" selected>stacked</option><option value="cutout">cutout (seam-free)</option></select></div>
    </div>
    <label>Colors (max) — empty = auto <span id="maxv">8</span></label>
    <input id="max_colors" type="range" min="2" max="64" value="8"/>
    <label>Speckle filter <span id="speckv">4</span></label>
    <input id="filter_speckle" type="range" min="0" max="32" value="4"/>
    <label>Simplify (px, 0 = off) <span id="simpv">1.0</span></label>
    <input id="simplify" type="range" min="0" max="30" value="10"/>
    <div class="row">
      <div><label>Color precision</label><select id="color_precision"><option>4</option><option>5</option><option selected>6</option><option>7</option><option>8</option></select></div>
      <div><label>Path precision</label><select id="path_precision"><option>1</option><option selected>2</option><option>3</option></select></div>
    </div>
    <div class="row">
      <div><label>Segmentation</label><select id="clustering"><option value="color-cluster" selected>color-cluster (flat)</option><option value="watershed">watershed (gradients)</option><option value="binary">binary (mono)</option></select></div>
      <div><label>Watershed detail</label><select id="watershed_detail"><option value="64">64 — small</option><option value="128">128 — medium</option><option value="160" selected>160 — high</option><option value="192">192 — max</option></select></div>
    </div>
    <button id="go">Convert to SVG</button>
    <div class="stats" id="stats">No conversion yet.</div>
    <a class="dl" id="dl" style="display:none">Download SVG</a>
  </div>
  <div class="panel">
    <div class="views">
      <div><label>Original</label><div class="view" id="orig"><span style="color:#666">—</span></div></div>
      <div><label>Vectorized SVG (rendered)</label><div class="view" id="vec"><span style="color:#666">—</span></div></div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let lastSvg = "";
let currentFile = null;
let currentName = "";
$('max_colors').oninput = e => $('maxv').textContent = e.target.value;
$('filter_speckle').oninput = e => $('speckv').textContent = e.target.value;
$('simplify').oninput = e => $('simpv').textContent = (e.target.value/10).toFixed(1);
$('preset').onchange = e => {
  const v = e.target.value;
  if (v === 'logo') $('max_colors').value = 8;
  if (v === 'illustration') $('max_colors').value = 16;
  if (v === 'photo') $('max_colors').value = 64;
  if (v === 'mono') $('max_colors').value = 8;
  $('maxv').textContent = $('max_colors').value;
  if (v === 'photo') { $('clustering').value = 'watershed'; $('watershed_detail').value = '160'; }
  if (v === 'logo' || v === 'mono') $('clustering').value = v === 'mono' ? 'binary' : 'color-cluster';
};
// Accept an image from any source (browse / paste / drop) and auto-convert it.
function setFile(f, name) {
  currentFile = f;
  currentName = name || f.name || 'pasted-image.png';
  const url = URL.createObjectURL(f);
  $('orig').innerHTML = `<img src="${url}"/>`;
  $('fname').textContent = currentName;
  convert();
}
$('file').onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  setFile(f, f.name);
};
// Paste: copy any image (screenshot, right-click → Copy image) then ⌘V/Ctrl+V here.
window.addEventListener('paste', e => {
  const items = (e.clipboardData && e.clipboardData.items) || [];
  for (const it of items) {
    if (it.type && it.type.startsWith('image/')) {
      const f = it.getAsFile();
      if (f) { e.preventDefault(); setFile(f, 'pasted-image.png'); return; }
    }
  }
});
// Drag & drop anywhere on the page.
window.addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('dragover'); });
window.addEventListener('dragleave', e => { if (e.relatedTarget === null) document.body.classList.remove('dragover'); });
window.addEventListener('drop', e => {
  e.preventDefault(); document.body.classList.remove('dragover');
  const f = (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) || null;
  if (f && f.type.startsWith('image/')) setFile(f, f.name);
});
async function convert() {
  const f = currentFile || $('file').files[0];
  if (!f) { alert('Paste, drop, or pick an image first'); return; }
  $('go').disabled = true; $('stats').textContent = 'Converting…';
  const fd = new FormData();
  fd.append('file', f, currentName || 'image.png');
  fd.append('preset', $('preset').value);
  fd.append('mode', $('mode').value);
  fd.append('hierarchical', $('hierarchical').value);
  fd.append('filter_speckle', $('filter_speckle').value);
  fd.append('color_precision', $('color_precision').value);
  fd.append('path_precision', $('path_precision').value);
  fd.append('clustering', $('clustering').value);
  fd.append('watershed_detail', $('watershed_detail').value);
  const mc = $('preset').value === 'photo' ? '' : $('max_colors').value;
  fd.append('max_colors', mc);
  const s = ($('simplify').value/10).toFixed(1);
  fd.append('simplify', s === '0.0' ? 'off' : s);
  try {
    const r = await fetch('/api/convert', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    lastSvg = j.svg;
    $('vec').innerHTML = j.svg;
    $('stats').textContent = `${j.width}x${j.height}  •  ${j.path_count} paths  •  ${(j.svg_bytes/1024).toFixed(1)} KB svg  •  ${j.elapsed_ms} ms`;
    const blob = new Blob([j.svg], {type:'image/svg+xml'});
    const dlName = (currentName.replace(/\.[^.]+$/, '') || 'output') + '.svg';
    const a = $('dl'); a.style.display='inline'; a.href = URL.createObjectURL(blob); a.download = dlName;
    a.textContent = '⬇ Download ' + dlName;
  } catch(e){ $('stats').textContent = 'Error: ' + e.message; }
  $('go').disabled = false;
}
$('go').onclick = convert;
</script>
</body>
</html>"#;
