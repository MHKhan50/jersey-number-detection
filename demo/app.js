// JerseyNet in the browser: finds jersey numbers in an image with ONNX Runtime Web.
// The image is processed on this device and never uploaded.
import { letterbox, decode, SIZE } from "./core.js";

const MODEL_URL = "jerseynet.onnx";
const MAX_SIDE = 2048;
const $ = (id) => document.getElementById(id);
const state = { session: null, image: null, dets: [], busy: false };

function setStatus(text, kind) {
  const s = $("status");
  s.textContent = text;
  s.className = "status " + kind;
}

async function loadModel() {
  setStatus("Loading the model (about 18 MB, only the first time)…", "busy");
  ort.env.wasm.numThreads = 1;
  state.session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  setStatus("Model ready. Upload an image.", "ok");
  $("file").disabled = false;
}

function readPixels(img) {
  const k = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * k)), h = Math.max(1, Math.round(img.naturalHeight * k));
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  return { rgba: ctx.getImageData(0, 0, w, h).data, w, h, canvas: c };
}

async function analyse(img) {
  if (state.busy) return;
  state.busy = true;
  try {
    setStatus("Finding numbers…", "busy");
    const px = readPixels(img);
    const box = letterbox(px.rgba, px.w, px.h);
    const t0 = performance.now();
    const out = await state.session.run({ image: new ort.Tensor("float32", box.data, [1, 3, SIZE, SIZE]) });
    const ms = performance.now() - t0;
    state.dets = decode(out.heatmap.data, out.wh.data, out.offset.data, box);
    state.image = px;
    render();
    setStatus(`Done in ${Math.round(ms)} ms (${px.w}×${px.h} px).`, "ok");
  } catch (e) {
    console.error(e);
    setStatus("Could not analyse this image: " + e.message, "error");
  } finally {
    state.busy = false;
  }
}

function render() {
  const { image, dets } = state;
  if (!image) return;
  const thr = parseFloat($("thr").value);
  $("thrval").textContent = thr.toFixed(2);
  const shown = dets.filter((d) => d.score >= thr);
  const c = $("out");
  const scale = Math.min(1, 900 / Math.max(image.w, image.h)) * (image.w < 400 ? 3 : 1);
  c.width = Math.round(image.w * scale); c.height = Math.round(image.h * scale);
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = image.w >= 400;
  ctx.drawImage(image.canvas, 0, 0, c.width, c.height);
  const lw = Math.max(2, Math.round(c.width / 300));
  ctx.font = `600 ${Math.max(12, lw * 6)}px system-ui, sans-serif`;
  shown.forEach((d, i) => {
    const x = d.x1 * scale, y = d.y1 * scale, w = (d.x2 - d.x1) * scale, h = (d.y2 - d.y1) * scale;
    ctx.strokeStyle = "#ffd400"; ctx.lineWidth = lw; ctx.strokeRect(x, y, w, h);
    const label = `${i + 1} · ${d.score.toFixed(2)}`;
    const tw = ctx.measureText(label).width + 8, th = lw * 6 + 6;
    ctx.fillStyle = "#ffd400"; ctx.fillRect(x, Math.max(0, y - th), tw, th);
    ctx.fillStyle = "#111"; ctx.fillText(label, x + 4, Math.max(th - 5, y - 5));
  });
  $("count").textContent = shown.length === 1 ? "1 number found" : `${shown.length} numbers found`;
  $("rows").innerHTML = shown.length
    ? shown.map((d, i) => `<tr><td>${i + 1}</td><td class="num">${d.score.toFixed(3)}</td>` +
        `<td class="num">${Math.round(d.x1)}, ${Math.round(d.y1)}</td><td class="num">${Math.round(d.x2 - d.x1)} × ${Math.round(d.y2 - d.y1)}</td></tr>`).join("")
    : `<tr><td colspan="4" class="muted">Nothing above this threshold. Try lowering it.</td></tr>`;
  $("results").hidden = false;
}

function load(file) {
  if (!file || !file.type.startsWith("image/")) return setStatus("Please choose an image file.", "error");
  const img = new Image();
  img.onload = () => { analyse(img); URL.revokeObjectURL(img.src); };
  img.onerror = () => setStatus("This file could not be read as an image.", "error");
  img.src = URL.createObjectURL(file);
}

$("file").addEventListener("change", (e) => load(e.target.files[0]));
$("thr").addEventListener("input", render);
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (!state.session) return; load(e.dataTransfer.files[0]); });

loadModel().catch((e) => { console.error(e); setStatus("The model could not be loaded: " + e.message, "error"); });
