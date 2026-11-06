"use strict";
const $ = (id) => document.getElementById(id);
const COLOURS = ["#e6194b", "#3cb44b", "#c9a800", "#4363d8", "#f58231", "#911eb4", "#008b8b", "#7a4b2a"];
let info = null, file = null, result = null, previewImg = null;

function text(el, s) { el.textContent = s; }
function pct(x) { return Math.round(x * 100) + "%"; }

async function loadInfo() {
  try {
    const r = await fetch("/api/info");
    info = await r.json();
    const ul = $("limits");
    info.limitations.forEach((t) => { const li = document.createElement("li"); li.textContent = t; ul.appendChild(li); });
    if (info.warnings && info.warnings.length) { text($("status"), "Setup notice: " + info.warnings.join(" ")); }
  } catch (e) { showError("Could not reach the server."); }
}

function showError(msg) { const e = $("error"); e.hidden = false; text(e, msg); }
function clearError() { $("error").hidden = true; text($("error"), ""); }

function setFile(f) {
  file = f; clearError(); $("result").hidden = true; $("explain").hidden = true;
  text($("filename"), f ? f.name + " (" + Math.round(f.size / 1024) + " KB)" : "No file selected");
  $("check").disabled = !f;
}

$("file").addEventListener("change", (e) => setFile(e.target.files[0] || null));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]); });

async function errorMessage(r) {
  try { const j = await r.json(); return typeof j.detail === "string" ? j.detail : "The request was not accepted."; }
  catch (e) { return "Unexpected server response (" + r.status + ")."; }
}

$("check").addEventListener("click", async () => {
  if (!file) return;
  clearError(); $("check").disabled = true; text($("status"), "Checking the fabric image...");
  const form = new FormData(); form.append("file", file);
  try {
    const r = await fetch("/api/predict", { method: "POST", body: form });
    if (!r.ok) { showError(await errorMessage(r)); text($("status"), ""); return; }
    result = await r.json(); text($("status"), ""); renderResult();
  } catch (e) { showError("Network error: could not reach the server."); text($("status"), ""); }
  finally { $("check").disabled = false; }
});

function renderResult() {
  $("result").hidden = false; $("explain").hidden = true;
  const v = $("verdict");
  const unusual = result.verdict.reason === "unusual_input";
  v.className = unusual ? "UNUSUAL" : result.verdict.decision;
  text(v, (unusual ? "REFER (unusual input): " : result.verdict.decision + ": ") + result.verdict.message);
  const ow = $("oodwarn");
  if (result.ood && result.ood.out_of_distribution) {
    ow.hidden = false;
    text(ow, "This image is unlike the training images (distance " + result.ood.distance.toFixed(1) + " vs typical limit " + result.ood.limit.toFixed(1) + "). Detections below may be unreliable.");
  } else { ow.hidden = true; }
  const slider = $("minshow");
  slider.value = Math.min(result.threshold, 1); text($("minshowval"), Number(slider.value).toFixed(2));
  const tv = info && info.threshold_validation;
  let t = "Decision threshold: " + result.threshold.toFixed(2) + " (" + result.threshold_source + ").";
  if (tv && tv.any_defect_recall != null) {
    t += " On validation images this threshold caught " + pct(tv.any_defect_recall) + " of images with defects and wrongly flagged " + pct(tv.false_alarm_rate_on_defect_free_images || 0) + " of defect-free images.";
  }
  text($("thrtext"), t);
  text($("meta"), "Model: " + result.model_version + " (" + result.backend + ") | processing time " + result.inference_ms + " ms | image " + result.image.original_size.join("x") + " px");
  previewImg = new Image();
  previewImg.onload = () => { drawCanvas(); renderTable(); };
  previewImg.src = "data:image/jpeg;base64," + result.preview_jpeg;
}

function visible() { const m = Number($("minshow").value); return result.detections.filter((d) => d.score >= m); }

function drawCanvas() {
  const c = $("canvas"), ctx = c.getContext("2d");
  c.width = previewImg.width; c.height = previewImg.height;
  ctx.drawImage(previewImg, 0, 0);
  visible().forEach((d) => {
    const [x1, y1, x2, y2] = d.box_norm;
    ctx.lineWidth = Math.max(2, c.width / 300);
    ctx.strokeStyle = COLOURS[d.cls % COLOURS.length];
    ctx.setLineDash(d.above_threshold ? [] : [6, 4]);
    ctx.strokeRect(x1 * c.width, y1 * c.height, (x2 - x1) * c.width, (y2 - y1) * c.height);
  });
  ctx.setLineDash([]);
}

function renderTable() {
  const tb = document.querySelector("#dets tbody"); tb.textContent = "";
  const list = visible();
  if (!list.length) { const tr = document.createElement("tr"); const td = document.createElement("td"); td.colSpan = 4; td.textContent = "No detections at this display level."; tr.appendChild(td); tb.appendChild(tr); return; }
  list.forEach((d) => {
    const tr = document.createElement("tr"); if (!d.above_threshold) tr.className = "weak";
    const t1 = document.createElement("td"); const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = COLOURS[d.cls % COLOURS.length];
    t1.appendChild(sw); t1.appendChild(document.createTextNode(d.name + (d.above_threshold ? "" : " (weak)")));
    const t2 = document.createElement("td"); t2.textContent = d.score.toFixed(2);
    const t3 = document.createElement("td");
    if (d.validation && d.validation.available) t3.textContent = "about " + pct(d.validation.precision) + " (n=" + d.validation.n + ")";
    else t3.textContent = d.validation ? "too few validation examples" : "n/a";
    const t4 = document.createElement("td"); const b = document.createElement("button"); b.className = "small"; b.textContent = "Why?";
    b.addEventListener("click", () => explain(d)); t4.appendChild(b);
    [t1, t2, t3, t4].forEach((t) => tr.appendChild(t)); tb.appendChild(tr);
  });
}

$("minshow").addEventListener("input", () => { text($("minshowval"), Number($("minshow").value).toFixed(2)); if (result) { drawCanvas(); renderTable(); } });

async function explain(d) {
  $("explain").hidden = false; $("explainpanels").textContent = "";
  text($("explainstatus"), "Computing explanation for the " + d.name + " detection (this can take a few seconds)...");
  $("explain").scrollIntoView({ behavior: "smooth" });
  const form = new FormData();
  form.append("file", file);
  ["x1", "y1", "x2", "y2"].forEach((k, i) => form.append(k, d.box_norm[i]));
  form.append("cls", d.cls);
  try {
    const r = await fetch("/api/explain", { method: "POST", body: form });
    if (!r.ok) { text($("explainstatus"), ""); showError(await errorMessage(r)); return; }
    const j = await r.json(); text($("explainstatus"), j.note ? "Note: " + j.note : "Done in " + j.seconds + " s.");
    const names = { occlusion: "Hiding test (what happens if this area is hidden?)", eigencam: "Overview (where the network's features respond most)" };
    Object.entries(j.methods).forEach(([m, e]) => {
      const p = document.createElement("div"); p.className = "panel";
      const h = document.createElement("h3"); h.textContent = names[m] || m; p.appendChild(h);
      if (!e.available) { const q = document.createElement("p"); q.className = "muted small"; q.textContent = "Not available: " + (e.reason || ""); p.appendChild(q); }
      else {
        const img = document.createElement("img"); img.alt = "Explanation overlay for " + m; img.src = "data:image/png;base64," + e.overlay_png; p.appendChild(img);
        if (e.drop_when_hiding_highlight !== undefined) {
          const s = document.createElement("div"); s.className = "sanity";
          s.textContent = "Check: hiding the highlighted area changes the detector's score from " + j.score_before.toFixed(2) + " to " + e.score_after_hiding_highlight.toFixed(2) +
            ". Hiding other areas of the same size lowers it by only " + Math.max(0, e.drop_when_hiding_random).toFixed(2) + " on average. " +
            (e.drop_when_hiding_highlight > e.drop_when_hiding_random + 0.05 ? "So the highlighted area matters more than a random area." : "So this highlight is NOT clearly more important than a random area: treat it with caution.");
          p.appendChild(s);
        }
      }
      $("explainpanels").appendChild(p);
    });
  } catch (e) { text($("explainstatus"), ""); showError("Could not compute the explanation."); }
}

loadInfo();
