// Diffusers Image Server UI(軽量版、4タブ)
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ---------------- タブ切替 ----------------
$$(".tab-btn[data-tab]").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab-btn[data-tab]").forEach((b) => b.classList.toggle("active", b === btn));
    $$(".tab-panel").forEach((p) =>
      p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab)
    );
  });
});

// ---------------- ステータスバー ----------------
async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    const vram = d.vram || {};
    const parts = [];
    parts.push(d.gpu_busy ? "生成中" : "待機中");
    if (vram.allocated_gb != null) parts.push(`VRAM ${Number(vram.allocated_gb).toFixed(1)}GB`);
    parts.push(`t2i:${d.t2i_loaded ? "済" : "-"}`);
    parts.push(`edit:${d.edit_loaded ? "済" : "-"}(${d.edit_quant || "-"})`);
    parts.push(`inpaint:${d.controlnet_inpaint_loaded ? "済" : "-"}`);
    const joyai = d.joyai || {};
    parts.push(`joyai:${joyai.loaded ? "済" : "-"}`);
    $("#status-text").textContent = parts.join(" | ");
  } catch (e) {
    $("#status-text").textContent = "ステータス取得失敗";
  }
}
$("#status-refresh-btn").addEventListener("click", refreshStatus);
setInterval(refreshStatus, 8000);
refreshStatus();

$$(".unload-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      await fetch("/api/unload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: btn.dataset.target }),
      });
    } finally {
      btn.disabled = false;
      refreshStatus();
    }
  });
});

// ---------------- 進捗ポーリング ----------------
function startProgressPolling(el) {
  const timer = setInterval(async () => {
    try {
      const r = await fetch("/api/progress");
      const p = await r.json();
      if (p && p.phase) {
        let text = `${p.mode || ""} ${p.phase}`;
        if (p.total_steps) text += ` ${p.step || 0}/${p.total_steps}`;
        el.textContent = text;
      }
    } catch (e) { /* ignore */ }
  }, 700);
  return () => { clearInterval(timer); el.textContent = ""; };
}

function showError(el, msg) {
  el.textContent = msg;
  el.classList.remove("hidden");
}
function clearError(el) {
  el.textContent = "";
  el.classList.add("hidden");
}
function showResult(panel, meta) {
  panel.innerHTML = "";
  const img = document.createElement("img");
  img.src = meta.image_url + "?t=" + Date.now();
  img.style.maxWidth = "100%";
  panel.appendChild(img);
  const info = document.createElement("p");
  info.className = "form-hint";
  const bits = [];
  if (meta.elapsed_s != null) bits.push(`${Number(meta.elapsed_s).toFixed(1)}s`);
  if (meta.peak_vram_gb != null) bits.push(`peak ${Number(meta.peak_vram_gb).toFixed(1)}GB`);
  if (meta.seed != null) bits.push(`seed ${meta.seed}`);
  if (meta.quant) bits.push(`quant ${meta.quant}`);
  info.textContent = bits.join(" / ");
  panel.appendChild(info);
}

// ---------------- 生成/編集タブ(edit-form) ----------------
// モデル選択に応じて表示するパラメータ群を切り替える
function editModelKind() {
  const v = $("#edit-model-select").value;
  if (v.startsWith("joyai")) return "joyai";
  if (v.startsWith("qwen-2512")) return "qwen2512";
  return "qwenedit";
}

function updateEditFields() {
  const kind = editModelKind();
  $$(".qwen-edit-only-fields").forEach((el) => el.classList.toggle("hidden", kind !== "qwenedit"));
  $$(".joyai-only-fields").forEach((el) => el.classList.toggle("hidden", kind !== "joyai"));
  $$(".qwen-2512-only-fields").forEach((el) => el.classList.toggle("hidden", kind !== "qwen2512"));
}
$("#edit-model-select").addEventListener("change", updateEditFields);
updateEditFields();

// 生成中の進捗表示(各タブの結果パネル上に動的に置く)
function ensureProgressEl(panelSel) {
  const panel = $(panelSel);
  let el = panel.parentElement.querySelector(".gen-progress-line");
  if (!el) {
    el = document.createElement("p");
    el.className = "form-hint gen-progress-line";
    panel.parentElement.insertBefore(el, panel);
  }
  return el;
}

$("#edit-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const errEl = $("#edit-error");
  clearError(errEl);
  const kind = editModelKind();
  const model = $("#edit-model-select").value;
  const gguf = model.endsWith("-gguf");
  const files = Array.from(form.images.files).filter((f) => f.size);
  const stopProg = startProgressPolling(ensureProgressEl("#edit-result-panel"));
  const btn = form.querySelector(".primary-btn");
  btn.disabled = true;
  try {
    let resp;
    const prompt = form.prompt.value;
    const neg = form.negative_prompt.value;
    if (kind === "joyai") {
      if (!files.length) throw new Error("JoyAI-Edit-Plus は参照画像が1〜3枚必要です");
      const fd = new FormData();
      files.forEach((f) => fd.append("images", f));
      fd.append("prompt", prompt);
      fd.append("negative_prompt", neg);
      fd.append("steps", form.joyai_steps.value);
      fd.append("guidance_scale", form.joyai_guidance_scale.value);
      fd.append("seed", form.joyai_seed.value);
      if (form.joyai_width.value) fd.append("width", form.joyai_width.value);
      if (form.joyai_height.value) fd.append("height", form.joyai_height.value);
      if (gguf) fd.append("quant", "gguf-q4_k_m");
      resp = await fetch("/api/joyai/edit", { method: "POST", body: fd });
    } else if (kind === "qwen2512") {
      const quant = gguf ? "gguf-q4_k_m" : "fp8-lightning";
      if (files.length === 0) {
        resp = await fetch("/api/t2i", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt, negative_prompt: neg,
            width: Number(form.qwen2512_width.value) || 1024,
            height: Number(form.qwen2512_height.value) || 1024,
            steps: Number(form.qwen2512_steps.value),
            cfg: Number(form.qwen2512_cfg.value),
            seed: Number(form.qwen2512_seed.value),
            lightning: form.qwen2512_lightning.checked,
            model: "2512", quant,
          }),
        });
      } else {
        const fd = new FormData();
        fd.append("image", files[0]);
        fd.append("prompt", prompt);
        fd.append("negative_prompt", neg);
        fd.append("strength", form.qwen2512_strength.value || "0.6");
        fd.append("steps", form.qwen2512_steps.value);
        fd.append("cfg", form.qwen2512_cfg.value);
        fd.append("seed", form.qwen2512_seed.value);
        fd.append("lightning", form.qwen2512_lightning.checked ? "true" : "false");
        fd.append("model", "2512");
        fd.append("quant", quant);
        resp = await fetch("/api/i2i", { method: "POST", body: fd });
      }
    } else {
      // Qwen-Image-Edit-2511(通常Edit / strength指定でI2Iモード)
      const fd = new FormData();
      files.forEach((f) => fd.append("images", f));
      fd.append("prompt", prompt);
      fd.append("negative_prompt", neg);
      fd.append("steps", form.steps.value);
      fd.append("cfg", form.cfg.value);
      fd.append("seed", form.seed.value);
      if (form.shift.value) fd.append("shift", form.shift.value);
      if (form.width.value) fd.append("width", form.width.value);
      if (form.height.value) fd.append("height", form.height.value);
      if (form.strength.value) fd.append("strength", form.strength.value);
      fd.append("lightning", form.lightning.checked ? "true" : "false");
      if (gguf) fd.append("quant", "gguf-q4_k_m");
      resp = await fetch("/api/edit", { method: "POST", body: fd });
    }
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);
    showResult($("#edit-result-panel"), data);
  } catch (e) {
    showError(errEl, String(e.message || e));
  } finally {
    btn.disabled = false;
    stopProg();
    refreshStatus();
  }
});

// ---------------- LLM プロンプト支援(全フォーム共通、フォーム毎のUndo付き) ----------------
$$(".gen-form").forEach((form) => {
  const prompt = form.querySelector('textarea[name="prompt"]');
  const undoBtn = form.querySelector(".llm-undo-btn");
  if (!prompt) return;
  let prev = null;
  const call = async (path, btn) => {
    const text = prompt.value.trim();
    if (!text) return;
    btn.disabled = true;
    const orig = btn.textContent;
    btn.textContent = "処理中…";
    try {
      const r = await fetch(path, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || r.statusText);
      prev = prompt.value;
      prompt.value = d.result;
      if (undoBtn) undoBtn.classList.remove("hidden");
    } catch (e) {
      const errEl = form.parentElement.querySelector(".form-error");
      if (errEl) showError(errEl, "LLM: " + String(e.message || e));
    } finally {
      btn.disabled = false;
      btn.textContent = orig;
    }
  };
  form.querySelectorAll(".llm-enhance-btn").forEach((b) =>
    b.addEventListener("click", () => call("/api/prompt/enhance", b)));
  form.querySelectorAll(".llm-translate-btn").forEach((b) =>
    b.addEventListener("click", () => call("/api/prompt/translate", b)));
  if (undoBtn) undoBtn.addEventListener("click", () => {
    if (prev != null) {
      prompt.value = prev;
      prev = null;
      undoBtn.classList.add("hidden");
    }
  });
});

// ---------------- Inpaint タブ(マスクエディタ + 送信) ----------------
const inpaintMask = { active: false, mode: "brush", drawing: false, dirty: false };

function inpaintSetupCanvases(img) {
  const base = $("#inpaint-base-canvas");
  const mask = $("#inpaint-mask-canvas");
  base.width = mask.width = img.naturalWidth;
  base.height = mask.height = img.naturalHeight;
  base.getContext("2d").drawImage(img, 0, 0);
  mask.getContext("2d").clearRect(0, 0, mask.width, mask.height);
  inpaintMask.dirty = false;
  $("#inpaint-mask-editor").classList.remove("hidden");
}

$("#inpaint-image-input").addEventListener("change", () => {
  const f = $("#inpaint-image-input").files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => inpaintSetupCanvases(img);
  img.src = URL.createObjectURL(f);
});

(() => {
  const mask = $("#inpaint-mask-canvas");
  const pos = (e) => {
    const r = mask.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (mask.width / r.width),
      y: (e.clientY - r.top) * (mask.height / r.height),
    };
  };
  const draw = (e) => {
    const ctx = mask.getContext("2d");
    const { x, y } = pos(e);
    const size = Number($("#mask-brush-size").value);
    ctx.globalCompositeOperation =
      inpaintMask.mode === "brush" ? "source-over" : "destination-out";
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.beginPath();
    ctx.arc(x, y, size * (mask.width / mask.getBoundingClientRect().width) / 2, 0, Math.PI * 2);
    ctx.fill();
    inpaintMask.dirty = true;
  };
  mask.addEventListener("pointerdown", (e) => {
    inpaintMask.drawing = true;
    mask.setPointerCapture(e.pointerId);
    draw(e);
  });
  mask.addEventListener("pointermove", (e) => { if (inpaintMask.drawing) draw(e); });
  ["pointerup", "pointercancel"].forEach((ev) =>
    mask.addEventListener(ev, () => { inpaintMask.drawing = false; }));
  $("#mask-mode-brush").addEventListener("click", () => {
    inpaintMask.mode = "brush";
    $("#mask-mode-brush").classList.add("active-tool");
    $("#mask-mode-eraser").classList.remove("active-tool");
  });
  $("#mask-mode-eraser").addEventListener("click", () => {
    inpaintMask.mode = "eraser";
    $("#mask-mode-eraser").classList.add("active-tool");
    $("#mask-mode-brush").classList.remove("active-tool");
  });
  $("#mask-clear-btn").addEventListener("click", () => {
    mask.getContext("2d").clearRect(0, 0, mask.width, mask.height);
    inpaintMask.dirty = false;
  });
})();

// マスクキャンバス(透明+白)を黒背景の白黒マスクPNGにして返す
function inpaintMaskBlob() {
  return new Promise((resolve) => {
    const mask = $("#inpaint-mask-canvas");
    const out = document.createElement("canvas");
    out.width = mask.width;
    out.height = mask.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(mask, 0, 0);
    out.toBlob(resolve, "image/png");
  });
}

$("#inpaint-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const errEl = $("#inpaint-error");
  clearError(errEl);
  const stopProg = startProgressPolling(ensureProgressEl("#inpaint-result-panel"));
  const btn = form.querySelector(".primary-btn");
  btn.disabled = true;
  try {
    const imgFile = form.image.files[0];
    if (!imgFile) throw new Error("元画像を選択してください");
    const uploaded = $("#inpaint-mask-upload").files[0];
    let maskBlob = uploaded;
    if (!maskBlob) {
      if (!inpaintMask.dirty) throw new Error("マスクを描くか、マスク画像をアップロードしてください");
      maskBlob = await inpaintMaskBlob();
    }
    const fd = new FormData();
    fd.append("image", imgFile);
    fd.append("mask_image", maskBlob, "mask.png");
    fd.append("prompt", form.prompt.value);
    fd.append("negative_prompt", form.negative_prompt.value);
    fd.append("controlnet_conditioning_scale", form.controlnet_conditioning_scale.value);
    fd.append("steps", form.steps.value);
    fd.append("cfg", form.cfg.value);
    fd.append("seed", form.seed.value);
    const resp = await fetch("/api/inpaint", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);
    showResult($("#inpaint-result-panel"), data);
  } catch (e) {
    showError(errEl, String(e.message || e));
  } finally {
    btn.disabled = false;
    stopProg();
    refreshStatus();
  }
});

// ---------------- アウトペイント タブ ----------------
$("#outpaint-form").querySelector('input[name="image"]').addEventListener("change", (e) => {
  const f = e.target.files[0];
  const prev = $("#outpaint-preview");
  if (f) {
    prev.src = URL.createObjectURL(f);
    prev.classList.remove("hidden");
  }
});

$("#outpaint-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const errEl = $("#outpaint-error");
  clearError(errEl);
  const stopProg = startProgressPolling(ensureProgressEl("#outpaint-result-panel"));
  const btn = form.querySelector(".primary-btn");
  btn.disabled = true;
  try {
    const fd = new FormData(form);
    const resp = await fetch("/api/outpaint", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);
    showResult($("#outpaint-result-panel"), data);
  } catch (e) {
    showError(errEl, String(e.message || e));
  } finally {
    btn.disabled = false;
    stopProg();
    refreshStatus();
  }
});

// ---------------- キャラシート タブ ----------------
let csJobId = null;
let csPollTimer = null;
let csFile = null;            // 選択済みファイル(File)
let csSplitRef = null;        // {split_id, figure_index} を使う場合
const csRendered = new Set(); // "key:url" 描画済み管理

function csShowError(msg) {
  const el = $("#cs-error-message");
  el.textContent = msg || "";
  el.classList.toggle("hidden", !msg);
}

function csShowStatus(msg) {
  const el = $("#cs-status-message");
  el.textContent = msg || "";
  el.classList.toggle("hidden", !msg);
}

function csSetInput(file) {
  csFile = file;
  csSplitRef = null;
  $("#cs-preview-image").src = URL.createObjectURL(file);
  $("#cs-preview-image").classList.remove("hidden");
  $("#cs-dropzone-text").innerHTML = file.name;
  $("#cs-generate-btn").disabled = false;
  csShowError("");
  csDetectSplit(file);
}

// 複数キャラ検出(失敗しても全体を使えばよいので握りつぶす)
async function csDetectSplit(file) {
  const section = $("#cs-split-section");
  section.classList.add("hidden");
  $("#cs-split-grid").innerHTML = "";
  try {
    const fd = new FormData();
    fd.append("image", file);
    const r = await fetch("/api/charsheet/split", { method: "POST", body: fd });
    if (!r.ok) return;
    const d = await r.json();
    if (!d.figures || d.count < 2) return;
    $("#cs-split-message").textContent =
      `${d.count} 体のキャラクターを検出しました。使う1体を選択してください。`;
    d.figures.forEach((fig) => {
      const cell = document.createElement("div");
      cell.className = "split-cell";
      cell.innerHTML = `<span class="split-cell-num">${fig.index + 1}</span><img src="${fig.url}">`;
      cell.addEventListener("click", () => {
        csSplitRef = { split_id: d.split_id, figure_index: fig.index };
        $("#cs-split-grid").querySelectorAll(".split-cell").forEach((el) =>
          el.classList.toggle("selected", el === cell));
        $("#cs-preview-image").src = fig.url;
      });
      $("#cs-split-grid").appendChild(cell);
    });
    section.classList.remove("hidden");
  } catch (e) { /* 検出失敗は無視して全体を使う */ }
}

$("#cs-use-whole-link").addEventListener("click", (e) => {
  e.preventDefault();
  csSplitRef = null;
  $("#cs-split-grid").querySelectorAll("img").forEach((el) => el.classList.remove("selected"));
  if (csFile) $("#cs-preview-image").src = URL.createObjectURL(csFile);
});

// --- 入力: クリック選択 + D&D(URL/file:///MIME空対応) ---
$("#cs-dropzone").addEventListener("click", () => $("#cs-file-input").click());
$("#cs-file-input").addEventListener("change", () => {
  const f = $("#cs-file-input").files[0];
  if (f) csSetInput(f);
});

// 枠外ドロップでのページ遷移を防止(ページ全体)
["dragover", "drop"].forEach((ev) =>
  window.addEventListener(ev, (e) => e.preventDefault())
);

(() => {
  const zone = $("#cs-dropzone");
  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
    })
  );
  zone.addEventListener("drop", async (e) => {
    const isImage = (f) => f.type.startsWith("image/") ||
      (!f.type && /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
    const f = [...(e.dataTransfer?.files || [])].find(isImage);
    if (f) return csSetInput(f);
    const url = (e.dataTransfer?.getData("text/uri-list") ||
                 e.dataTransfer?.getData("text/plain") || "").split("\n")[0].trim();
    if (!url) {
      csShowError("ドロップから画像を取り出せませんでした。クリックで選択してください。");
      return;
    }
    csShowStatus("画像を取得中…");
    try {
      let resp = null;
      if (!url.startsWith("file:")) {
        try {
          resp = await fetch(url, { mode: "cors" });
          if (!resp.ok) throw new Error(resp.status);
        } catch (_) { resp = null; }
      }
      if (!resp) {
        resp = await fetch(`/api/fetch-image?url=${encodeURIComponent(url)}`);
        if (!resp.ok) {
          const detail = (await resp.json().catch(() => null))?.detail;
          throw new Error(detail || resp.status);
        }
      }
      const blob = await resp.blob();
      if (!blob.type.startsWith("image/")) throw new Error("画像ではありません");
      const name = url.startsWith("data:") ? "dropped.png"
        : (decodeURIComponent(url.split("/").pop().split("?")[0]) || "dropped.png");
      csSetInput(new File([blob], name, { type: blob.type }));
      csShowStatus("");
    } catch (err) {
      csShowStatus("");
      csShowError(`取り込めませんでした(${err.message})。クリックで選択してください。`);
    }
  });
})();

// --- 生成とポーリング ---
$("#cs-generate-btn").addEventListener("click", async () => {
  if (!csFile && !csSplitRef) return;
  csShowError("");
  $("#cs-generate-btn").disabled = true;
  try {
    const fd = new FormData();
    fd.append("seed", $("#cs-seed-input").value || "0");
    if (csSplitRef) {
      fd.append("split_id", csSplitRef.split_id);
      fd.append("figure_index", String(csSplitRef.figure_index));
    } else {
      fd.append("image", csFile);
    }
    const resp = await fetch("/api/charsheet/generate", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);
    csOpenJob(data.job_id);
  } catch (e) {
    csShowError(String(e.message || e));
    $("#cs-generate-btn").disabled = false;
  }
});

function csOpenJob(jobId) {
  csJobId = jobId;
  csRendered.clear();
  $("#cs-views-grid").innerHTML = "";
  $("#cs-sheet-panel").classList.add("hidden");
  csStartPolling();
}

function csStartPolling() {
  if (csPollTimer) clearInterval(csPollTimer);
  csPoll();
  csPollTimer = setInterval(csPoll, 1500);
}

async function csPoll() {
  if (!csJobId) return;
  let job;
  try {
    const r = await fetch(`/api/charsheet/jobs/${csJobId}`);
    if (!r.ok) return;
    job = await r.json();
  } catch (e) { return; }
  csRenderJob(job);
  if (job.status === "done" || job.status === "error") {
    clearInterval(csPollTimer);
    csPollTimer = null;
    $("#cs-generate-btn").disabled = !(csFile || csSplitRef);
    csRefreshGallery();
  }
}

const CS_STATUS_JA = {
  queued: "待機", running: "生成中", done: "完了", error: "失敗",
  refining: "修正中", removing_bg: "背景削除中",
};

function csRenderJob(job) {
  const grid = $("#cs-views-grid");
  (job.views || []).forEach((v) => {
    let tile = grid.querySelector(`[data-key="${v.key}"]`);
    if (!tile) {
      tile = document.createElement("div");
      tile.className = "view-tile";
      tile.dataset.key = v.key;
      tile.innerHTML =
        `<div class="view-tile-image-wrap"><img alt="">` +
        `<span class="view-tile-status"></span></div>` +
        `<div class="view-tile-label"><span class="ja">${v.label_ja}</span>` +
        (v.label_en ? ` <span class="en">${v.label_en}</span>` : "") + `</div>` +
        `<div class="view-tile-actions">` +
        `<button class="tiny-btn secondary view-bg-btn">背景削除</button></div>`;
      tile.querySelector(".view-bg-btn").addEventListener("click", () => csRemoveBg(v.key));
      grid.appendChild(tile);
    }
    tile.querySelector(".view-tile-status").textContent = CS_STATUS_JA[v.status] || v.status;
    const img = tile.querySelector("img");
    const mark = `${v.key}:${v.url}:${job.status}`;
    if (v.url && v.status === "done" && !csRendered.has(mark)) {
      csRendered.add(mark);
      img.src = v.url + "?t=" + Date.now();
      img.style.display = "block";
    }
  });

  // 進捗
  const wrap = $("#cs-overall-progress");
  const running = !(job.status === "done" || job.status === "error");
  wrap.classList.toggle("hidden", !running);
  if (running && job.total) {
    $("#cs-overall-progress-count").textContent = `${job.progress}/${job.total}`;
    $("#cs-overall-progress-fill").style.width =
      `${Math.round((job.progress / job.total) * 100)}%`;
  }
  csShowStatus(
    running
      ? `状態: ${CS_STATUS_JA[job.status] || job.status} (${job.progress}/${job.total})`
      : (job.error ? `エラー: ${job.error}` : (job.refine_error ? `一部失敗: ${job.refine_error}` : "完了"))
  );

  // シート + ダウンロード
  if (job.sheet_url) {
    $("#cs-sheet-image").src = job.sheet_url + "?t=" + (job.status === "done" ? Date.now() : "");
    $("#cs-sheet-panel").classList.remove("hidden");
  }
}

function csDownload(url, filename) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename || "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

$("#cs-download-zip-btn").addEventListener("click", () => {
  if (csJobId) csDownload(`/api/charsheet/jobs/${csJobId}/download.zip`, `character_sheet_${csJobId}.zip`);
});
$("#cs-download-sheet-btn").addEventListener("click", () => {
  if (csJobId) csDownload(`/api/charsheet/jobs/${csJobId}/sheet.png`, `character_sheet_${csJobId}.png`);
});
$("#cs-remove-bg-all-btn").addEventListener("click", () => csRemoveBg("all"));

async function csRemoveBg(key) {
  if (!csJobId) return;
  csShowError("");
  try {
    const r = await fetch(`/api/charsheet/jobs/${csJobId}/remove_bg`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, method: $("#cs-bgmethod-input").value }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    csRendered.clear(); // 画像が変わるので再描画させる
    csStartPolling();
  } catch (e) {
    csShowError(String(e.message || e));
  }
}

// ---------------- ギャラリー(キャラシート履歴) ----------------
async function csRefreshGallery() {
  try {
    const r = await fetch("/api/charsheet/jobs");
    if (!r.ok) return;
    const d = await r.json();
    const jobs = d.jobs || [];
    $("#gallery-count").textContent = `${jobs.length} 件`;
    const grid = $("#gallery-grid");
    grid.innerHTML = "";
    jobs.forEach((j) => {
      const card = document.createElement("div");
      card.className = "gallery-card";
      const when = new Date(j.mtime * 1000).toLocaleString("ja-JP");
      card.innerHTML =
        `<div class="gallery-card-image-wrap"><img src="${j.thumb_url}" alt="">` +
        (j.zip_url ? `<button class="tiny-btn gallery-card-download-btn">ZIP</button>` : "") +
        `</div><div class="form-hint">${when}</div>`;
      card.querySelector("img").addEventListener("click", () => {
        csOpenJob(j.job_id);
        document.querySelector('.tab-btn[data-tab="charsheet"]')?.click();
        $("#cs-views-panel").scrollIntoView({ behavior: "smooth" });
      });
      const dl = card.querySelector(".gallery-card-download-btn");
      if (dl) dl.addEventListener("click", (e) => {
        e.stopPropagation();
        csDownload(j.zip_url, `character_sheet_${j.job_id}.zip`);
      });
      grid.appendChild(card);
    });
  } catch (e) { /* ギャラリーは補助機能なので失敗は無視 */ }
}

$("#gallery-clear-btn").addEventListener("click", () => {
  // 表示のみクリア(ファイルは削除しない)。再読込で戻る。
  $("#gallery-grid").innerHTML = "";
  $("#gallery-count").textContent = "0 件(表示をクリア)";
});

csRefreshGallery();
