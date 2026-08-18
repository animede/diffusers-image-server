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

// ---------------- 画像生成/編集タブ ----------------
const GEN_MODES = {
  "edit":          { api: "edit",  quant: "default" },
  "edit-gguf":     { api: "edit",  quant: "gguf-q4_k_m" },
  "t2i-2512":      { api: "t2i",   model: "2512", quant: "fp8-lightning" },
  "t2i-2512-gguf": { api: "t2i",   model: "2512", quant: "gguf-q4_k_m" },
  "t2i-qwen":      { api: "t2i",   model: "qwen-image", quant: "fp8-lightning" },
  "t2i-qwen-gguf": { api: "t2i",   model: "qwen-image", quant: "gguf-q4_k_m" },
  "i2i-2512":      { api: "i2i",   model: "2512", quant: "default" },
  "i2i-qwen":      { api: "i2i",   model: "qwen-image", quant: "default" },
  "joyai":         { api: "joyai", quant: "none" },
  "joyai-gguf":    { api: "joyai", quant: "gguf-q4_k_m" },
};

$("#gen-mode-select").addEventListener("change", () => {
  const v = $("#gen-mode-select").value;
  const isEdit = v.startsWith("edit");
  const isT2i = v.startsWith("t2i");
  $("#gen-strength-label").style.display = (isEdit || v.startsWith("i2i")) ? "" : "none";
  $("#gen-lightning-row").style.display = v.startsWith("joyai") ? "none" : "";
  $("#gen-image-fields").style.display = isT2i ? "none" : "";
});

$("#gen-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const errEl = $("#gen-error");
  clearError(errEl);
  const mode = GEN_MODES[$("#gen-mode-select").value];
  const stopProg = startProgressPolling($("#gen-progress"));
  const btn = form.querySelector(".primary-btn");
  btn.disabled = true;
  try {
    let resp;
    const prompt = form.prompt.value;
    const neg = form.negative_prompt.value;
    const steps = form.steps.value, cfg = form.cfg.value, seed = form.seed.value;
    const width = form.width.value, height = form.height.value;
    const strength = form.strength.value;
    const lightning = form.lightning.checked;
    const files = form.images ? Array.from(form.images.files) : [];

    if (mode.api === "t2i") {
      const body = {
        prompt, negative_prompt: neg,
        width: Number(width) || 1328, height: Number(height) || 1328,
        steps: Number(steps), cfg: Number(cfg), seed: Number(seed),
        lightning, model: mode.model, quant: mode.quant,
      };
      resp = await fetch("/api/t2i", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } else if (mode.api === "i2i") {
      if (!files.length) throw new Error("I2I は参照画像が1枚必要です");
      const fd = new FormData();
      fd.append("image", files[0]);
      fd.append("prompt", prompt);
      fd.append("negative_prompt", neg);
      fd.append("strength", strength || "0.6");
      fd.append("steps", steps); fd.append("cfg", cfg); fd.append("seed", seed);
      fd.append("lightning", lightning ? "true" : "false");
      if (mode.model) fd.append("model", mode.model);
      fd.append("quant", mode.quant);
      resp = await fetch("/api/i2i", { method: "POST", body: fd });
    } else if (mode.api === "joyai") {
      if (!files.length) throw new Error("JoyAI は参照画像が1〜3枚必要です");
      const fd = new FormData();
      files.forEach((f) => fd.append("images", f));
      fd.append("prompt", prompt);
      fd.append("negative_prompt", neg);
      fd.append("seed", seed);
      if (width) fd.append("width", width);
      if (height) fd.append("height", height);
      fd.append("quant", mode.quant);
      resp = await fetch("/api/joyai/edit", { method: "POST", body: fd });
    } else {
      // edit(0〜3枚、strength 任意)
      const fd = new FormData();
      files.forEach((f) => fd.append("images", f));
      fd.append("prompt", prompt);
      fd.append("negative_prompt", neg);
      fd.append("steps", steps); fd.append("cfg", cfg); fd.append("seed", seed);
      fd.append("lightning", lightning ? "true" : "false");
      if (width) fd.append("width", width);
      if (height) fd.append("height", height);
      if (strength) fd.append("strength", strength);
      fd.append("quant", mode.quant);
      resp = await fetch("/api/edit", { method: "POST", body: fd });
    }
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);
    showResult($("#gen-result-panel"), data);
  } catch (e) {
    showError(errEl, String(e.message || e));
  } finally {
    btn.disabled = false;
    stopProg();
    refreshStatus();
  }
});

// LLM プロンプト支援
async function llmCall(path) {
  const form = $("#gen-form");
  const text = form.prompt.value.trim();
  if (!text) return;
  try {
    const r = await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    form.prompt.value = d.result;
  } catch (e) {
    showError($("#gen-error"), "LLM: " + String(e.message || e));
  }
}
$("#llm-enhance-btn").addEventListener("click", () => llmCall("/api/prompt/enhance"));
$("#llm-translate-btn").addEventListener("click", () => llmCall("/api/prompt/translate"));

// ---------------- Inpaint タブ ----------------
$("#inpaint-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const errEl = $("#inpaint-error");
  clearError(errEl);
  const stopProg = startProgressPolling($("#inpaint-progress"));
  const btn = form.querySelector(".primary-btn");
  btn.disabled = true;
  try {
    const fd = new FormData(form);
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
$("#outpaint-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const errEl = $("#outpaint-error");
  clearError(errEl);
  const stopProg = startProgressPolling($("#outpaint-progress"));
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

function csRenderJob(job) {
  const panel = $("#cs-result-panel");
  if (!panel.dataset.jobId || panel.dataset.jobId !== job.job_id) {
    panel.innerHTML = "";
    panel.dataset.jobId = job.job_id;
    const grid = document.createElement("div");
    grid.style.display = "grid";
    grid.style.gridTemplateColumns = "repeat(4, 1fr)";
    grid.style.gap = "6px";
    grid.id = "cs-grid";
    panel.appendChild(grid);
  }
  const grid = $("#cs-grid");
  (job.views || []).forEach((v) => {
    let tile = grid.querySelector(`[data-key="${v.key}"]`);
    if (!tile) {
      tile = document.createElement("div");
      tile.dataset.key = v.key;
      tile.dataset.rendered = "";
      const cap = document.createElement("div");
      cap.className = "form-hint";
      cap.textContent = v.label_ja;
      const img = document.createElement("img");
      img.style.width = "100%";
      img.style.display = "none";
      tile.appendChild(cap);
      tile.appendChild(img);
      grid.appendChild(tile);
    }
    const img = tile.querySelector("img");
    const cap = tile.querySelector("div");
    cap.textContent = `${v.label_ja}(${v.status})`;
    if (v.url && v.status === "done" && tile.dataset.rendered !== v.url + job.status) {
      img.src = v.url + "?t=" + Date.now();
      img.style.display = "block";
      tile.dataset.rendered = v.url + job.status;
    }
  });
  $("#cs-progress").textContent =
    `状態: ${job.status} (${job.progress}/${job.total})` + (job.error ? ` エラー: ${job.error}` : "");
  if (job.zip_url) {
    const a = $("#cs-zip-link");
    a.href = job.zip_url;
    a.classList.remove("hidden");
  }
  if (job.status === "done" || job.status === "error") {
    $("#cs-actions").classList.remove("hidden");
    const sel = $("#cs-refine-key");
    if (!sel.options.length) {
      (job.views || []).forEach((v) => {
        const o = document.createElement("option");
        o.value = v.key;
        o.textContent = v.label_ja;
        sel.appendChild(o);
      });
    }
  }
}

function csStartPolling() {
  if (csPollTimer) clearInterval(csPollTimer);
  csPollTimer = setInterval(async () => {
    if (!csJobId) return;
    try {
      const r = await fetch(`/api/charsheet/jobs/${csJobId}`);
      if (!r.ok) return;
      const job = await r.json();
      csRenderJob(job);
      if (job.status === "done" || job.status === "error") {
        clearInterval(csPollTimer);
        csPollTimer = null;
      }
    } catch (e) { /* ignore */ }
  }, 1500);
}

$("#cs-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const errEl = $("#cs-error");
  clearError(errEl);
  const btn = form.querySelector(".primary-btn");
  btn.disabled = true;
  try {
    const fd = new FormData(form);
    const resp = await fetch("/api/charsheet/generate", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);
    csJobId = data.job_id;
    $("#cs-result-panel").dataset.jobId = "";
    csStartPolling();
  } catch (e) {
    showError(errEl, String(e.message || e));
  } finally {
    btn.disabled = false;
  }
});

$("#cs-refine-btn").addEventListener("click", async () => {
  if (!csJobId) return;
  const key = $("#cs-refine-key").value;
  const instruction = $("#cs-refine-instruction").value.trim();
  if (!instruction) return showError($("#cs-error"), "修正指示を入力してください");
  clearError($("#cs-error"));
  try {
    const r = await fetch(`/api/charsheet/jobs/${csJobId}/refine`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, instruction, seed: 0 }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    csStartPolling();
  } catch (e) {
    showError($("#cs-error"), String(e.message || e));
  }
});

$("#cs-undo-btn").addEventListener("click", async () => {
  if (!csJobId) return;
  const key = $("#cs-refine-key").value;
  try {
    const r = await fetch(`/api/charsheet/jobs/${csJobId}/undo`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    csRenderJob(d);
    $("#cs-result-panel").dataset.jobId = "";
  } catch (e) {
    showError($("#cs-error"), String(e.message || e));
  }
});

$("#cs-removebg-btn").addEventListener("click", async () => {
  if (!csJobId) return;
  try {
    const r = await fetch(`/api/charsheet/jobs/${csJobId}/remove_bg`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: "all", method: "anime" }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    csStartPolling();
  } catch (e) {
    showError($("#cs-error"), String(e.message || e));
  }
});
