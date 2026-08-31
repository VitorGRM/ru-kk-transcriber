/* Mixed Russian/Kazakh transcriber — front-end controller. */
'use strict';

const $ = (sel) => document.querySelector(sel);
const LANG_LABEL = { ru: 'RUSSIAN', kk: 'KAZAKH' };
const OTHER = { ru: 'kk', kk: 'ru' };

const state = {
  jobId: null,
  filename: '',
  duration: 0,
  segments: [],
  result: null,
  textDirty: false,   // the text box has manual edits that segments no longer match
  busy: false,
};

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */
async function boot() {
  try {
    const info = await (await fetch('/api/system')).json();
    renderHardware(info);
    const sel = $('#model');
    info.models.forEach((m) => {
      const o = document.createElement('option');
      o.value = m;
      o.textContent = m + (m === info.defaults.model ? '  (recommended)' : '');
      if (m === info.defaults.model) o.selected = true;
      sel.appendChild(o);
    });
    if (info.cached_models.length === 0) $('#first-run-note').hidden = false;
  } catch (err) {
    showError('Could not reach the local server: ' + err.message);
  }
  updateRoutingHint();
  refreshGpu();
}

/* pywebview injects its bridge asynchronously and announces it with this event.
   Its presence is how the page knows it is in the desktop window rather than a
   browser tab — which changes what "Download" actually means. */
window.addEventListener('pywebviewready', () => {
  document.body.classList.add('desktop');
  const label = document.querySelector('.exports .lbl');
  if (label) label.textContent = 'Save as';
});

/* ------------------------------------------------------------------ */
/* GPU runtime                                                         */
/* ------------------------------------------------------------------ */
async function refreshGpu() {
  let g;
  try {
    g = await (await fetch('/api/gpu/status')).json();
  } catch {
    return;
  }

  const box = $('#gpubox');
  const title = $('#gpu-title');
  const body = $('#gpu-body');
  const btn = $('#gpu-install');

  // Nothing useful to say to someone without an NVIDIA GPU.
  if (!g.gpu_present) { box.hidden = true; return; }

  box.hidden = false;

  if (g.installed) {
    box.classList.add('ready');
    btn.hidden = true;
    title.textContent = `GPU acceleration is on — ${g.gpu_name}`;
    body.textContent = `The CUDA runtime is installed (${fmtGb(g.bytes_on_disk)}). `
                     + 'Transcription runs on the GPU, which is many times faster '
                     + 'than the CPU on the large model.';
    return;
  }

  box.classList.remove('ready');

  if (!g.supported) {
    btn.hidden = true;
    title.textContent = `${g.gpu_name} found, but not set up for GPU use`;
    body.textContent = 'Automatic setup is Windows-only. On Linux or macOS, '
                     + 'run ./setup.sh --gpu to install the CUDA libraries.';
    return;
  }

  btn.hidden = false;
  btn.disabled = false;
  const vram = g.vram_mb ? ` with ${(g.vram_mb / 1024).toFixed(1)} GB of VRAM` : '';
  title.textContent = `${g.gpu_name} detected${vram}`;
  body.textContent = `Transcription currently runs on the CPU. Setting up the GPU `
                   + `downloads about ${g.estimated_download_mb} MB once — no `
                   + `Python, no CUDA toolkit, no admin rights needed.`;
}

function fmtGb(bytes) {
  return bytes > 1073741824
    ? (bytes / 1073741824).toFixed(1) + ' GB'
    : Math.round(bytes / 1048576) + ' MB';
}

$('#gpu-install').addEventListener('click', async () => {
  const btn = $('#gpu-install');
  btn.disabled = true;
  btn.textContent = 'Setting up…';
  $('#gpu-bar').hidden = false;
  $('#gpu-status').hidden = false;
  $('#gpu-status').textContent = 'Starting…';

  try {
    const res = await fetch('/api/gpu/install', { method: 'POST' });
    if (!res.ok) throw new Error((await res.json()).detail || 'Could not start.');
  } catch (err) {
    showError(err.message);
    btn.disabled = false;
    btn.textContent = 'Enable GPU acceleration';
    return;
  }

  const es = new EventSource('/api/gpu/progress');
  es.onmessage = (ev) => {
    const e = JSON.parse(ev.data);
    if (e.percent != null) $('#gpu-bar-fill').style.width = e.percent + '%';
    if (e.message) $('#gpu-status').textContent = e.message;

    if (e.stage === 'done') {
      es.close();
      $('#gpu-bar-fill').style.width = '100%';
      refreshGpu();
      // The device menu was showing CPU-only advice until now.
      $('#device-hint').textContent =
        'GPU acceleration is ready. Quantisation is chosen to fit the card.';
    } else if (e.stage === 'error') {
      es.close();
      showError(e.message);
      btn.disabled = false;
      btn.textContent = 'Retry GPU setup';
      $('#gpu-bar').hidden = true;
    }
  };
  es.onerror = () => {
    es.close();
    btn.disabled = false;
    btn.textContent = 'Retry GPU setup';
  };
});

function renderHardware(info) {
  const gpu = info.hardware.gpu;
  const el = $('#hardware');
  if (gpu.available) {
    const vram = gpu.vram_mb ? ` · ${(gpu.vram_mb / 1024).toFixed(1)} GB VRAM` : '';
    el.innerHTML = `GPU: <b>${escapeHtml(gpu.name)}</b>${vram}`;
    $('#device-hint').textContent =
      'A GPU was detected. Quantisation is chosen automatically to fit its memory.';
  } else {
    el.innerHTML = `Running on <b>CPU</b> · ${info.hardware.cpu_threads} threads`;
    $('#device-hint').textContent =
      'No usable GPU found. CPU works but is markedly slower on the large model.';
  }
}

/* ------------------------------------------------------------------ */
/* File input                                                          */
/* ------------------------------------------------------------------ */
const dz = $('#dropzone');
dz.addEventListener('click', (e) => { if (e.target.id !== 'browse') $('#file-input').click(); });
dz.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file-input').click(); }
});
$('#browse').addEventListener('click', (e) => { e.stopPropagation(); $('#file-input').click(); });
$('#file-input').addEventListener('change', (e) => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
});
['dragenter', 'dragover'].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
dz.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) uploadFile(f);
});
$('#fp-clear').addEventListener('click', resetInput);

async function uploadFile(file) {
  hideError();
  $('#fp-name').textContent = file.name;
  $('#fp-meta').textContent = 'uploading…';
  $('#filepill').hidden = false;
  $('#start').disabled = true;

  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed.');

    state.jobId = data.job_id;
    state.filename = data.filename;
    state.duration = data.duration;
    $('#fp-meta').textContent =
      `${formatTime(data.duration)} · ${(data.size_bytes / 1048576).toFixed(1)} MB`;
    $('#start').disabled = false;
    $('#player').src = `/api/audio/${data.job_id}`;
  } catch (err) {
    showError(err.message);
    resetInput();
  }
}

function resetInput() {
  state.jobId = null;
  $('#filepill').hidden = true;
  $('#file-input').value = '';
  $('#start').disabled = true;
}

/* ------------------------------------------------------------------ */
/* Settings widgets                                                    */
/* ------------------------------------------------------------------ */
$('#beam').addEventListener('input', (e) => { $('#beam-out').value = e.target.value; });
$('#vad').addEventListener('input', (e) => {
  $('#vad-out').value = Number(e.target.value).toFixed(2);
});
$('#silence').addEventListener('input', (e) => { $('#sil-out').value = e.target.value; });
$('#routing').addEventListener('change', updateRoutingHint);

function updateRoutingHint() {
  const hints = {
    fast: 'One pass per segment. Quickest, but a misdetected language stays wrong.',
    balanced: 'Segments where Russian and Kazakh are close are decoded both ways and '
            + 'the better reading wins. Best accuracy for the time.',
    maximum: 'Every segment is decoded in both languages and compared. Roughly twice '
           + 'the work, and the most reliable on heavily mixed speech.',
  };
  $('#routing-hint').textContent = hints[$('#routing').value];
}

/* ------------------------------------------------------------------ */
/* Transcription                                                       */
/* ------------------------------------------------------------------ */
$('#start').addEventListener('click', startTranscription);

async function startTranscription() {
  if (!state.jobId || state.busy) return;
  hideError();
  state.busy = true;
  $('#start').disabled = true;
  $('#progress-card').hidden = false;
  $('#result-card').hidden = true;
  $('#livefeed').innerHTML = '';
  setProgress(0, 'Starting…');

  const fd = new FormData();
  fd.append('job_id', state.jobId);
  fd.append('model', $('#model').value);
  fd.append('device', $('#device').value);
  fd.append('routing_mode', $('#routing').value);
  fd.append('beam_size', $('#beam').value);
  fd.append('hotwords', $('#hotwords').value);
  fd.append('vad_aggressiveness', $('#vad').value);
  fd.append('boundary_gap_ms', $('#silence').value);
  fd.append('use_context', $('#context').checked ? 'true' : 'false');

  try {
    const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || 'Could not start.');
    listenForProgress();
  } catch (err) {
    finishBusy();
    showError(err.message);
  }
}

function listenForProgress() {
  const es = new EventSource(`/api/progress/${state.jobId}`);

  es.onmessage = (ev) => {
    const e = JSON.parse(ev.data);
    switch (e.stage) {
      case 'audio':
      case 'load':
        setProgress(2, e.message);
        break;
      case 'segmented':
        setProgress(4, `Found ${e.chunks} speech segments across `
                     + `${formatTime(e.duration)}. Transcribing…`);
        break;
      case 'chunk': {
        const pct = 4 + (e.index / e.total) * 94;
        setProgress(pct, `Segment ${e.index} of ${e.total} — `
                       + `${LANG_LABEL[e.language] || e.language}`);
        addFeedRow(e);
        break;
      }
      case 'done':
        setProgress(100, 'Finished.');
        es.close();
        finishBusy();
        renderResult(e.result);
        break;
      case 'error':
        es.close();
        finishBusy();
        showError(e.message);
        $('#progress-card').hidden = true;
        break;
    }
  };

  es.onerror = () => {
    es.close();
    if (state.busy) {
      finishBusy();
      showError('Lost the connection to the local server. Check the terminal it is '
              + 'running in for details.');
    }
  };
}

function finishBusy() {
  state.busy = false;
  $('#start').disabled = !state.jobId;
}

function setProgress(pct, msg) {
  $('#bar-fill').style.width = Math.min(100, pct) + '%';
  $('#status').textContent = msg;
}

function addFeedRow(e) {
  const row = document.createElement('div');
  row.className = 'feed-row';
  row.innerHTML = `<span class="t">${formatTime(e.start)}</span>`
                + `<span class="x">${escapeHtml((e.text || '').slice(0, 160))}</span>`;
  const feed = $('#livefeed');
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
}

/* ------------------------------------------------------------------ */
/* Result rendering                                                    */
/* ------------------------------------------------------------------ */
function renderResult(result) {
  state.result = result;
  state.segments = result.segments;
  state.textDirty = false;

  $('#progress-card').hidden = true;
  $('#result-card').hidden = false;
  renderStats(result);
  rebuildText();
  renderSegments();
  $('#result-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderStats(r) {
  const share = r.language_share || {};
  const chips = [
    `<span class="chip"><b>${formatTime(r.duration)}</b> audio</span>`,
    `<span class="chip"><b>${r.segments.length}</b> segments</span>`,
  ];
  Object.entries(share).forEach(([lang, frac]) => {
    chips.push(`<span class="chip"><b>${Math.round(frac * 100)}%</b> `
             + `${LANG_LABEL[lang] || lang.toUpperCase()}</span>`);
  });
  if (r.elapsed) chips.push(`<span class="chip">took <b>${r.elapsed}s</b></span>`);
  const d = r.detail || {};
  if (d.device) {
    chips.push(`<span class="chip">${escapeHtml(d.device)}/${escapeHtml(d.compute_type)}</span>`);
  }
  if (d.dual_decoded_chunks) {
    chips.push(`<span class="chip"><b>${d.dual_decoded_chunks}</b> resolved both ways</span>`);
  }
  $('#stats').innerHTML = chips.join('');
}

/* -- text view -- */
function buildText(tagged) {
  const parts = [];
  let current = null, buf = [], prevEnd = null;

  for (const s of state.segments) {
    const t = s.text.trim();
    if (!t) continue;
    if (tagged && s.language !== current) {
      if (buf.length) { parts.push(buf.join(' ')); buf = []; }
      current = s.language;
      parts.push(`[${LANG_LABEL[current] || current}]`);
    } else if (prevEnd !== null && s.start - prevEnd > 1.6 && buf.length) {
      parts.push(buf.join(' '));
      buf = [];
    }
    buf.push(t);
    prevEnd = s.end;
  }
  if (buf.length) parts.push(buf.join(' '));
  return parts.join('\n\n');
}

function rebuildText() {
  $('#transcript').value = buildText($('#show-tags').checked);
  state.textDirty = false;
  updateCounter();
}

$('#show-tags').addEventListener('change', () => {
  if (state.textDirty &&
      !confirm('Rebuilding the text box will discard the edits you made in it. Continue?')) {
    $('#show-tags').checked = !$('#show-tags').checked;
    return;
  }
  rebuildText();
});

$('#transcript').addEventListener('input', () => {
  state.textDirty = true;
  updateCounter();
});

function updateCounter() {
  const v = $('#transcript').value;
  const words = v.trim() ? v.trim().split(/\s+/).length : 0;
  $('#counter').textContent = `${words.toLocaleString()} words · `
                            + `${v.length.toLocaleString()} characters`;
}

$('#copy').addEventListener('click', async () => {
  const btn = $('#copy');
  const text = $('#transcript').value;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    $('#transcript').select();
    document.execCommand('copy');
  }
  btn.textContent = 'Copied';
  setTimeout(() => { btn.textContent = 'Copy all'; }, 1400);
});

$('#select-all').addEventListener('click', () => {
  $('#transcript').focus();
  $('#transcript').select();
});

/* -- tabs -- */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');
    const view = tab.dataset.view;
    $('#view-text').hidden = view !== 'text';
    $('#view-segments').hidden = view !== 'segments';
  });
});

/* -- segments view -- */
$('#only-uncertain').addEventListener('change', renderSegments);

function isUncertain(s) {
  return s.lang_confidence < 0.75 || s.confidence < 0.45;
}

function renderSegments() {
  const onlyUncertain = $('#only-uncertain').checked;
  const list = state.segments.filter((s) => !onlyUncertain || isUncertain(s));
  const box = $('#segments');
  box.innerHTML = '';

  if (!list.length) {
    box.innerHTML = '<p class="hint">Nothing to show — no segments were flagged as uncertain.</p>';
  }

  list.forEach((seg) => {
    const row = document.createElement('div');
    row.className = 'seg' + (isUncertain(seg) ? ' uncertain' : '');
    row.dataset.id = seg.id;

    const time = document.createElement('button');
    time.className = 'seg-time';
    time.type = 'button';
    time.textContent = formatTime(seg.start);
    time.title = 'Play from here';
    time.addEventListener('click', () => {
      const p = $('#player');
      p.currentTime = seg.start;
      p.play();
    });

    const text = document.createElement('div');
    text.className = 'seg-text';
    text.contentEditable = 'true';
    text.spellcheck = false;
    text.textContent = seg.text;
    text.addEventListener('blur', () => {
      const v = text.textContent.trim();
      if (v !== seg.text) {
        seg.text = v;
        if (!state.textDirty) rebuildText();
      }
    });

    const side = document.createElement('div');
    side.className = 'seg-side';

    const lang = document.createElement('button');
    lang.className = 'langbtn';
    lang.type = 'button';
    lang.dataset.lang = seg.language;
    lang.textContent = LANG_LABEL[seg.language] || seg.language;
    lang.title = (seg.routing ? seg.routing + '\n' : '')
               + `Click to transcribe this segment as `
               + `${LANG_LABEL[OTHER[seg.language]] || ''} instead.`;
    lang.addEventListener('click', () => redecodeSegment(seg, lang));

    const conf = document.createElement('span');
    const pct = Math.round(seg.lang_confidence * 100);
    conf.className = 'conf' + (isUncertain(seg) ? ' low' : '');
    conf.textContent = `${pct}% sure`;

    side.append(lang, conf);
    row.append(time, text, side);
    box.appendChild(row);
  });

  $('#seg-counter').textContent = `${list.length} of ${state.segments.length} shown`;
}

async function redecodeSegment(seg, button) {
  const target = OTHER[seg.language];
  button.disabled = true;
  button.textContent = '…';

  const fd = new FormData();
  fd.append('job_id', state.jobId);
  fd.append('start', seg.start);
  fd.append('end', seg.end);
  fd.append('language', target);
  fd.append('beam_size', $('#beam').value);
  fd.append('hotwords', $('#hotwords').value);

  try {
    const res = await fetch('/api/redecode', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Re-transcription failed.');

    const idx = state.segments.findIndex((s) => s.id === seg.id);
    if (idx !== -1) {
      const replacements = data.segments.length
        ? data.segments
        : [{ ...seg, language: target, routing: 'manual override — no speech found' }];
      state.segments.splice(idx, 1, ...replacements);
      state.segments.forEach((s, i) => { s.id = i; });
    }
    if (!state.textDirty) rebuildText();
    renderSegments();
  } catch (err) {
    showError(err.message);
    button.disabled = false;
    button.textContent = LANG_LABEL[seg.language];
  }
}

/* ------------------------------------------------------------------ */
/* Export                                                              */
/* ------------------------------------------------------------------ */
document.querySelectorAll('.exports button').forEach((btn) => {
  btn.addEventListener('click', () => exportAs(btn.dataset.fmt));
});

async function exportAs(fmt) {
  const stem = (state.filename || 'transcript').replace(/\.[^.]+$/, '');
  const btn = document.querySelector(`.exports button[data-fmt="${fmt}"]`);

  // The text box is the authority for plain-text exports, so manual edits there
  // are never silently dropped.
  if ((fmt === 'txt' || fmt === 'tagged') && state.textDirty) {
    await save($('#transcript').value, `${stem}.txt`, 'text/plain;charset=utf-8', btn);
    return;
  }

  const fd = new FormData();
  fd.append('job_id', state.jobId);
  fd.append('fmt', fmt);
  fd.append('segments', JSON.stringify(state.segments));

  try {
    const res = await fetch('/api/export', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || 'Export failed.');
    const mime = res.headers.get('content-type') || 'text/plain;charset=utf-8';
    await save(await res.text(), `${stem}.${fmt === 'tagged' ? 'txt' : fmt}`, mime, btn);
  } catch (err) {
    showError(err.message);
  }
}

/* Every export is text, so both paths below take a string.

   In the desktop window there is nothing to download *to* — the page is not in
   a browser and has no Downloads folder of its own. It hands the text to Python
   instead, which opens the system Save dialog. In a browser the old anchor
   trick still applies. */
async function save(text, name, mime, btn) {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.save_text) {
    try {
      const path = await window.pywebview.api.save_text(name, text);
      if (path) flash(btn, 'Saved');
      return;
    } catch (err) {
      // Fall through: a browser download is better than losing the export.
      console.warn('Native save failed, falling back to a download.', err);
    }
  }
  download(new Blob([text], { type: mime }), name);
  flash(btn, 'Saved');
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* Briefly swap a button's label, the same acknowledgement "Copy all" gives. */
function flash(btn, label) {
  if (!btn || btn.dataset.flashing) return;
  const original = btn.textContent;
  btn.dataset.flashing = '1';
  btn.textContent = label;
  setTimeout(() => {
    btn.textContent = original;
    delete btn.dataset.flashing;
  }, 1400);
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
function formatTime(seconds) {
  if (!seconds && seconds !== 0) return '—';
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h
    ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
    : `${m}:${String(sec).padStart(2, '0')}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function showError(msg) {
  const el = $('#error');
  el.textContent = msg;
  el.hidden = false;
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() { $('#error').hidden = true; }

boot();
