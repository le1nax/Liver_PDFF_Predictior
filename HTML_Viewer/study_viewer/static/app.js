// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let authToken = null;
let studyCases = [];
let clickableCases = [];
let currentCase = null;
let currentKind = 'mask';
let activeItem = null;

// Viewer state
let sliceIdx = 0;
let maxSlice = 0;
let brightness = 1.0;
let contrast = 1.0;
let zoom = 1.0;
let panX = 0;
let panY = 0;
let activeTool = 'wl';
let dragStart = null;
let currentImg = null;

// Preload/cache state
let studyWithDataCount = 0;
let sliceLoadToken = 0;
let studyPreloadToken = 0;
let preloadAroundTimer = null;
let preloadAroundKey = null;
let patientsFullyPreloaded = 0;

const SLICE_IMAGE_CACHE_MAX = 320;
const sliceImageCache = new Map(); // key -> { img, promise }

const STUDY_PRELOAD_CONCURRENCY = 6; // browser per-host limit is typically ~6
const STUDY_PRELOAD_THROTTLE_MS = 0;
const ADJACENT_SLICE_PRELOAD_RANGE = 8;

const preloadElsByPid = new Map(); // pid -> HTMLElement
const patientPreloadState = new Map(); // pid -> { kind, total, loaded, lastPct, done }
const patientLoadedKeys = new Map(); // pid -> Set("kind:sliceIdx")
const slicePrefetchInFlight = new Map(); // url -> Promise<void>

// Classification state: { patient_id: grade }
const classifications = {};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const authSection = document.getElementById('auth-section');
const authStatus = document.getElementById('auth-status');
const loginBtn = document.getElementById('login');
const passwordInput = document.getElementById('password');

const appSection = document.getElementById('app');
const totalEl = document.getElementById('total');
const loadedEl = document.getElementById('loaded');
const classifiedCountEl = document.getElementById('classified-count');
const classifiableCountEl = document.getElementById('classifiable-count');
const sbList = document.getElementById('sb-list');

const canvas = document.getElementById('viewer-canvas');
const ctx = canvas.getContext('2d');
const canvasWrap = document.getElementById('canvas-wrap');

const overlayTL = document.getElementById('overlay-tl');
const overlayTR = document.getElementById('overlay-tr');
const overlayBL = document.getElementById('overlay-bl');
const overlayBR = document.getElementById('overlay-br');

const sliceSlider = document.getElementById('slice-slider');
const sliceLabel = document.getElementById('slice-label');
const classifyPanel = document.getElementById('classify-panel');
const gradeRadios = document.querySelectorAll('input[name="grade"]');
const submitBtn = document.getElementById('submit-btn');
const submitStatus = document.getElementById('submit-status');

// ---------------------------------------------------------------------------
// Image cache + preload helpers
// ---------------------------------------------------------------------------
function sliceCacheKey(patientId, kind, sliceIdx) {
  return `${authToken || ''}:${patientId}:${kind}:${sliceIdx}`;
}

function sliceUrl(patientId, kind, sliceIdx) {
  return `/api/slice/${patientId}/${kind}/${sliceIdx}?token=${authToken}`;
}

function cacheGet(key) {
  const entry = sliceImageCache.get(key);
  if (!entry) return null;
  // LRU touch
  sliceImageCache.delete(key);
  sliceImageCache.set(key, entry);
  return entry;
}

function cacheSet(key, entry) {
  if (sliceImageCache.has(key)) sliceImageCache.delete(key);
  sliceImageCache.set(key, entry);
  while (sliceImageCache.size > SLICE_IMAGE_CACHE_MAX) {
    const oldestKey = sliceImageCache.keys().next().value;
    sliceImageCache.delete(oldestKey);
  }
}

function clearSliceCache() {
  sliceImageCache.clear();
}

function caseDefaultKind(caseData) {
  return caseData.kinds && caseData.kinds.length > 0 ? caseData.kinds[0] : 'mask';
}

function caseMiddleSliceIdx(caseData) {
  const n = Math.max(0, caseData.num_slices || 0);
  if (n <= 0) return 0;
  return Math.floor((n - 1) / 2);
}

function updateLoadedMeta(preloadDone = null, preloadTotal = null) {
  if (preloadDone != null && preloadTotal != null && preloadDone < preloadTotal) {
    loadedEl.textContent = `${studyWithDataCount} with images · preloading ${preloadDone}/${preloadTotal}`;
  } else {
    loadedEl.textContent = `${studyWithDataCount} with images`;
  }
}

function setRowPreloadPercent(patientId, percent) {
  const el = preloadElsByPid.get(patientId);
  if (!el) return;
  const pct = Math.max(0, Math.min(100, Math.round(percent)));
  const text = `${pct}%`;
  if (el.textContent !== text) el.textContent = text;
  el.classList.toggle('preload-done', pct >= 100);
}

function isRowPreloadDone(patientId) {
  const el = preloadElsByPid.get(patientId);
  return Boolean(el && el.classList.contains('preload-done'));
}

function initPatientPreload(caseData) {
  if (!caseData || !caseData.patient_id || (caseData.num_slices || 0) <= 0) return;
  const pid = caseData.patient_id;
  const kind = caseDefaultKind(caseData);
  const total = Math.max(1, caseData.num_slices || 0);
  patientPreloadState.set(pid, { kind, total, loaded: 0, lastPct: 0, done: false });
  patientLoadedKeys.set(pid, new Set());
}

function markSlicePreloaded(patientId, kind, sliceIdx) {
  const state = patientPreloadState.get(patientId);
  if (!state) return;
  if (state.kind !== kind) return;

  const keys = patientLoadedKeys.get(patientId);
  const k = `${kind}:${sliceIdx}`;
  if (keys && keys.has(k)) return;
  if (keys) keys.add(k);

  state.loaded += 1;
  const pct = Math.floor((state.loaded / state.total) * 100);
  if (pct !== state.lastPct) {
    state.lastPct = pct;
    setRowPreloadPercent(patientId, pct);
  }
  if (!state.done && state.loaded >= state.total) {
    state.done = true;
    setRowPreloadPercent(patientId, 100);
    patientsFullyPreloaded += 1;
    updateLoadedMeta(patientsFullyPreloaded, studyWithDataCount);
  }
}

function slicePreloadOrder(numSlices, centerIdx) {
  const n = Math.max(0, numSlices || 0);
  if (n <= 0) return [];
  const c = Math.max(0, Math.min(n - 1, centerIdx || 0));
  const order = [c];
  for (let off = 1; order.length < n; off++) {
    const left = c - off;
    const right = c + off;
    if (left >= 0) order.push(left);
    if (right < n) order.push(right);
  }
  return order;
}

function prefetchSlice(patientId, kind, sliceIdx) {
  if (!authToken) return Promise.reject(new Error('Not authenticated'));
  const url = sliceUrl(patientId, kind, sliceIdx);
  const inFlight = slicePrefetchInFlight.get(url);
  if (inFlight) return inFlight;

  const p = fetch(url, { cache: 'force-cache' }).then(async (res) => {
    if (!res.ok) throw new Error(`Prefetch failed: ${res.status}`);
    // Ensure full body is read so the browser can cache it.
    await res.arrayBuffer();
  }).finally(() => {
    slicePrefetchInFlight.delete(url);
  });

  slicePrefetchInFlight.set(url, p);
  return p;
}

function loadSliceImage(patientId, kind, sliceIdx) {
  if (!authToken) return Promise.reject(new Error('Not authenticated'));

  const key = sliceCacheKey(patientId, kind, sliceIdx);
  const cached = cacheGet(key);
  if (cached && cached.img && cached.img.complete && cached.img.naturalWidth > 0) {
    markSlicePreloaded(patientId, kind, sliceIdx);
    return Promise.resolve(cached.img);
  }
  if (cached && cached.promise) return cached.promise;

  const url = sliceUrl(patientId, kind, sliceIdx);
  const img = new window.Image();
  const promise = new Promise((resolve, reject) => {
    img.onload = () => {
      cacheSet(key, { img });
      markSlicePreloaded(patientId, kind, sliceIdx);
      resolve(img);
    };
    img.onerror = () => {
      sliceImageCache.delete(key);
      reject(new Error(`Failed to load image: ${url}`));
    };
  });
  cacheSet(key, { img, promise });
  img.src = url;
  return promise;
}

function preloadCaseMiddle(caseData) {
  if (!caseData || !caseData.patient_id || (caseData.num_slices || 0) <= 0) return;
  if (isRowPreloadDone(caseData.patient_id)) return;
  const kind = caseDefaultKind(caseData);
  const idx = caseMiddleSliceIdx(caseData);
  loadSliceImage(caseData.patient_id, kind, idx).catch(() => {});
}

function preloadSlicesAround(patientId, kind, centerIdx, maxIdx) {
  const start = Math.max(0, centerIdx - ADJACENT_SLICE_PRELOAD_RANGE);
  const end = Math.min(maxIdx, centerIdx + ADJACENT_SLICE_PRELOAD_RANGE);
  for (let i = start; i <= end; i++) {
    if (i === centerIdx) continue;
    loadSliceImage(patientId, kind, i).catch(() => {});
  }
}

function schedulePreloadAroundCurrent() {
  if (!currentCase) return;
  const pid = currentCase.patient_id;
  const kind = currentKind;
  const idx = sliceIdx;
  const key = `${pid}:${kind}:${idx}`;
  if (key === preloadAroundKey) return;
  preloadAroundKey = key;
  if (preloadAroundTimer) clearTimeout(preloadAroundTimer);
  preloadAroundTimer = setTimeout(() => {
    if (!currentCase) return;
    if (currentCase.patient_id !== pid) return;
    if (currentKind !== kind) return;
    preloadSlicesAround(pid, kind, idx, maxSlice);
  }, 80);
}

function preloadNeighborCases() {
  if (!currentCase || !clickableCases.length) return;
  const idx = clickableCases.findIndex(c => c.patient_id === currentCase.patient_id);
  if (idx < 0) return;
  const offsets = [1, -1, 2, -2];
  offsets.forEach(off => {
    const j = idx + off;
    if (j >= 0 && j < clickableCases.length) preloadCaseMiddle(clickableCases[j]);
  });
}

function startBackgroundPreload() {
  if (!clickableCases.length) return;
  const localToken = ++studyPreloadToken;
  const total = studyWithDataCount;
  let nextIndex = 0;
  updateLoadedMeta(patientsFullyPreloaded, total);

  const worker = async () => {
    while (localToken === studyPreloadToken) {
      const i = nextIndex++;
      if (i >= total) return;
      const c = clickableCases[i];
      const pid = c.patient_id;
      const state = patientPreloadState.get(pid);
      if (!state || state.done) continue;
      const kind = state.kind;
      const order = slicePreloadOrder(state.total, caseMiddleSliceIdx(c));
      for (const sliceIdx of order) {
        if (localToken !== studyPreloadToken) return;
        if (state.done) break;
        try {
          await prefetchSlice(pid, kind, sliceIdx);
          markSlicePreloaded(pid, kind, sliceIdx);
        } catch (_) {
          // ignore missing slices / auth issues
        }
        if (STUDY_PRELOAD_THROTTLE_MS > 0) {
          await new Promise(r => setTimeout(r, STUDY_PRELOAD_THROTTLE_MS));
        }
      }
    }
  };

  // Non-blocking: kick off background workers
  setTimeout(() => {
    for (let i = 0; i < STUDY_PRELOAD_CONCURRENCY; i++) worker();
  }, 0);
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
function authHeaders() {
  return { Authorization: `Bearer ${authToken}` };
}

async function login() {
  const password = passwordInput.value.trim();
  if (!password) { authStatus.textContent = 'Enter the password.'; return; }
  authStatus.textContent = 'Checking...';
  const res = await fetch('/api/auth', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password })
  });
  if (!res.ok) { authStatus.textContent = 'Invalid password.'; return; }
  const data = await res.json();
  authToken = data.token;
  clearSliceCache();
  studyPreloadToken += 1; // cancel any prior preload loops
  authSection.classList.add('hidden');
  appSection.classList.remove('hidden');
  await loadStudy();
}

loginBtn.addEventListener('click', login);
passwordInput.addEventListener('keydown', e => { if (e.key === 'Enter') login(); });

// ---------------------------------------------------------------------------
// Study loading
// ---------------------------------------------------------------------------
async function loadStudy() {
  const res = await fetch('/api/study', { headers: authHeaders() });
  if (!res.ok) return;
  const data = await res.json();
  studyCases = data.cases;
  clickableCases = data.cases.filter(c => c.num_slices > 0);
  const withData = clickableCases.length;
  studyWithDataCount = withData;
  patientsFullyPreloaded = 0;
  patientPreloadState.clear();
  patientLoadedKeys.clear();
  slicePrefetchInFlight.clear();
  totalEl.textContent = data.total_cases;
  updateLoadedMeta();
  classifiableCountEl.textContent = withData;

  sbList.innerHTML = '';
  preloadElsByPid.clear();
  data.cases.forEach(c => {
    const item = document.createElement('div');
    item.className = 'sb-item';
    item.dataset.pid = c.patient_id;
    if (c.num_slices > 0) {
      item.classList.add('sb-item-clickable');
      item.addEventListener('click', () => selectCase(c, item));
      item.addEventListener('mouseenter', () => preloadCaseMiddle(c));
    } else {
      item.classList.add('sb-item-nodata');
    }

    const top = document.createElement('div');
    top.className = 'sb-item-top';

    const pid = document.createElement('span');
    pid.className = 'sb-item-pid';
    pid.textContent = c.patient_id;

    const preload = document.createElement('span');
    preload.className = 'sb-item-preload';
    preload.textContent = c.num_slices > 0 ? '0%' : '';
    preloadElsByPid.set(c.patient_id, preload);
    initPatientPreload(c);

    const statusDot = document.createElement('span');
    statusDot.className = 'sb-item-dot';
    statusDot.title = 'Not classified';

    const right = document.createElement('div');
    right.className = 'sb-item-right';
    right.appendChild(statusDot);
    right.appendChild(preload);

    top.appendChild(pid);
    top.appendChild(right);
    item.appendChild(top);
    sbList.appendChild(item);
  });

  // Background preload all slices per case (default kind)
  startBackgroundPreload();
}

// ---------------------------------------------------------------------------
// Case selection
// ---------------------------------------------------------------------------
function selectCase(caseData, itemEl) {
  if (activeItem) activeItem.classList.remove('sb-item-active');
  activeItem = itemEl;
  itemEl.classList.add('sb-item-active');
  itemEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

  currentCase = caseData;
  currentKind = caseData.kinds && caseData.kinds.length > 0 ? caseData.kinds[0] : 'mask';
  maxSlice = Math.max(0, caseData.num_slices - 1);
  sliceIdx = Math.floor(maxSlice / 2);

  syncSlider();
  resetView();
  loadSlice();
  preloadNeighborCases();
  syncRadios();
}

function syncRadios() {
  const grade = currentCase ? classifications[currentCase.patient_id] : null;
  gradeRadios.forEach(r => { r.checked = (r.value === grade); });
}

function resetView() {
  brightness = 1.0;
  contrast = 1.0;
  zoom = 1.0;
  panX = 0;
  panY = 0;
  updateOverlays();
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------
gradeRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    if (!currentCase) return;
    classifications[currentCase.patient_id] = radio.value;
    updateSidebarDot(currentCase.patient_id);
    updateProgress();
  });
});

function updateSidebarDot(pid) {
  const item = sbList.querySelector(`[data-pid="${pid}"]`);
  if (!item) return;
  const dot = item.querySelector('.sb-item-dot');
  if (!dot) return;
  const grade = classifications[pid];
  dot.className = 'sb-item-dot';
  if (grade) {
    dot.classList.add('dot-classified', `dot-${grade}`);
    dot.title = grade;
  } else {
    dot.title = 'Not classified';
  }
}

function updateProgress() {
  const count = Object.keys(classifications).length;
  classifiedCountEl.textContent = count;
  const classifiable = clickableCases.length;
  submitBtn.disabled = count < classifiable;
  if (count >= classifiable && classifiable > 0) {
    submitBtn.classList.add('submit-ready');
  } else {
    submitBtn.classList.remove('submit-ready');
  }
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------
submitBtn.addEventListener('click', async () => {
  submitBtn.disabled = true;
  submitStatus.textContent = 'Submitting...';
  try {
    const res = await fetch('/api/submit', {
      method: 'POST',
      headers: { ...authHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ classifications })
    });
    const data = await res.json();
    if (data.ok) {
      submitStatus.textContent = `Saved: ${data.filename} (${data.correct}/${data.total} correct, ${(data.accuracy * 100).toFixed(1)}%)`;
    } else {
      submitStatus.textContent = `Error: ${data.error}`;
      submitBtn.disabled = false;
    }
  } catch (e) {
    submitStatus.textContent = `Error: ${e.message}`;
    submitBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Prev / Next patient navigation
// ---------------------------------------------------------------------------
function navigatePatient(direction) {
  if (!clickableCases.length) return;

  let idx = -1;
  if (currentCase) {
    idx = clickableCases.findIndex(c => c.patient_id === currentCase.patient_id);
  }
  idx += direction;
  if (idx < 0) idx = clickableCases.length - 1;
  if (idx >= clickableCases.length) idx = 0;

  const nextCase = clickableCases[idx];
  const itemEl = sbList.querySelector(`[data-pid="${nextCase.patient_id}"]`);
  if (itemEl) selectCase(nextCase, itemEl);
}

// ---------------------------------------------------------------------------
// Slice slider
// ---------------------------------------------------------------------------
function syncSlider() {
  sliceSlider.max = maxSlice;
  sliceSlider.value = sliceIdx;
  sliceLabel.textContent = `${sliceIdx + 1} / ${maxSlice + 1}`;
}

sliceSlider.addEventListener('input', () => {
  if (!currentCase) return;
  const val = parseInt(sliceSlider.value, 10);
  if (val !== sliceIdx) {
    sliceIdx = val;
    loadSlice();
  }
});

// ---------------------------------------------------------------------------
// Image loading & rendering
// ---------------------------------------------------------------------------
function loadSlice() {
  if (!currentCase) return;
  const pid = currentCase.patient_id;
  const kind = currentKind;
  const idx = sliceIdx;
  const token = ++sliceLoadToken;

  loadSliceImage(pid, kind, idx).then(img => {
    if (token !== sliceLoadToken) return; // stale response
    currentImg = img;
    drawCanvas();
    updateOverlays();
    syncSlider();
    schedulePreloadAroundCurrent();
  }).catch(() => {
    // ignore load errors; keep last-rendered image
  });
}

function drawCanvas() {
  if (!currentImg) return;
  const wrap = canvasWrap.getBoundingClientRect();
  canvas.width = wrap.width;
  canvas.height = wrap.height;

  ctx.save();
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.filter = `brightness(${brightness}) contrast(${contrast})`;

  const fitScale = Math.min(canvas.width / currentImg.width, canvas.height / currentImg.height);
  const scale = fitScale * zoom;
  const drawW = currentImg.width * scale;
  const drawH = currentImg.height * scale;
  const cx = (canvas.width - drawW) / 2 + panX;
  const cy = (canvas.height - drawH) / 2 + panY;

  ctx.drawImage(currentImg, cx, cy, drawW, drawH);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Overlays
// ---------------------------------------------------------------------------
function updateOverlays() {
  if (!currentCase) {
    overlayTL.textContent = '';
    overlayTR.textContent = '';
    overlayBL.textContent = '';
    overlayBR.textContent = '';
    return;
  }
  overlayTL.innerHTML = `<div>${currentCase.patient_id}</div>`;
  overlayTR.innerHTML = `<div>${currentKind.toUpperCase()}</div>`;
  overlayBL.innerHTML = `<div>Slice: ${sliceIdx + 1} / ${maxSlice + 1}</div>`;
  overlayBR.innerHTML = `<div>W/L: ${contrast.toFixed(2)} / ${brightness.toFixed(2)}</div><div>Zoom: ${(zoom * 100).toFixed(0)}%</div>`;
}

// ---------------------------------------------------------------------------
// Mouse interactions
// ---------------------------------------------------------------------------
canvasWrap.addEventListener('mousedown', e => {
  e.preventDefault();
  dragStart = { x: e.clientX, y: e.clientY, brightness, contrast, zoom, panX, panY, sliceIdx };
});

window.addEventListener('mousemove', e => {
  if (!dragStart || !currentCase) return;
  const dx = e.clientX - dragStart.x;
  const dy = e.clientY - dragStart.y;

  if (activeTool === 'wl') {
    contrast = Math.max(0.1, dragStart.contrast + dx * 0.005);
    brightness = Math.max(0.1, dragStart.brightness - dy * 0.005);
    drawCanvas();
    updateOverlays();
  } else if (activeTool === 'scroll') {
    const newSlice = Math.round(dragStart.sliceIdx - dy * 0.15);
    const clamped = Math.max(0, Math.min(maxSlice, newSlice));
    if (clamped !== sliceIdx) { sliceIdx = clamped; loadSlice(); }
  } else if (activeTool === 'zoom') {
    zoom = Math.max(0.1, Math.min(10, dragStart.zoom + dy * -0.005));
    drawCanvas();
    updateOverlays();
  } else if (activeTool === 'pan') {
    panX = dragStart.panX + dx;
    panY = dragStart.panY + dy;
    drawCanvas();
  }
});

window.addEventListener('mouseup', () => { dragStart = null; });

canvasWrap.addEventListener('wheel', e => {
  if (!currentCase) return;
  e.preventDefault();
  if (e.ctrlKey) {
    zoom = Math.max(0.1, Math.min(10, zoom + (e.deltaY > 0 ? -0.1 : 0.1)));
    drawCanvas();
    updateOverlays();
  } else {
    const dir = e.deltaY > 0 ? 1 : -1;
    const next = Math.max(0, Math.min(maxSlice, sliceIdx + dir));
    if (next !== sliceIdx) { sliceIdx = next; loadSlice(); }
  }
}, { passive: false });

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------
// Nav buttons (top-right)
document.getElementById('nav-btns').addEventListener('click', e => {
  const btn = e.target.closest('.tb-btn');
  if (!btn) return;
  if (btn.dataset.action === 'prev') navigatePatient(-1);
  if (btn.dataset.action === 'next') navigatePatient(1);
});

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const key = e.key.toLowerCase();
  if (key === 'escape') { resetView(); drawCanvas(); }
  if (key === 'arrowleft') { navigatePatient(-1); }
  if (key === 'arrowright') { navigatePatient(1); }
  if (key === 'arrowup' && currentCase) {
    e.preventDefault();
    const next = Math.max(0, sliceIdx - 1);
    if (next !== sliceIdx) { sliceIdx = next; loadSlice(); }
  }
  if (key === 'arrowdown' && currentCase) {
    e.preventDefault();
    const next = Math.min(maxSlice, sliceIdx + 1);
    if (next !== sliceIdx) { sliceIdx = next; loadSlice(); }
  }

  // Number keys 1-4 for classification
  if (currentCase && ['1','2','3','4'].includes(key)) {
    const grades = ['healthy', 'mild', 'moderate', 'severe'];
    const grade = grades[parseInt(key) - 1];
    classifications[currentCase.patient_id] = grade;
    syncRadios();
    updateSidebarDot(currentCase.patient_id);
    updateProgress();
  }
});

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------
window.addEventListener('resize', () => { if (currentImg) drawCanvas(); });
