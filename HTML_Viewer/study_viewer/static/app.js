// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let authToken = null;
let studyCases = [];
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

const toolbar = document.getElementById('toolbar');
const classifyPanel = document.getElementById('classify-panel');
const gradeRadios = document.querySelectorAll('input[name="grade"]');
const submitBtn = document.getElementById('submit-btn');
const submitStatus = document.getElementById('submit-status');

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
  const withData = data.cases.filter(c => c.num_slices > 0).length;
  totalEl.textContent = data.total_cases;
  loadedEl.textContent = `${withData} with images`;
  classifiableCountEl.textContent = withData;

  sbList.innerHTML = '';
  data.cases.forEach(c => {
    const item = document.createElement('div');
    item.className = 'sb-item';
    item.dataset.pid = c.patient_id;
    if (c.num_slices > 0) {
      item.classList.add('sb-item-clickable');
      item.addEventListener('click', () => selectCase(c, item));
    } else {
      item.classList.add('sb-item-nodata');
    }

    const top = document.createElement('div');
    top.className = 'sb-item-top';

    const pid = document.createElement('span');
    pid.className = 'sb-item-pid';
    pid.textContent = c.patient_id;

    const statusDot = document.createElement('span');
    statusDot.className = 'sb-item-dot';
    statusDot.title = 'Not classified';

    top.appendChild(pid);
    top.appendChild(statusDot);
    item.appendChild(top);
    sbList.appendChild(item);
  });
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

  resetView();
  loadSlice();
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
  const classifiable = studyCases.filter(c => c.num_slices > 0).length;
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
  if (!studyCases.length) return;
  const clickable = studyCases.filter(c => c.num_slices > 0);
  if (!clickable.length) return;

  let idx = -1;
  if (currentCase) {
    idx = clickable.findIndex(c => c.patient_id === currentCase.patient_id);
  }
  idx += direction;
  if (idx < 0) idx = clickable.length - 1;
  if (idx >= clickable.length) idx = 0;

  const nextCase = clickable[idx];
  const itemEl = sbList.querySelector(`[data-pid="${nextCase.patient_id}"]`);
  if (itemEl) selectCase(nextCase, itemEl);
}

// ---------------------------------------------------------------------------
// Image loading & rendering
// ---------------------------------------------------------------------------
function loadSlice() {
  if (!currentCase) return;
  const img = new window.Image();
  img.onload = () => {
    currentImg = img;
    drawCanvas();
    updateOverlays();
  };
  img.src = `/api/slice/${currentCase.patient_id}/${currentKind}/${sliceIdx}?token=${authToken}`;
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

toolbar.addEventListener('click', e => {
  const btn = e.target.closest('.tb-btn');
  if (!btn) return;
  if (btn.dataset.action === 'reset') { resetView(); drawCanvas(); return; }
  const tool = btn.dataset.tool;
  if (tool) {
    activeTool = tool;
    toolbar.querySelectorAll('.tb-btn[data-tool]').forEach(b => b.classList.remove('tb-active'));
    btn.classList.add('tb-active');
    updateCursor();
  }
});

function updateCursor() {
  const cursors = { wl: 'ns-resize', scroll: 'row-resize', zoom: 'zoom-in', pan: 'grab' };
  canvasWrap.style.cursor = cursors[activeTool] || 'default';
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const key = e.key.toLowerCase();
  const toolMap = { w: 'wl', s: 'scroll', z: 'zoom', t: 'pan' };
  if (toolMap[key]) {
    activeTool = toolMap[key];
    toolbar.querySelectorAll('.tb-btn[data-tool]').forEach(b => b.classList.remove('tb-active'));
    const btn = toolbar.querySelector(`[data-tool="${activeTool}"]`);
    if (btn) btn.classList.add('tb-active');
    updateCursor();
  }
  if (key === 'escape') { resetView(); drawCanvas(); }
  if (key === 'arrowleft') { navigatePatient(-1); }
  if (key === 'arrowright') { navigatePatient(1); }

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
