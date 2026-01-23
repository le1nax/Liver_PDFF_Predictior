const patientSelect = document.getElementById("patientSelect");
const refreshBtn = document.getElementById("refreshBtn");
const augmentBtn = document.getElementById("augmentBtn");

const t2Canvas = document.getElementById("t2Canvas");
const ffCanvas = document.getElementById("ffCanvas");

const t2Slice = document.getElementById("t2Slice");
const ffSlice = document.getElementById("ffSlice");
const t2SliceLabel = document.getElementById("t2SliceLabel");
const ffSliceLabel = document.getElementById("ffSliceLabel");

const t2Overlay = document.getElementById("t2Overlay");
const ffOverlay = document.getElementById("ffOverlay");
const ffEroded = document.getElementById("ffEroded");

const t2Meta = document.getElementById("t2Meta");
const ffMeta = document.getElementById("ffMeta");

const overlayColor = {
  t2: [228, 108, 42],
  ff: [11, 110, 110],
};

const imageCache = new Map();
const augmentState = {
  t2: { rot: 0, rotDeg: 0, flipX: false, flipY: false },
  ff: { rot: 0, rotDeg: 0, flipX: false, flipY: false },
};

async function fetchPatients() {
  const res = await fetch("/api/patients");
  const data = await res.json();
  patientSelect.innerHTML = "";
  data.patients.forEach((pid) => {
    const option = document.createElement("option");
    option.value = pid;
    option.textContent = pid;
    patientSelect.appendChild(option);
  });
  if (data.patients.length > 0) {
    patientSelect.value = data.patients[0];
    await loadPatient();
  }
}

async function loadPatient() {
  const patientId = patientSelect.value;
  if (!patientId) return;

  const infoRes = await fetch(`/api/info/${patientId}`);
  const info = await infoRes.json();
  if (info.error) {
    t2Meta.textContent = info.error;
    ffMeta.textContent = info.error;
    return;
  }

  t2Slice.max = Math.max(info.t2.slices - 1, 0);
  ffSlice.max = Math.max(info.ff.slices - 1, 0);
  t2Slice.value = Math.floor(info.t2.slices / 2);
  ffSlice.value = Math.floor(info.ff.slices / 2);

  t2Meta.textContent = `Shape: ${info.t2.shape.join(" x ")}`;
  ffMeta.textContent = `Shape: ${info.ff.shape.join(" x ")}`;

  t2Overlay.disabled = !info.t2.has_seg;
  ffOverlay.disabled = !info.ff.has_seg;

  preloadSlices(patientId, "t2", info.t2.slices);
  preloadSlices(patientId, "ff", info.ff.slices);
  if (t2Overlay.checked && info.t2.has_seg) {
    preloadSlices(patientId, "t2_seg", info.t2.slices);
  }
  if (ffOverlay.checked && info.ff.has_seg) {
    const maskKind = ffEroded.checked ? "ff_seg_eroded" : "ff_seg";
    preloadSlices(patientId, maskKind, info.ff.slices);
  }

  await renderAll();
}

async function renderAll() {
  await Promise.all([renderView("t2"), renderView("ff")]);
}

async function renderView(kind) {
  const patientId = patientSelect.value;
  if (!patientId) return;

  const sliceIdx = kind === "t2" ? Number(t2Slice.value) : Number(ffSlice.value);
  const canvas = kind === "t2" ? t2Canvas : ffCanvas;
  const overlayEnabled = kind === "t2" ? t2Overlay.checked : ffOverlay.checked;
  const sliceLabel = kind === "t2" ? t2SliceLabel : ffSliceLabel;

  sliceLabel.textContent = `Slice ${sliceIdx}`;

  const baseImg = await loadImage(`/api/slice/${patientId}/${kind}/${sliceIdx}`);
  let maskKind = kind === "t2" ? "t2_seg" : "ff_seg";
  if (kind === "ff" && ffEroded.checked) {
    maskKind = "ff_seg_eroded";
  }
  let maskImg = null;
  if (overlayEnabled) {
    try {
      maskImg = await loadImage(`/api/slice/${patientId}/${maskKind}/${sliceIdx}`);
    } catch (err) {
      maskImg = null;
    }
  }

  drawComposite(canvas, baseImg, maskImg, overlayColor[kind], augmentState[kind]);
}

function loadImage(src) {
  if (imageCache.has(src)) {
    return imageCache.get(src);
  }
  const promise = new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${src}`));
    img.src = src;
  });
  imageCache.set(src, promise);
  return promise;
}

function drawComposite(canvas, baseImg, maskImg, color, augment) {
  const w = baseImg.width;
  const h = baseImg.height;
  const offscreen = document.createElement("canvas");
  offscreen.width = w;
  offscreen.height = h;
  const offCtx = offscreen.getContext("2d");
  offCtx.clearRect(0, 0, w, h);
  offCtx.drawImage(baseImg, 0, 0);

  if (maskImg) {
    const overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = maskImg.width;
    overlayCanvas.height = maskImg.height;
    const overlayCtx = overlayCanvas.getContext("2d");
    overlayCtx.drawImage(maskImg, 0, 0);

    const maskData = overlayCtx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
    const data = maskData.data;
    for (let i = 0; i < data.length; i += 4) {
      const value = data[i];
      if (value > 0) {
        data[i] = color[0];
        data[i + 1] = color[1];
        data[i + 2] = color[2];
        data[i + 3] = 110;
      } else {
        data[i + 3] = 0;
      }
    }
    overlayCtx.putImageData(maskData, 0, 0);
    offCtx.drawImage(overlayCanvas, 0, 0);
  }

  const ctx = canvas.getContext("2d");
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w / 2, h / 2);
  const rotQuarter = (augment?.rot || 0) * (Math.PI / 2);
  const rotFine = ((augment?.rotDeg || 0) * Math.PI) / 180;
  ctx.rotate(rotQuarter + rotFine);
  ctx.scale(augment?.flipX ? -1 : 1, augment?.flipY ? -1 : 1);
  ctx.drawImage(offscreen, -w / 2, -h / 2);
  ctx.restore();
}

function preloadSlices(patientId, kind, count) {
  for (let i = 0; i < count; i += 1) {
    const url = `/api/slice/${patientId}/${kind}/${i}`;
    loadImage(url).catch(() => null);
  }
}

function randomizeAugment() {
  ["t2", "ff"].forEach((kind) => {
    const fineAngles = Array.from({ length: 15 }, (_, i) => i + 1);
    augmentState[kind] = {
      rot: Math.floor(Math.random() * 4),
      rotDeg: (Math.random() < 0.5 ? -1 : 1) * fineAngles[Math.floor(Math.random() * fineAngles.length)],
      flipX: Math.random() < 0.5,
      flipY: Math.random() < 0.5,
    };
  });
  renderAll();
}

patientSelect.addEventListener("change", () => {
  imageCache.clear();
  loadPatient();
});
refreshBtn.addEventListener("click", fetchPatients);
t2Slice.addEventListener("input", () => renderView("t2"));
ffSlice.addEventListener("input", () => renderView("ff"));
t2Overlay.addEventListener("change", () => renderView("t2"));
ffOverlay.addEventListener("change", () => {
  renderView("ff");
  const patientId = patientSelect.value;
  if (patientId) {
    const maskKind = ffEroded.checked ? "ff_seg_eroded" : "ff_seg";
    const count = Number(ffSlice.max) + 1;
    preloadSlices(patientId, maskKind, count);
  }
});
ffEroded.addEventListener("change", () => renderView("ff"));
augmentBtn.addEventListener("click", randomizeAugment);

fetchPatients();
