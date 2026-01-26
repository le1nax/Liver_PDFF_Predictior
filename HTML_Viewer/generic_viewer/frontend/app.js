const patientSelect = document.getElementById("patientSelect");
const refreshBtn = document.getElementById("refreshBtn");
const augmentBtn = document.getElementById("augmentBtn");
const splitSelect = document.getElementById("splitSelect");
const sampleBtn = document.getElementById("sampleBtn");

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
const ffLegend = document.getElementById("ffLegend");

const overlayColor = {
  t2: [228, 108, 42],
  ff: [228, 108, 42],
};

const customColormapValues = [
  0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
];
const customColormapColors = [
  "#00008B", "#00008B", "#0000FF", "#00CED1", "#008000",
  "#7CFC00", "#FFFF00", "#FFD700", "#FFA500", "#F28C28",
  "#E8742A", "#DE5B2A", "#D44227", "#C92A22", "#B71B1B",
  "#A51214", "#930A0D", "#8B0000", "#7A0000", "#6A0000", "#5A0000",
];

const imageCache = new Map();
const augmentState = {
  t2: { rot: 0, rotDeg: 0, flipX: false, flipY: false },
  ff: { rot: 0, rotDeg: 0, flipX: false, flipY: false },
};

async function fetchPatients() {
  const res = await fetch(`/api/patients_with_ff?split=${splitSelect.value}`);
  const data = await res.json();
  patientSelect.innerHTML = "";
  data.patients.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.patient_id;
    const suffix =
      entry.median_ff_percent === null
        ? "n/a"
        : `${entry.median_ff_percent.toFixed(1)}%`;
    option.textContent = `${entry.patient_id} (${suffix})`;
    patientSelect.appendChild(option);
  });
  if (data.patients.length > 0) {
    patientSelect.value = data.patients[0].patient_id;
    await loadPatient();
  }
}

async function sampleFromSplit() {
  const res = await fetch(`/api/training_sampler/sample?split=${splitSelect.value}`);
  const data = await res.json();
  if (data.error) {
    t2Meta.textContent = data.error;
    ffMeta.textContent = data.error;
    return;
  }
  patientSelect.value = data.patient_id;
  imageCache.clear();
  await loadPatient();
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

  drawComposite(
    canvas,
    baseImg,
    maskImg,
    overlayColor[kind],
    augmentState[kind],
    kind === "ff"
  );
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

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    parseInt(value.slice(0, 2), 16),
    parseInt(value.slice(2, 4), 16),
    parseInt(value.slice(4, 6), 16),
  ];
}

function getCustomColor(value) {
  const v = Math.max(0, Math.min(1, value));
  for (let i = 0; i < customColormapValues.length - 1; i += 1) {
    const v0 = customColormapValues[i];
    const v1 = customColormapValues[i + 1];
    if (v >= v0 && v <= v1) {
      const t = v1 === v0 ? 0 : (v - v0) / (v1 - v0);
      const [r0, g0, b0] = hexToRgb(customColormapColors[i]);
      const [r1, g1, b1] = hexToRgb(customColormapColors[i + 1]);
      return [
        Math.round(r0 + (r1 - r0) * t),
        Math.round(g0 + (g1 - g0) * t),
        Math.round(b0 + (b1 - b0) * t),
      ];
    }
  }
  return hexToRgb(customColormapColors[customColormapColors.length - 1]);
}

function applyColormap(ctx, width, height) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const value = data[i] / 255.0;
    const [r, g, b] = getCustomColor(value);
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawLegend(canvas) {
  if (!canvas) return;
  const width = canvas.parentElement ? canvas.parentElement.clientWidth : 300;
  canvas.width = Math.max(10, width);
  canvas.height = 16;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  const data = imageData.data;
  for (let x = 0; x < canvas.width; x += 1) {
    const value = x / (canvas.width - 1);
    const [r, g, b] = getCustomColor(value);
    for (let y = 0; y < canvas.height; y += 1) {
      const idx = (y * canvas.width + x) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawComposite(canvas, baseImg, maskImg, color, augment, useColormap = false) {
  const w = baseImg.width;
  const h = baseImg.height;
  const offscreen = document.createElement("canvas");
  offscreen.width = w;
  offscreen.height = h;
  const offCtx = offscreen.getContext("2d");
  offCtx.clearRect(0, 0, w, h);
  offCtx.drawImage(baseImg, 0, 0);
  if (useColormap) {
    applyColormap(offCtx, w, h);
  }

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
splitSelect.addEventListener("change", fetchPatients);
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
sampleBtn.addEventListener("click", sampleFromSplit);

drawLegend(ffLegend);
window.addEventListener("resize", () => drawLegend(ffLegend));
fetchPatients();
