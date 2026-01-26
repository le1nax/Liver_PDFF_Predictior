const patientSelect = document.getElementById("patientSelect");
const splitSelect = document.getElementById("splitSelect");
const summaryBtn = document.getElementById("summaryBtn");
const sampleBtn = document.getElementById("sampleBtn");
const binEdgesInput = document.getElementById("binEdges");
const binProbsInput = document.getElementById("binProbs");
const erosionInput = document.getElementById("erosionIters");
const summaryTable = document.getElementById("summaryTable");
const sampleMeta = document.getElementById("sampleMeta");

const t2Canvas = document.getElementById("t2Canvas");
const ffCanvas = document.getElementById("ffCanvas");
const t2Slice = document.getElementById("t2Slice");
const ffSlice = document.getElementById("ffSlice");
const t2SliceLabel = document.getElementById("t2SliceLabel");
const ffSliceLabel = document.getElementById("ffSliceLabel");
const t2Overlay = document.getElementById("t2Overlay");
const ffOverlay = document.getElementById("ffOverlay");
const ffEroded = document.getElementById("ffEroded");
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

function parseNumberList(value) {
  return value
    .split(/[,\s]+/)
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => Number(entry))
    .filter((entry) => Number.isFinite(entry));
}

function buildPayload() {
  const binEdges = parseNumberList(binEdgesInput.value);
  const binProbs = parseNumberList(binProbsInput.value);
  const erosionIters = Number(erosionInput.value);
  if (binEdges.length < 2 || binProbs.length !== binEdges.length - 1) {
    throw new Error("bin_edges must be one longer than bin_probs");
  }
  return {
    bin_edges: binEdges,
    bin_probs: binProbs,
    split: splitSelect.value,
    erosion_iters: Number.isFinite(erosionIters) ? erosionIters : 3,
  };
}

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
    sampleMeta.textContent = info.error;
    return;
  }

  t2Slice.max = Math.max(info.t2.slices - 1, 0);
  ffSlice.max = Math.max(info.ff.slices - 1, 0);
  t2Slice.value = Math.floor(info.t2.slices / 2);
  ffSlice.value = Math.floor(info.ff.slices / 2);

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

  drawComposite(canvas, baseImg, maskImg, overlayColor[kind], kind === "ff");
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

function drawComposite(canvas, baseImg, maskImg, color, useColormap = false) {
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
  ctx.drawImage(offscreen, 0, 0);
}

function preloadSlices(patientId, kind, count) {
  for (let i = 0; i < count; i += 1) {
    const url = `/api/slice/${patientId}/${kind}/${i}`;
    loadImage(url).catch(() => null);
  }
}

function renderSummary(summary) {
  summaryTable.innerHTML = "";
  if (!summary || !summary.bins) {
    summaryTable.textContent = "No summary available.";
    return;
  }

  const header = document.createElement("div");
  header.className = "table-row table-head";
  header.innerHTML = "<div>Bin</div><div>Count</div><div>Prob</div><div>Weight</div>";
  summaryTable.appendChild(header);

  summary.bins.forEach((bin) => {
    const row = document.createElement("div");
    row.className = "table-row";
    row.innerHTML = `<div>${bin.range}</div><div>${bin.count}</div><div>${bin.prob.toFixed(
      2
    )}</div><div>${bin.weight.toFixed(3)}</div>`;
    summaryTable.appendChild(row);
  });
}

async function loadSummary() {
  try {
    const payload = buildPayload();
    const res = await fetch("/api/sampling/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    renderSummary(data);
  } catch (err) {
    summaryTable.textContent = err.message;
  }
}

async function sampleNext() {
  try {
    const payload = buildPayload();
    const res = await fetch("/api/sampling/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) {
      sampleMeta.textContent = data.error;
      return;
    }
    sampleMeta.textContent = `Patient ${data.patient_id} | FF ${data.median_ff_percent.toFixed(
      2
    )}% | Bin ${data.bin_range}`;
    patientSelect.value = data.patient_id;
    imageCache.clear();
    await loadPatient();
  } catch (err) {
    sampleMeta.textContent = err.message;
  }
}

patientSelect.addEventListener("change", () => {
  imageCache.clear();
  loadPatient();
});
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
summaryBtn.addEventListener("click", loadSummary);
sampleBtn.addEventListener("click", sampleNext);

drawLegend(ffLegend);
window.addEventListener("resize", () => drawLegend(ffLegend));
fetchPatients();
