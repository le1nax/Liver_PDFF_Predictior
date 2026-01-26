const patientSelect = document.getElementById("patientSelect");
const zSlider = document.getElementById("zSlider");
const opacitySlider = document.getElementById("opacitySlider");
const zValue = document.getElementById("zValue");
const opacityValue = document.getElementById("opacityValue");
const sliceGrid = document.getElementById("sliceGrid");
const status = document.getElementById("status");
const legend = document.getElementById("legend");
const prevButton = document.getElementById("prevButton");
const nextButton = document.getElementById("nextButton");
const outlineToggle = document.getElementById("outlineToggle");
const maskToggle = document.getElementById("maskToggle");
const erodedMaskToggle = document.getElementById("erodedMaskToggle");
const erosionSlider = document.getElementById("erosionSlider");
const erosionValue = document.getElementById("erosionValue");

const state = {
  allPatients: [],
  patients: [],
  depth: 0,
  width: 0,
  height: 0,
  z: 0,
  opacity: 0.55,
  stacks: new Map(),
  pageStart: 0,
  pageSize: 25,
  pageDepth: 0,
  showOutline: true,
  showMask: false,
  showErodedMask: false,
  erosionSize: 3,
  erodedCache: new Map(),
};

const CUSTOM_STOPS = [
  { t: 0.0, c: [0, 0, 139], label: "0%" },
  { t: 0.05, c: [0, 0, 139], label: "5%" },
  { t: 0.1, c: [0, 0, 255], label: "10%" },
  { t: 0.15, c: [0, 206, 209], label: "15%" },
  { t: 0.2, c: [0, 128, 0], label: "20%" },
  { t: 0.25, c: [124, 252, 0], label: "25%" },
  { t: 0.3, c: [255, 255, 0], label: "30%" },
  { t: 0.35, c: [255, 215, 0], label: "35%" },
  { t: 0.4, c: [255, 165, 0], label: "40%" },
  { t: 0.45, c: [242, 140, 40], label: "45%" },
  { t: 0.5, c: [232, 116, 42], label: "50%" },
  { t: 0.55, c: [222, 91, 42], label: "55%" },
  { t: 0.6, c: [212, 66, 39], label: "60%" },
  { t: 0.65, c: [201, 42, 34], label: "65%" },
  { t: 0.7, c: [183, 27, 27], label: "70%" },
  { t: 0.75, c: [165, 18, 20], label: "75%" },
  { t: 0.8, c: [147, 10, 13], label: "80%" },
  { t: 0.85, c: [139, 0, 0], label: "85%" },
  { t: 0.9, c: [122, 0, 0], label: "90%" },
  { t: 0.95, c: [106, 0, 0], label: "95%" },
  { t: 1.0, c: [90, 0, 0], label: "100%" },
];

const palette = buildPalette();

async function loadPatients() {
  const res = await fetch("/api/patients");
  const data = await res.json();
  patientSelect.innerHTML = "";
  data.patients.forEach((id) => {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = id;
    patientSelect.appendChild(option);
  });
  if (data.patients.length > 0) {
    state.allPatients = data.patients;
    state.pageStart = 0;
    state.patients = state.allPatients.slice(0, state.pageSize);
    patientSelect.value = state.patients[0];
    renderLegend();
    await preloadStacks();
    renderGrid();
  }
}

async function preloadStacks() {
  state.stacks.clear();
  state.erodedCache.clear();
  status.textContent = "Preloading stacks for 25 patients...";
  let minDepth = Infinity;
  let width = 0;
  let height = 0;

  renderGrid();
  const queue = state.allPatients.map((patient, index) => ({ patient, index }));
  let completed = 0;
  const concurrency = Math.min(16, queue.length || 1);

  const worker = async () => {
    while (queue.length) {
      const { patient, index } = queue.shift();
      status.textContent = `Preloading ${patient} (${completed + 1}/${state.allPatients.length})`;
      const res = await fetch(
        `/api/slices?patient=${encodeURIComponent(patient)}&start=0&count=99999`
      );
      const data = await res.json();
      width = data.width;
      height = data.height;
      minDepth = Math.min(minDepth, data.depth);

      const slices = data.slices.map((slice) => ({
        z: slice.z,
        t2: decodeBase64(slice.t2),
        ff: decodeBase64(slice.ff),
        seg: decodeBase64(slice.seg),
      }));
      state.stacks.set(patient, {
        width: data.width,
        height: data.height,
        depth: data.depth,
        slices,
      });
      completed += 1;
      if (completed % 3 === 0 || completed === state.allPatients.length) {
        status.textContent = `Preloading (${completed}/${state.allPatients.length})`;
      }
      if (state.patients.includes(patient)) {
        renderGrid();
      }
    }
  };

  await Promise.all(Array.from({ length: concurrency }, () => worker()));

  state.depth = Number.isFinite(minDepth) ? minDepth : 0;
  state.width = width;
  state.height = height;
  zSlider.value = "0";
  zValue.textContent = "0";
  status.textContent = "Stacks ready.";
}

function renderGrid() {
  const depth = updatePageDepth();
  if (!depth) {
    return;
  }
  const zIndex = Math.min(state.z, depth - 1);
  sliceGrid.innerHTML = "";
  state.patients.forEach((patient) => {
    const stack = state.stacks.get(patient);
    const tile = document.createElement("div");
    tile.className = "tile";

    const label = document.createElement("div");
    label.className = "label";
    label.textContent = patient;
    tile.appendChild(label);

    if (!stack || !stack.slices.length) {
      tile.classList.add("loading");
      sliceGrid.appendChild(tile);
      return;
    }
    const safeIndex = Math.min(zIndex, stack.slices.length - 1);
    const slice = stack.slices[safeIndex];

    const canvas = document.createElement("canvas");
    canvas.width = stack.width;
    canvas.height = stack.height;
    const ctx = canvas.getContext("2d");

    const t2 = slice.t2;
    const ff = slice.ff;
    const seg = slice.seg;
    const imageData = ctx.createImageData(stack.width, stack.height);
    const opacity = state.opacity;
    const edgeMask = state.showOutline
      ? computeSegBoundary(seg, stack.width, stack.height, 5)
      : null;
    const erodedMask = state.showErodedMask
      ? getErodedSlice(patient, stack, safeIndex, state.erosionSize)
      : null;

    for (let i = 0; i < t2.length; i++) {
      const base = t2[i];
      const [fr, fg, fb] = palette[ff[i]];
      let r = Math.round(base * (1 - opacity) + fr * opacity);
      let g = Math.round(base * (1 - opacity) + fg * opacity);
      let b = Math.round(base * (1 - opacity) + fb * opacity);
      if (state.showMask && seg[i] > 0) {
        const maskOpacity = 0.35;
        r = Math.round(r * (1 - maskOpacity) + 90 * maskOpacity);
        g = Math.round(g * (1 - maskOpacity) + 190 * maskOpacity);
        b = Math.round(b * (1 - maskOpacity) + 220 * maskOpacity);
      }
      if (erodedMask && erodedMask[i]) {
        const maskOpacity = 0.35;
        r = Math.round(r * (1 - maskOpacity) + 90 * maskOpacity);
        g = Math.round(g * (1 - maskOpacity) + 190 * maskOpacity);
        b = Math.round(b * (1 - maskOpacity) + 220 * maskOpacity);
      }
      const idx = i * 4;
      if (edgeMask && edgeMask[i]) {
        imageData.data[idx] = 255;
        imageData.data[idx + 1] = 255;
        imageData.data[idx + 2] = 255;
      } else {
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
      }
      imageData.data[idx + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);

    tile.appendChild(canvas);
    sliceGrid.appendChild(tile);
  });
  status.textContent = `Showing z slice ${zIndex} for ${state.patients.length} patients`;
}

function inferDepth() {
  for (const stack of state.stacks.values()) {
    if (stack.depth) {
      return stack.depth;
    }
  }
  return 0;
}

function updatePageDepth() {
  let maxDepth = 0;
  let found = false;
  for (const patient of state.patients) {
    const stack = state.stacks.get(patient);
    if (stack && stack.depth) {
      found = true;
      maxDepth = Math.max(maxDepth, stack.depth);
    }
  }
  if (!found) {
    const fallback = state.pageDepth || state.depth || inferDepth() || getGlobalMaxDepth();
    return fallback;
  }
  state.pageDepth = maxDepth || getGlobalMaxDepth();
  const maxIndex = Math.max(0, state.pageDepth - 1);
  if (Number(zSlider.max) !== maxIndex) {
    zSlider.max = String(maxIndex);
  }
  if (state.z > maxIndex) {
    state.z = maxIndex;
    zSlider.value = String(maxIndex);
    zValue.textContent = String(maxIndex);
  }
  return state.pageDepth;
}

function getGlobalMaxDepth() {
  let maxDepth = 0;
  for (const stack of state.stacks.values()) {
    if (stack && stack.depth) {
      maxDepth = Math.max(maxDepth, stack.depth);
    }
  }
  return maxDepth;
}

function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function buildPalette() {
  const stops = CUSTOM_STOPS;
  const palette = new Array(256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let left = stops[0];
    let right = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s].t && t <= stops[s + 1].t) {
        left = stops[s];
        right = stops[s + 1];
        break;
      }
    }
    const span = right.t - left.t || 1;
    const localT = (t - left.t) / span;
    palette[i] = [
      Math.round(left.c[0] + (right.c[0] - left.c[0]) * localT),
      Math.round(left.c[1] + (right.c[1] - left.c[1]) * localT),
      Math.round(left.c[2] + (right.c[2] - left.c[2]) * localT),
    ];
  }
  return palette;
}

function renderLegend() {
  if (!legend) {
    return;
  }
  const gradient = CUSTOM_STOPS.map(
    (stop) => `rgb(${stop.c[0]}, ${stop.c[1]}, ${stop.c[2]}) ${(stop.t * 100).toFixed(1)}%`
  ).join(", ");
  legend.innerHTML = "";

  const bar = document.createElement("div");
  bar.className = "legend-bar";
  bar.style.background = `linear-gradient(90deg, ${gradient})`;

  const labels = document.createElement("div");
  labels.className = "legend-labels";
  CUSTOM_STOPS.forEach((stop) => {
    const label = document.createElement("div");
    label.className = "legend-label";
    label.textContent = stop.label;
    labels.appendChild(label);
  });

  legend.appendChild(bar);
  legend.appendChild(labels);
}

function computeSegBoundary(seg, width, height, thickness) {
  const size = width * height;
  const mask = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    if (seg[i] > 0) {
      mask[i] = 1;
    }
  }
  let edge = mask;
  for (let step = 0; step < thickness; step++) {
    const expanded = new Uint8Array(size);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      for (let x = 0; x < width; x++) {
        const idx = row + x;
        if (edge[idx]) {
          expanded[idx] = 1;
          if (x > 0) expanded[idx - 1] = 1;
          if (x < width - 1) expanded[idx + 1] = 1;
          if (y > 0) expanded[idx - width] = 1;
          if (y < height - 1) expanded[idx + width] = 1;
        }
      }
    }
    edge = expanded;
  }

  const boundary = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    if (edge[i] && !mask[i]) {
      boundary[i] = 1;
    }
  }
  return boundary;
}

function getErodedSlice(patient, stack, zIndex, iterations) {
  const key = `${patient}|${iterations}|${zIndex}`;
  const cached = state.erodedCache.get(key);
  if (cached) {
    return cached;
  }
  const depth = stack.slices.length;
  const slice = computeErodedSlice(
    stack.slices.map((s) => s.seg),
    stack.width,
    stack.height,
    depth,
    zIndex,
    iterations
  );
  state.erodedCache.set(key, slice);
  return slice;
}

function computeErodedSlice(segSlices, width, height, depth, zIndex, iterations) {
  if (!iterations) {
    return segSlices[zIndex];
  }
  const zStart = Math.max(0, zIndex - iterations);
  const zEnd = Math.min(depth - 1, zIndex + iterations);
  const bandDepth = zEnd - zStart + 1;
  const size = width * height;

  let current = new Array(bandDepth);
  for (let z = 0; z < bandDepth; z++) {
    current[z] = new Uint8Array(segSlices[zStart + z]);
  }

  for (let iter = 0; iter < iterations; iter++) {
    const next = new Array(bandDepth);
    for (let z = 0; z < bandDepth; z++) {
      const out = new Uint8Array(size);
      const slice = current[z];
      const sliceAbove = z > 0 ? current[z - 1] : null;
      const sliceBelow = z < bandDepth - 1 ? current[z + 1] : null;
      for (let y = 0; y < height; y++) {
        const row = y * width;
        for (let x = 0; x < width; x++) {
          const idx = row + x;
          if (!slice[idx]) {
            continue;
          }
          if (x > 0 && !slice[idx - 1]) continue;
          if (x < width - 1 && !slice[idx + 1]) continue;
          if (y > 0 && !slice[idx - width]) continue;
          if (y < height - 1 && !slice[idx + width]) continue;
          if (sliceAbove && !sliceAbove[idx]) continue;
          if (sliceBelow && !sliceBelow[idx]) continue;
          out[idx] = 1;
        }
      }
      next[z] = out;
    }
    current = next;
  }
  return current[zIndex - zStart];
}

patientSelect.addEventListener("change", async (event) => {
  const selected = event.target.value;
  const startIndex = state.allPatients.indexOf(selected);
  state.pageStart = Math.max(0, startIndex);
  state.patients = state.allPatients.slice(state.pageStart, state.pageStart + state.pageSize);
  renderGrid();
});

nextButton.addEventListener("click", () => {
  if (!state.allPatients.length) {
    return;
  }
  state.pageStart = (state.pageStart + state.pageSize) % state.allPatients.length;
  state.patients = state.allPatients.slice(state.pageStart, state.pageStart + state.pageSize);
  if (state.patients.length < state.pageSize) {
    state.patients = state.patients.concat(
      state.allPatients.slice(0, state.pageSize - state.patients.length)
    );
  }
  patientSelect.value = state.patients[0];
  renderGrid();
});

prevButton.addEventListener("click", () => {
  if (!state.allPatients.length) {
    return;
  }
  let start = state.pageStart - state.pageSize;
  if (start < 0) {
    start = state.allPatients.length + start;
  }
  state.pageStart = start % state.allPatients.length;
  state.patients = state.allPatients.slice(state.pageStart, state.pageStart + state.pageSize);
  if (state.patients.length < state.pageSize) {
    state.patients = state.patients.concat(
      state.allPatients.slice(0, state.pageSize - state.patients.length)
    );
  }
  patientSelect.value = state.patients[0];
  renderGrid();
});

zSlider.addEventListener("input", async (event) => {
  const value = Number(event.target.value);
  zValue.textContent = String(value);
  state.z = value;
  renderGrid();
});

opacitySlider.addEventListener("input", (event) => {
  const value = Number(event.target.value);
  opacityValue.textContent = value.toFixed(2);
  state.opacity = value;
  renderGrid();
});

outlineToggle.addEventListener("change", (event) => {
  state.showOutline = event.target.checked;
  renderGrid();
});

maskToggle.addEventListener("change", (event) => {
  state.showMask = event.target.checked;
  renderGrid();
});

erodedMaskToggle.addEventListener("change", (event) => {
  state.showErodedMask = event.target.checked;
  renderGrid();
});

erosionSlider.addEventListener("input", (event) => {
  const value = Number(event.target.value);
  state.erosionSize = value;
  erosionValue.textContent = String(value);
  state.erodedCache.clear();
  renderGrid();
});

loadPatients().catch((error) => {
  status.textContent = `Error: ${error.message}`;
});
