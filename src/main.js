import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const video = document.querySelector("#webcam");
const cameraToggleButton = document.querySelector("#camera-toggle");
const canvas = document.querySelector("#overlay");
const statusEl = document.querySelector("#status");
const handCountEl = document.querySelector("#hand-count");
const pointingStateEl = document.querySelector("#pointing-state");
const confidenceEl = document.querySelector("#confidence");
const sampleCountEl = document.querySelector("#sample-count");
const fitErrorEl = document.querySelector("#fit-error");
const coverageEl = document.querySelector("#coverage");
const calibrationToggleButton = document.querySelector("#calibration-toggle");
const captureSampleButton = document.querySelector("#capture-sample");
const undoSampleButton = document.querySelector("#undo-sample");
const clearCalibrationButton = document.querySelector("#clear-calibration");
const copyDiagnosticsButton = document.querySelector("#copy-diagnostics");
const resetCalibrationButton = document.querySelector("#reset-calibration");
const ctx = canvas.getContext("2d");

const STORAGE_KEY = "hands-mode.calibration.v2";
const MIN_SAMPLES_FOR_MODEL = 24;
const FEATURE_COUNT = 22;
const isFiniteVector = (values, size) =>
  Array.isArray(values) && values.length === size && [...values].every(Number.isFinite);
const isValidSample = (sample) =>
  isFiniteVector(sample?.feature, FEATURE_COUNT) && sample.feature[0] === 1 &&
  Number.isFinite(sample.target?.x) && Number.isFinite(sample.target?.y);
const isValidModel = (model) =>
  isFiniteVector(model?.xCoefficients, FEATURE_COUNT) &&
  isFiniteVector(model.yCoefficients, FEATURE_COUNT) &&
  Number.isFinite(Math.hypot(...model.xCoefficients)) &&
  Number.isFinite(Math.hypot(...model.yCoefficients)) &&
  Number.isInteger(model.sampleCount) && model.sampleCount >= MIN_SAMPLES_FOR_MODEL &&
  [model.meanError, model.maxError].every((value) => Number.isFinite(value) && value >= 0) &&
  [model.medianError, model.xNorm, model.yNorm, model.coverageCount]
    .every((value) => value === undefined || (Number.isFinite(value) && value >= 0)) &&
  (model.samples === undefined || (Array.isArray(model.samples) &&
    model.samples.length === model.sampleCount && model.samples.every(isValidSample)));
const isValidLandmark = (point) =>
  Number.isFinite(point?.x) && Number.isFinite(point.y) &&
  Number.isFinite(point.z === undefined ? 0 : point.z);
const macbookScreen = {
  aspectWidth: 16,
  aspectHeight: 10,
};

let handLandmarker;
let lastVideoTime = -1;
let smoothedPointer = null;
let latestObservation = null;
let pendingSamples = [];
let calibrationModel = null;
let isCalibrating = false;
let trackingState = "stopped";
let cameraSession = 0;
let animationFrameId = null;

const calibrationTargets = [
  [0.08, 0.08],
  [0.27, 0.08],
  [0.5, 0.08],
  [0.73, 0.08],
  [0.92, 0.08],
  [0.12, 0.28],
  [0.34, 0.28],
  [0.5, 0.28],
  [0.66, 0.28],
  [0.88, 0.28],
  [0.08, 0.5],
  [0.27, 0.5],
  [0.5, 0.5],
  [0.73, 0.5],
  [0.92, 0.5],
  [0.12, 0.72],
  [0.34, 0.72],
  [0.5, 0.72],
  [0.66, 0.72],
  [0.88, 0.72],
  [0.08, 0.92],
  [0.27, 0.92],
  [0.5, 0.92],
  [0.73, 0.92],
  [0.92, 0.92],
].map(([x, y]) => ({ x, y }));

const setStatus = (message) => {
  statusEl.textContent = message;
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

const resizeCanvas = () => {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;

  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height };
};

const getScreenRect = (width, height) => {
  const screenAspect = macbookScreen.aspectWidth / macbookScreen.aspectHeight;
  const viewportAspect = width / height;

  if (viewportAspect > screenAspect) {
    const rectHeight = height;
    const rectWidth = rectHeight * screenAspect;
    return {
      x: (width - rectWidth) / 2,
      y: 0,
      width: rectWidth,
      height: rectHeight,
    };
  }

  const rectWidth = width;
  const rectHeight = rectWidth / screenAspect;
  return {
    x: 0,
    y: (height - rectHeight) / 2,
    width: rectWidth,
    height: rectHeight,
  };
};

const mapToScreen = (point, screenRect) => ({
  x: screenRect.x + point.x * screenRect.width,
  y: screenRect.y + point.y * screenRect.height,
});

const mirrorPoint = (landmark) => ({
  x: 1 - landmark.x,
  y: landmark.y,
  z: landmark.z ?? 0,
});

const mirrorWorldPoint = (landmark) => ({
  x: -(landmark?.x ?? 0),
  y: landmark?.y ?? 0,
  z: landmark?.z ?? 0,
});

const smoothPointer = (pointer) => {
  if (!smoothedPointer) {
    smoothedPointer = pointer;
    return pointer;
  }

  const smoothing = 0.3;
  smoothedPointer = {
    ...pointer,
    x: smoothedPointer.x + (pointer.x - smoothedPointer.x) * smoothing,
    y: smoothedPointer.y + (pointer.y - smoothedPointer.y) * smoothing,
  };

  return smoothedPointer;
};

const getFingerCurl = (hand) => {
  const wrist = mirrorPoint(hand[0]);
  const indexTip = mirrorPoint(hand[8]);
  const tips = [mirrorPoint(hand[12]), mirrorPoint(hand[16]), mirrorPoint(hand[20])];
  const indexLength = distance(wrist, indexTip) || 1;
  const averageOtherLength =
    tips.reduce((sum, tip) => sum + distance(wrist, tip), 0) / tips.length;

  return clamp((indexLength - averageOtherLength) / indexLength, 0, 1);
};

const estimateRawTarget = (hand) => {
  if (![0, 6, 7, 8, 12, 16, 20].every((index) => isValidLandmark(hand?.[index]))) {
    return null;
  }
  const wrist = mirrorPoint(hand[0]);
  const indexPip = mirrorPoint(hand[6]);
  const indexDip = mirrorPoint(hand[7]);
  const indexTip = mirrorPoint(hand[8]);
  const segmentDx = indexTip.x - indexDip.x;
  const segmentDy = indexTip.y - indexDip.y;
  const segmentLength = Math.hypot(segmentDx, segmentDy);

  if (segmentLength < 0.006) {
    return null;
  }

  const dirX = segmentDx / segmentLength;
  const dirY = segmentDy / segmentLength;
  const fingerLength = distance(indexPip, indexTip);
  const wristToTip = distance(wrist, indexTip);
  const curl = getFingerCurl(hand);
  const distanceBoost = clamp(0.18 / segmentLength, 2.8, 8.5);
  const extension = segmentLength * distanceBoost + fingerLength * 1.6 + wristToTip * 0.25;
  const rayEnd = {
    x: indexTip.x + dirX * extension,
    y: indexTip.y + dirY * extension,
  };

  if (![segmentLength, curl, rayEnd.x, rayEnd.y].every(Number.isFinite)) return null;

  return {
    x: clamp(rayEnd.x, 0, 1),
    y: clamp(rayEnd.y, 0, 1),
    confidence: clamp((segmentLength - 0.006) / 0.035, 0, 1) * (0.55 + curl * 0.45),
    ray: {
      start: indexTip,
      end: rayEnd,
    },
    metrics: {
      dirX,
      dirY,
      segmentLength,
      fingerLength,
      wristToTip,
      curl,
    },
  };
};

const buildFeature = (hand, worldHand, raw) => {
  const wrist = mirrorPoint(hand[0]);
  const indexPip = mirrorPoint(hand[6]);
  const indexDip = mirrorPoint(hand[7]);
  const indexTip = mirrorPoint(hand[8]);
  const worldTip = mirrorWorldPoint(worldHand?.[8]);
  const worldDip = mirrorWorldPoint(worldHand?.[7]);
  const worldDir = {
    x: worldTip.x - worldDip.x,
    y: worldTip.y - worldDip.y,
    z: worldTip.z - worldDip.z,
  };

  return [
    1,
    raw.x,
    raw.y,
    raw.metrics.dirX,
    raw.metrics.dirY,
    raw.metrics.segmentLength,
    raw.metrics.fingerLength,
    raw.metrics.wristToTip,
    raw.metrics.curl,
    indexTip.x,
    indexTip.y,
    indexDip.x,
    indexDip.y,
    indexPip.x,
    indexPip.y,
    wrist.x,
    wrist.y,
    indexTip.z,
    indexDip.z,
    worldDir.x,
    worldDir.y,
    worldDir.z,
  ];
};

const observeHand = (results) => {
  const hand = results.landmarks?.[0];

  if (!hand) {
    return null;
  }

  const raw = estimateRawTarget(hand);

  if (!raw) {
    return null;
  }

  const worldHand = results.worldLandmarks?.[0];
  if (worldHand && ![7, 8].every((index) => isValidLandmark(worldHand[index]))) return null;
  const feature = buildFeature(hand, worldHand, raw);
  return isFiniteVector(feature, FEATURE_COUNT) ? { raw, feature } : null;
};

const solveLinearSystem = (matrix, values) => {
  const size = values.length;
  if (!size || matrix.length !== size || !matrix.every((row) => isFiniteVector(row, size)) ||
      !values.every((row) => isFiniteVector(row, 2))) {
    throw new Error("Calibration matrix contains invalid values.");
  }
  const augmented = matrix.map((row, index) => [...row, ...values[index]]);
  const columnCount = augmented[0].length;

  for (let column = 0; column < size; column += 1) {
    let pivotRow = column;

    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivotRow][column])) {
        pivotRow = row;
      }
    }

    if (!Number.isFinite(augmented[pivotRow][column]) || Math.abs(augmented[pivotRow][column]) < 1e-10) {
      throw new Error("Calibration matrix is unstable.");
    }

    [augmented[column], augmented[pivotRow]] = [augmented[pivotRow], augmented[column]];

    const pivot = augmented[column][column];
    for (let item = column; item < columnCount; item += 1) {
      augmented[column][item] /= pivot;
    }

    for (let row = 0; row < size; row += 1) {
      if (row === column) {
        continue;
      }

      const factor = augmented[row][column];
      for (let item = column; item < columnCount; item += 1) {
        augmented[row][item] -= factor * augmented[column][item];
      }
    }
  }

  const solution = augmented.map((row) => row.slice(size));
  if (!solution.every((row) => isFiniteVector(row, 2))) {
    throw new Error("Calibration solution is not finite.");
  }
  return solution;
};

const fitCoordinates = (samples, weights) => {
  const featureCount = samples[0].feature.length;
  const matrix = Array.from({ length: featureCount }, () => Array(featureCount).fill(0));
  const values = Array.from({ length: featureCount }, () => [0, 0]);

  for (const [sampleIndex, sample] of samples.entries()) {
    const weight = weights?.[sampleIndex] ?? 1;

    for (let row = 0; row < featureCount; row += 1) {
      values[row][0] += sample.feature[row] * sample.target.x * weight;
      values[row][1] += sample.feature[row] * sample.target.y * weight;

      for (let column = 0; column < featureCount; column += 1) {
        matrix[row][column] += sample.feature[row] * sample.feature[column] * weight;
      }
    }
  }

  for (let index = 0; index < featureCount; index += 1) {
    matrix[index][index] += 0.01;
  }

  // Both coordinates share the same weighted matrix and pivot operations.
  const coefficients = solveLinearSystem(matrix, values);
  return {
    xCoefficients: coefficients.map(([x]) => x),
    yCoefficients: coefficients.map(([, y]) => y),
  };
};

const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
};

const getCoverageCount = (samples) => {
  const coveredTargets = new Set(
    samples.map((sample) => {
      const x = Math.round(sample.target.x * 100);
      const y = Math.round(sample.target.y * 100);
      return `${x}:${y}`;
    }),
  );

  return coveredTargets.size;
};

const coefficientNorm = (coefficients) => Math.hypot(...coefficients);

const predictWithModel = (feature, model) => ({
  x: dot(feature, model.xCoefficients),
  y: dot(feature, model.yCoefficients),
});

const fitModelFromSamples = (samples) => {
  if (!Array.isArray(samples) || samples.length < MIN_SAMPLES_FOR_MODEL || !samples.every(isValidSample)) {
    return null;
  }

  let weights = samples.map(() => 1);
  let model = null;
  let errors;

  for (let pass = 0; pass < 4; pass += 1) {
    try {
      model = fitCoordinates(samples, weights);
    } catch {
      return null;
    }

    errors = samples.map((sample) => {
      const prediction = predictWithModel(sample.feature, model);
      return Math.hypot(prediction.x - sample.target.x, prediction.y - sample.target.y);
    });
    if (!errors.every(Number.isFinite)) return null;
    if (pass < 3) {
      const typicalError = Math.max(median(errors), 0.025);
      weights = errors.map((error) => 1 / (1 + (error / (typicalError * 2)) ** 2));
    }
  }

  const fitted = {
    ...model,
    sampleCount: samples.length,
    meanError: errors.reduce((sum, error) => sum + error, 0) / errors.length,
    maxError: Math.max(...errors),
    medianError: median(errors),
    coverageCount: getCoverageCount(samples),
    xNorm: coefficientNorm(model.xCoefficients),
    yNorm: coefficientNorm(model.yCoefficients),
  };
  return isValidModel(fitted) ? fitted : null;
};

const predictCalibratedTarget = (observation) => {
  if (!calibrationModel) {
    return observation.raw;
  }

  const calibrated = predictWithModel(observation.feature, calibrationModel);
  if (!Number.isFinite(calibrated.x) || !Number.isFinite(calibrated.y)) return null;

  return {
    ...observation.raw,
    x: clamp(calibrated.x, 0, 1),
    y: clamp(calibrated.y, 0, 1),
  };
};

const saveCalibration = () => {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        pendingSamples,
        model: calibrationModel,
        savedAt: Date.now(),
      }),
    );
    return true;
  } catch {
    return false;
  }
};

const loadCalibration = () => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    const storedPending = Array.isArray(saved.pendingSamples)
      ? saved.pendingSamples : Array.isArray(saved.samples) ? saved.samples : [];
    calibrationModel = isValidModel(saved.model) ? saved.model : null;
    const retainedSamples = !calibrationModel && Array.isArray(saved.model?.samples)
      ? saved.model.samples : [];
    const candidates = [...retainedSamples, ...storedPending];
    pendingSamples = candidates.filter(isValidSample);
    let notice = saved.model && !calibrationModel
      ? "Invalid saved calibration model was rejected. " : "";
    if (pendingSamples.length !== candidates.length) {
      notice += "Invalid calibration samples were not loaded. ";
    }
    if (!calibrationModel && pendingSamples.length >= MIN_SAMPLES_FOR_MODEL) {
      const fitted = fitModelFromSamples(pendingSamples);
      if (fitted) {
        calibrationModel = { ...fitted, samples: pendingSamples };
        pendingSamples = [];
        if (saved.model) notice += "A model was rebuilt from retained valid samples. ";
      } else {
        notice += "Saved samples could not produce a finite model. ";
      }
    }
    return notice;
  } catch {
    pendingSamples = [];
    calibrationModel = null;
    return "Saved calibration could not be read and was not loaded. ";
  }
};

const hasCalibrationData = () => Boolean(calibrationModel || pendingSamples.length);

const getAcceptedSampleCount = () => calibrationModel?.sampleCount ?? 0;

const getCurrentTarget = () =>
  calibrationTargets[(getAcceptedSampleCount() + pendingSamples.length) % calibrationTargets.length];

const updateCalibrationUi = () => {
  cameraToggleButton.textContent = trackingState === "tracking"
    ? "Stop Camera"
    : trackingState === "loading" || trackingState === "requesting-camera"
      ? "Cancel Camera Start"
      : "Start Camera";
  if (trackingState === "tracking" && latestObservation) {
    pointingStateEl.textContent = isCalibrating
      ? "Calibrating"
      : calibrationModel
        ? "Calibrated"
        : "Geometry";
  }
  sampleCountEl.textContent = calibrationModel
    ? pendingSamples.length
      ? `${calibrationModel.sampleCount}+${pendingSamples.length}`
      : String(calibrationModel.sampleCount)
    : String(pendingSamples.length);
  fitErrorEl.textContent = calibrationModel
    ? `${Math.round(calibrationModel.meanError * 100)}%/${Math.round(calibrationModel.maxError * 100)}%`
    : "n/a";
  coverageEl.textContent = calibrationModel
    ? `${calibrationModel.coverageCount ?? getCoverageCount(calibrationModel.samples ?? [])}/25`
    : `${getCoverageCount(pendingSamples)}/25`;
  calibrationToggleButton.textContent = isCalibrating
    ? "Pause Calibration"
    : hasCalibrationData()
      ? "Resume Calibration"
      : "Start Calibration";
  calibrationToggleButton.disabled = trackingState !== "tracking";
  captureSampleButton.disabled =
    trackingState !== "tracking" || !isCalibrating || !latestObservation;
  undoSampleButton.disabled = pendingSamples.length === 0;
  clearCalibrationButton.disabled = pendingSamples.length === 0;
  resetCalibrationButton.disabled = !hasCalibrationData();
};

const getDiagnostics = () => ({
  hasModel: Boolean(calibrationModel),
  acceptedSamples: calibrationModel?.sampleCount ?? 0,
  pendingSamples: pendingSamples.length,
  coverage: calibrationModel?.coverageCount ?? getCoverageCount(pendingSamples),
  meanError: calibrationModel?.meanError ?? null,
  medianError: calibrationModel?.medianError ?? null,
  maxError: calibrationModel?.maxError ?? null,
  xCoefficientNorm: calibrationModel?.xNorm ?? null,
  yCoefficientNorm: calibrationModel?.yNorm ?? null,
  nextTarget: getCurrentTarget(),
});

const copyDiagnostics = async () => {
  const diagnostics = getDiagnostics();
  const text = JSON.stringify(diagnostics, null, 2);

  try {
    await navigator.clipboard.writeText(text);
    setStatus("Copied model diagnostics to the clipboard.");
  } catch {
    console.log(diagnostics);
    setStatus("Could not copy diagnostics, so I logged them to the browser console.");
  }
};

const toggleCalibration = () => {
  isCalibrating = !isCalibrating;
  updateCalibrationUi();
  if (isCalibrating) {
    canvas.focus({ preventScroll: true });
  }
  setStatus(
    isCalibrating
      ? "Point at the pink target and press Space or Capture Sample. You can pause anytime."
      : "Calibration paused. Samples are saved locally.",
  );
};

const captureSample = () => {
  if (!isCalibrating) {
    return;
  }

  if (!latestObservation) {
    setStatus("I do not see a clear pointing hand yet.");
    return;
  }

  const sample = {
    feature: [...latestObservation.feature],
    target: getCurrentTarget(),
    createdAt: Date.now(),
  };
  const candidateSamples = [...(calibrationModel?.samples ?? []), ...pendingSamples, sample];
  const nextModel = isValidSample(sample) ? fitModelFromSamples(candidateSamples) : null;
  if (!isValidSample(sample) || (candidateSamples.length >= MIN_SAMPLES_FOR_MODEL && !nextModel)) {
    setStatus("Capture rejected: samples could not produce a finite calibration. The active model and saved data were kept.");
    return;
  }

  if (nextModel) {
    calibrationModel = { ...nextModel, samples: candidateSamples };
    pendingSamples = [];
    smoothedPointer = null;
  } else {
    pendingSamples.push(sample);
  }

  const saved = saveCalibration();
  updateCalibrationUi();
  setStatus(
    calibrationModel
      ? `Captured sample ${calibrationModel.sampleCount}. Model refit.${
          saved
            ? " Calibration saved locally."
            : " Browser storage is unavailable, so this session was not saved."
        }`
      : `Captured ${pendingSamples.length}. Need ${MIN_SAMPLES_FOR_MODEL} before prediction switches to calibrated.${
          saved ? "" : " Browser storage is unavailable, so this session was not saved."
        }`,
  );
};

const undoSample = () => {
  pendingSamples.pop();
  const saved = saveCalibration();
  updateCalibrationUi();
  setStatus(
    `Removed the last pending capture. The active model was unchanged.${
      saved ? "" : " Browser storage is unavailable, so this change was not saved."
    }`,
  );
};

const clearPendingSamples = () => {
  pendingSamples = [];
  const saved = saveCalibration();
  updateCalibrationUi();
  setStatus(
    `Pending captures cleared. The active model was kept.${
      saved ? "" : " Browser storage is unavailable, so this change was not saved."
    }`,
  );
};

const resetCalibration = () => {
  const confirmed = window.confirm(
    "Reset the active model and delete all stored calibration samples? This cannot be undone.",
  );

  if (!confirmed) {
    return;
  }

  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    setStatus("Could not remove calibration data because browser storage is unavailable.");
    return;
  }

  pendingSamples = [];
  calibrationModel = null;
  smoothedPointer = null;
  isCalibrating = trackingState === "tracking";
  updateCalibrationUi();
  if (isCalibrating) {
    canvas.focus({ preventScroll: true });
  }
  setStatus(
    isCalibrating
      ? "Calibration reset. Point at the pink target and press Space or Capture Sample."
      : "Calibration reset. Start calibration when the webcam is ready.",
  );
};

const drawScreenFrame = (screenRect) => {
  ctx.save();
  ctx.fillStyle = "#050608";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
  ctx.lineWidth = 2;
  ctx.strokeRect(screenRect.x, screenRect.y, screenRect.width, screenRect.height);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.055)";
  ctx.lineWidth = 1;

  for (let x = 1; x < 4; x += 1) {
    const lineX = screenRect.x + (screenRect.width * x) / 4;
    ctx.beginPath();
    ctx.moveTo(lineX, screenRect.y);
    ctx.lineTo(lineX, screenRect.y + screenRect.height);
    ctx.stroke();
  }

  for (let y = 1; y < 4; y += 1) {
    const lineY = screenRect.y + (screenRect.height * y) / 4;
    ctx.beginPath();
    ctx.moveTo(screenRect.x, lineY);
    ctx.lineTo(screenRect.x + screenRect.width, lineY);
    ctx.stroke();
  }

  ctx.restore();
};

const drawTarget = (target, screenRect) => {
  const point = mapToScreen(target, screenRect);

  ctx.save();
  ctx.strokeStyle = "#ff5c7a";
  ctx.fillStyle = "rgba(255, 92, 122, 0.16)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 22, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(point.x - 34, point.y);
  ctx.lineTo(point.x + 34, point.y);
  ctx.moveTo(point.x, point.y - 34);
  ctx.lineTo(point.x, point.y + 34);
  ctx.stroke();
  ctx.restore();
};

const drawPointer = (prediction, screenRect) => {
  const start = mapToScreen(prediction.ray.start, screenRect);
  const end = mapToScreen(
    {
      x: clamp(prediction.ray.end.x, -0.5, 1.5),
      y: clamp(prediction.ray.end.y, -0.5, 1.5),
    },
    screenRect,
  );
  const point = mapToScreen(prediction, screenRect);

  ctx.save();
  ctx.strokeStyle = "rgba(255, 218, 106, 0.68)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(95, 255, 154, 0.2)";
  ctx.strokeStyle = "#5fff9a";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 18, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(point.x - 28, point.y);
  ctx.lineTo(point.x + 28, point.y);
  ctx.moveTo(point.x, point.y - 28);
  ctx.lineTo(point.x, point.y + 28);
  ctx.stroke();
  ctx.restore();
};

const clearPointer = (mode) => {
  latestObservation = null;
  smoothedPointer = null;
  pointingStateEl.textContent = mode;
  confidenceEl.textContent = "0%";
};

const drawTracking = (results) => {
  const { width, height } = resizeCanvas();
  const screenRect = getScreenRect(width, height);
  drawScreenFrame(screenRect);

  const hands = results.landmarks ?? [];
  handCountEl.textContent = String(hands.length);
  const previouslyObservedHand = Boolean(latestObservation);
  latestObservation = observeHand(results);

  if (isCalibrating) {
    drawTarget(getCurrentTarget(), screenRect);
  }

  if (!latestObservation) {
    clearPointer(hands.length ? "Unclear" : "No hand");
    if (previouslyObservedHand) {
      updateCalibrationUi();
    }
    return;
  }

  const prediction = predictCalibratedTarget(latestObservation);
  if (!prediction) {
    clearPointer("Invalid prediction");
    updateCalibrationUi();
    setStatus("Pointing estimate was invalid; waiting for a valid hand observation.");
    return;
  }
  const pointer = smoothPointer(prediction);
  confidenceEl.textContent = `${Math.round(pointer.confidence * 100)}%`;
  if (!previouslyObservedHand) {
    updateCalibrationUi();
  }
  drawPointer(pointer, screenRect);
};

const stopTracks = (stream) => {
  for (const track of stream?.getTracks() ?? []) {
    if (track.readyState !== "ended") track.stop();
  }
};

const endCameraSession = (state, mode, message) => {
  // Retire pending starts and queued frames before releasing their resources.
  cameraSession += 1;
  trackingState = state;
  cancelAnimationFrame(animationFrameId);
  animationFrameId = null;
  const stream = video.srcObject;
  video.srcObject = null;
  video.pause();
  stopTracks(stream);
  const detector = handLandmarker;
  handLandmarker = null;
  try {
    detector?.close();
  } catch (error) {
    console.error(error);
  }
  lastVideoTime = -1;
  isCalibrating = false;
  clearPointer(mode);
  handCountEl.textContent = "0";
  updateCalibrationUi();
  setStatus(message);
  try {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  } catch (error) {
    console.error(error);
  }
};

const stopCamera = () => {
  const pending = trackingState === "loading" || trackingState === "requesting-camera";
  endCameraSession("stopped", "Camera stopped", pending
    ? "Camera start cancelled. Any pending camera request will be released if it completes. Calibration kept."
    : "Camera stopped. This app has released its camera stream. Calibration kept.");
};

const predict = (session = cameraSession) => {
  if (session !== cameraSession || trackingState !== "tracking") return;
  try {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const results = handLandmarker.detectForVideo(video, performance.now());
      if (session !== cameraSession) return;
      drawTracking(results);
    }
  } catch (error) {
    console.error(error);
    if (session !== cameraSession) return;
    endCameraSession("error", "Error", `Hand tracking stopped: ${error.message}. Select Start Camera to try again.`);
    return;
  }

  animationFrameId = requestAnimationFrame(() => predict(session));
};

const startCamera = async (session) => {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });

  if (session !== cameraSession) {
    stopTracks(stream);
    return false;
  }
  video.srcObject = stream;
  for (const track of stream.getTracks()) {
    const stopForLostInput = (event) => {
      if (session === cameraSession) {
        const reason = event.type === "mute" ? "Camera input interrupted." : "Camera stream ended.";
        endCameraSession("stopped", "Camera stopped", `${reason} Select Start Camera to reconnect. Calibration kept.`);
      }
    };
    track.addEventListener("ended", stopForLostInput, { once: true });
    track.addEventListener("mute", stopForLostInput, { once: true });
  }
  await video.play();
  if (session !== cameraSession) {
    stopTracks(stream);
    return false;
  }
  return true;
};

const createHandLandmarker = async (filesetResolver, delegate) =>
  HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate,
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });

const describeCameraError = (error) => {
  if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
    return {
      mode: "Camera denied",
      message:
        "Webcam access was denied. Allow camera access for this site in your browser settings, then select Start Camera.",
    };
  }

  if (error?.name === "NotFoundError" || error?.name === "DevicesNotFoundError") {
    return {
      mode: "No camera",
      message: "No webcam was found. Connect or enable a webcam, then select Start Camera.",
    };
  }

  if (error?.name === "NotReadableError" || error?.name === "TrackStartError") {
    return {
      mode: "Camera busy",
      message:
        "The webcam could not be opened. Close other apps using it, check system camera access, then select Start Camera.",
    };
  }

  return {
    mode: "Error",
    message: `Could not start the webcam: ${error?.message || "unknown error"}. Check browser camera access, then select Start Camera.`,
  };
};

const startTracking = async (calibrationNotice = "") => {
  if (trackingState !== "stopped" && trackingState !== "error") return;
  const session = ++cameraSession;

  if (!navigator.mediaDevices?.getUserMedia) {
    trackingState = "error";
    pointingStateEl.textContent = "Unavailable";
    updateCalibrationUi();
    setStatus(
      "Webcam access is unavailable here. Use a current desktop browser and open the app from localhost or HTTPS.",
    );
    return;
  }

  try {
    trackingState = "loading";
    pointingStateEl.textContent = "Loading";
    updateCalibrationUi();
    setStatus(
      "Loading the hand tracker. Webcam access is needed; frames are processed locally in this browser.",
    );

    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm",
    );
    if (session !== cameraSession) return;

    let detector;
    try {
      detector = await createHandLandmarker(filesetResolver, "GPU");
    } catch {
      if (session !== cameraSession) return;
      detector = await createHandLandmarker(filesetResolver, "CPU");
    }
    if (session !== cameraSession) {
      detector.close();
      return;
    }
    handLandmarker = detector;
  } catch (error) {
    if (session !== cameraSession) return;
    console.error(error);
    endCameraSession("error", "Load error", `Could not load the MediaPipe hand tracker: ${error.message}. Check your internet connection, then select Start Camera.`);
    return;
  }

  trackingState = "requesting-camera";
  pointingStateEl.textContent = "Awaiting camera";
  updateCalibrationUi();
  setStatus(
    "Webcam access is required to locate your hand. When the browser asks, choose Allow. Frames are processed locally.",
  );

  try {
    if (!await startCamera(session)) return;
  } catch (error) {
    if (session !== cameraSession) return;
    console.error(error);
    const cameraError = describeCameraError(error);
    endCameraSession("error", cameraError.mode, cameraError.message);
    return;
  }

  trackingState = "tracking";
  pointingStateEl.textContent = "No hand";
  updateCalibrationUi();
  setStatus(calibrationNotice + (
    calibrationModel
      ? `Webcam ready. Loaded a calibrated model with ${calibrationModel.sampleCount} samples. Show one pointing hand.`
      : pendingSamples.length
        ? `Webcam ready. Loaded ${pendingSamples.length} pending captures. Resume calibration when ready.`
        : "Webcam ready. Show one pointing hand, or start calibration to adapt the estimate to this setup."
  ));
  predict(session);
};

const init = () => {
  return startTracking(loadCalibration());
};

cameraToggleButton.addEventListener("click", () => {
  if (trackingState === "stopped" || trackingState === "error") {
    startTracking();
  } else {
    stopCamera();
  }
});
window.addEventListener("pagehide", stopCamera);
window.addEventListener("resize", resizeCanvas);
window.addEventListener("keydown", (event) => {
  if (
    event.code !== "Space" ||
    !isCalibrating ||
    event.defaultPrevented ||
    event.altKey || event.ctrlKey || event.metaKey || event.shiftKey ||
    event.target?.closest?.("button, input, select, textarea, a[href], [contenteditable]")
  ) {
    return;
  }

  event.preventDefault();
  if (!event.repeat) {
    captureSample();
  }
});
calibrationToggleButton.addEventListener("click", toggleCalibration);
captureSampleButton.addEventListener("click", captureSample);
undoSampleButton.addEventListener("click", undoSample);
clearCalibrationButton.addEventListener("click", clearPendingSamples);
copyDiagnosticsButton.addEventListener("click", copyDiagnostics);
resetCalibrationButton.addEventListener("click", resetCalibration);
init();
