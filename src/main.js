import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const video = document.querySelector("#webcam");
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
let trackingState = "loading";

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

  return {
    raw,
    feature: buildFeature(hand, results.worldLandmarks?.[0], raw),
  };
};

const solveLinearSystem = (matrix, values) => {
  const size = values.length;
  const augmented = matrix.map((row, index) => [...row, ...values[index]]);
  const columnCount = augmented[0].length;

  for (let column = 0; column < size; column += 1) {
    let pivotRow = column;

    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivotRow][column])) {
        pivotRow = row;
      }
    }

    if (Math.abs(augmented[pivotRow][column]) < 1e-10) {
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

  return augmented.map((row) => row.slice(size));
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

const coefficientNorm = (coefficients) =>
  Math.hypot(...coefficients.filter((value) => Number.isFinite(value)));

const predictWithModel = (feature, model) => ({
  x: dot(feature, model.xCoefficients),
  y: dot(feature, model.yCoefficients),
});

const fitModelFromSamples = (samples) => {
  if (samples.length < MIN_SAMPLES_FOR_MODEL) {
    return null;
  }

  let weights = samples.map(() => 1);
  let model = null;

  for (let pass = 0; pass < 4; pass += 1) {
    model = fitCoordinates(samples, weights);

    const errors = samples.map((sample) => {
      const prediction = predictWithModel(sample.feature, model);
      return Math.hypot(prediction.x - sample.target.x, prediction.y - sample.target.y);
    });
    const typicalError = Math.max(median(errors), 0.025);
    weights = errors.map((error) => 1 / (1 + (error / (typicalError * 2)) ** 2));
  }

  const errors = samples.map((sample) => {
    const prediction = predictWithModel(sample.feature, model);
    return Math.hypot(prediction.x - sample.target.x, prediction.y - sample.target.y);
  });

  return {
    ...model,
    sampleCount: samples.length,
    meanError: errors.reduce((sum, error) => sum + error, 0) / errors.length,
    maxError: Math.max(...errors),
    medianError: median(errors),
    coverageCount: getCoverageCount(samples),
    xNorm: coefficientNorm(model.xCoefficients),
    yNorm: coefficientNorm(model.yCoefficients),
  };
};

const predictCalibratedTarget = (observation) => {
  if (!calibrationModel) {
    return observation.raw;
  }

  const calibrated = predictWithModel(observation.feature, calibrationModel);

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
    pendingSamples = Array.isArray(saved.pendingSamples)
      ? saved.pendingSamples
      : Array.isArray(saved.samples)
        ? saved.samples
        : [];
    calibrationModel = saved.model ?? fitModelFromSamples(pendingSamples);
  } catch {
    pendingSamples = [];
    calibrationModel = null;
  }
};

const hasCalibrationData = () => Boolean(calibrationModel || pendingSamples.length);

const getAcceptedSampleCount = () => calibrationModel?.sampleCount ?? 0;

const getCurrentTarget = () =>
  calibrationTargets[(getAcceptedSampleCount() + pendingSamples.length) % calibrationTargets.length];

const updateCalibrationUi = () => {
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

  pendingSamples.push({
    feature: [...latestObservation.feature],
    target: getCurrentTarget(),
    createdAt: Date.now(),
  });

  const candidateSamples = calibrationModel?.samples
    ? [...calibrationModel.samples, ...pendingSamples]
    : [...pendingSamples];
  const nextModel = fitModelFromSamples(candidateSamples);

  if (nextModel) {
    calibrationModel = {
      ...nextModel,
      samples: candidateSamples,
    };
    pendingSamples = [];
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

  const pointer = smoothPointer(predictCalibratedTarget(latestObservation));
  confidenceEl.textContent = `${Math.round(pointer.confidence * 100)}%`;
  if (!previouslyObservedHand) {
    updateCalibrationUi();
  }
  drawPointer(pointer, screenRect);
};

const predict = () => {
  try {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const results = handLandmarker.detectForVideo(video, performance.now());
      drawTracking(results);
    }
  } catch (error) {
    console.error(error);
    trackingState = "error";
    clearPointer("Error");
    handCountEl.textContent = "0";
    updateCalibrationUi();
    setStatus(`Hand tracking stopped: ${error.message}. Reload the page to try again.`);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  requestAnimationFrame(predict);
};

const startCamera = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();
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
        "Webcam access was denied. Allow camera access for this site in your browser settings, then reload.",
    };
  }

  if (error?.name === "NotFoundError" || error?.name === "DevicesNotFoundError") {
    return {
      mode: "No camera",
      message: "No webcam was found. Connect or enable a webcam, then reload.",
    };
  }

  if (error?.name === "NotReadableError" || error?.name === "TrackStartError") {
    return {
      mode: "Camera busy",
      message:
        "The webcam could not be opened. Close other apps using it, check system camera access, then reload.",
    };
  }

  return {
    mode: "Error",
    message: `Could not start the webcam: ${error?.message || "unknown error"}. Check browser camera access, then reload.`,
  };
};

const init = async () => {
  loadCalibration();
  updateCalibrationUi();

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

    try {
      handLandmarker = await createHandLandmarker(filesetResolver, "GPU");
    } catch {
      handLandmarker = await createHandLandmarker(filesetResolver, "CPU");
    }
  } catch (error) {
    console.error(error);
    trackingState = "error";
    pointingStateEl.textContent = "Load error";
    updateCalibrationUi();
    setStatus(
      `Could not load the MediaPipe hand tracker: ${error.message}. Check your internet connection, then reload.`,
    );
    return;
  }

  trackingState = "requesting-camera";
  pointingStateEl.textContent = "Awaiting camera";
  updateCalibrationUi();
  setStatus(
    "Webcam access is required to locate your hand. When the browser asks, choose Allow. Frames are processed locally.",
  );

  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    const cameraError = describeCameraError(error);
    trackingState = "error";
    pointingStateEl.textContent = cameraError.mode;
    updateCalibrationUi();
    setStatus(cameraError.message);
    return;
  }

  trackingState = "tracking";
  pointingStateEl.textContent = "No hand";
  updateCalibrationUi();
  setStatus(
    calibrationModel
      ? `Webcam ready. Loaded a calibrated model with ${calibrationModel.sampleCount} samples. Show one pointing hand.`
      : pendingSamples.length
        ? `Webcam ready. Loaded ${pendingSamples.length} pending captures. Resume calibration when ready.`
        : "Webcam ready. Show one pointing hand, or start calibration to adapt the estimate to this setup.",
  );
  predict();
};

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
