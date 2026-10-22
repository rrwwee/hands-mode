import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../src/main.js", import.meta.url), "utf8")
  .replace(/^import \{[\s\S]*?\} from "[^"]+";\n/, "")
  .replace(/\ninit\(\);\s*$/, "\nglobalThis.ready = init();");
const storageKey = "hands-mode.calibration.v2";

// Execute the actual application with in-memory browser/MediaPipe substitutes.
// No network, webcam, graphics, clipboard or OS input is used.
async function app(saved) {
  const data = new Map(saved ? [[storageKey, saved]] : []);
  const elements = new Map();
  const drawing = new Proxy({}, { get: () => () => {} });
  const context = vm.createContext({
    console: { log() {}, error() {} },
    performance: { now: () => 1 },
    requestAnimationFrame() {},
    navigator: {
      mediaDevices: { getUserMedia: async () => ({ simulated: true }) },
      clipboard: { writeText: async () => {} },
    },
    localStorage: {
      getItem: (key) => data.get(key) ?? null,
      setItem: (key, value) => data.set(key, value),
      removeItem: (key) => data.delete(key),
    },
    window: { devicePixelRatio: 1, addEventListener() {}, confirm: () => true },
    document: {
      querySelector: (id) => {
        if (!elements.has(id)) elements.set(id, {
          clientWidth: 800, clientHeight: 500,
          addEventListener() {}, getContext: () => drawing,
          play: async () => {}, currentTime: 0,
        });
        return elements.get(id);
      },
    },
    FilesetResolver: { forVisionTasks: async () => ({}) },
    HandLandmarker: { createFromOptions: async () => ({
      detectForVideo: () => ({ landmarks: [] }),
    }) },
  });
  vm.runInContext(source, context);
  await context.ready;
  return {
    data, elements, context,
    run: (code) => vm.runInContext(code, context),
    json: (code) => JSON.parse(vm.runInContext(`JSON.stringify(${code})`, context)),
  };
}

function samples(count, outlier = false) {
  return Array.from({ length: count }, (_, i) => {
    const x = (i % 5) / 4;
    const y = Math.floor(i / 5) / Math.max(1, Math.ceil(count / 5) - 1);
    return {
      feature: Array.from({ length: 22 }, (_, j) => j === 0 ? 1
        : j === 1 ? x : j === 2 ? y : Math.sin((i + 1) * (j + 1)) * 0.1),
      target: { x: outlier && i === 2 ? 4 : 0.1 + 0.8 * x, y: 0.2 + 0.6 * y },
    };
  });
}

// Frozen scalar-axis reference from the original implementation. This remains
// separate from the shared solve to catch coupling of X and Y or weight changes.
function referenceAxis(samples, axis, weights) {
  const size = samples[0].feature.length;
  const matrix = Array.from({ length: size }, () => Array(size).fill(0));
  const values = Array(size).fill(0);
  samples.forEach((sample, i) => {
    for (let r = 0; r < size; r++) {
      values[r] += sample.feature[r] * sample.target[axis] * weights[i];
      for (let c = 0; c < size; c++) {
        matrix[r][c] += sample.feature[r] * sample.feature[c] * weights[i];
      }
    }
  });
  matrix.forEach((row, i) => { row[i] += 0.01; row.push(values[i]); });
  for (let c = 0; c < size; c++) {
    let pivot = c;
    for (let r = c + 1; r < size; r++) {
      if (Math.abs(matrix[r][c]) > Math.abs(matrix[pivot][c])) pivot = r;
    }
    [matrix[c], matrix[pivot]] = [matrix[pivot], matrix[c]];
    const divisor = matrix[c][c];
    for (let j = c; j <= size; j++) matrix[c][j] /= divisor;
    for (let r = 0; r < size; r++) {
      if (r === c) continue;
      const factor = matrix[r][c];
      for (let j = c; j <= size; j++) matrix[r][j] -= factor * matrix[c][j];
    }
  }
  return matrix.map((row) => row[size]);
}

function referenceModel(samples) {
  let weights = samples.map(() => 1);
  let xCoefficients, yCoefficients, errors;
  for (let pass = 0; pass < 4; pass++) {
    xCoefficients = referenceAxis(samples, "x", weights);
    yCoefficients = referenceAxis(samples, "y", weights);
    errors = samples.map(({ feature, target }) => Math.hypot(
      feature.reduce((sum, v, i) => sum + v * xCoefficients[i], 0) - target.x,
      feature.reduce((sum, v, i) => sum + v * yCoefficients[i], 0) - target.y,
    ));
    const median = [...errors].sort((a, b) => a - b)[Math.floor(errors.length / 2)];
    weights = errors.map((e) => 1 / (1 + (e / (Math.max(median, 0.025) * 2)) ** 2));
  }
  return { xCoefficients, yCoefficients, meanError: errors.reduce((a, b) => a + b) / errors.length,
    maxError: Math.max(...errors), medianError: [...errors].sort((a, b) => a - b)[Math.floor(errors.length / 2)] };
}

function close(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);
}

for (const [name, input] of [
  ["first fit", samples(24)], ["later fit", samples(75)],
  ["outlier", samples(25, true)],
  ["repeated/collinear features", samples(24).map((s) => ({ ...s, feature: Array(22).fill(1) }))],
]) {
  test(`shared fit matches original independent-axis fit: ${name}`, async () => {
    const a = await app();
    a.context.input = input;
    const actual = a.json("fitModelFromSamples(input)");
    const expected = referenceModel(input);
    for (const axis of ["xCoefficients", "yCoefficients"]) {
      actual[axis].forEach((value, i) => close(value, expected[axis][i]));
    }
    for (const key of ["meanError", "maxError", "medianError"]) close(actual[key], expected[key]);
    assert.equal(actual.sampleCount, input.length);
    assert.ok(Number.isFinite(actual.xNorm) && Number.isFinite(actual.yNorm));
  });
}

test("solver pivots both coordinates and rejects a singular matrix", async () => {
  const a = await app();
  assert.deepEqual(a.json("solveLinearSystem([[0, 2], [1, 3]], [[4, 8], [5, 10]])"), [[-1, -2], [2, 4]]);
  assert.throws(() => a.run("solveLinearSystem([[1, 1], [2, 2]], [[1, 2], [2, 4]])"), /unstable/);
});

test("24 captures activate a model; later captures refit; reload retains it", async () => {
  const a = await app();
  a.context.input = samples(25);
  a.run("toggleCalibration()");
  for (let i = 0; i < 23; i++) a.run(`latestObservation = { feature: input[${i}].feature }; captureSample()`);
  assert.equal(a.run("calibrationModel"), null);
  assert.equal(a.run("pendingSamples.length"), 23);
  a.run("latestObservation = { feature: input[23].feature }; captureSample()");
  assert.equal(a.run("calibrationModel.sampleCount"), 24);
  assert.equal(a.run("pendingSamples.length"), 0);
  assert.equal(a.run("calibrationModel.coverageCount"), 24);
  assert.deepEqual(a.json("getCurrentTarget()"), { x: 0.92, y: 0.92 });
  a.run("latestObservation = { feature: input[24].feature }; captureSample()");
  const model = a.json("calibrationModel");
  assert.equal(model.sampleCount, 25);
  assert.equal(model.coverageCount, 25);
  assert.deepEqual(a.json("getCurrentTarget()"), { x: 0.08, y: 0.08 });
  a.run("toggleCalibration(); captureSample()");
  assert.deepEqual(a.json("calibrationModel"), model);
  const loaded = await app(a.data.get(storageKey));
  assert.deepEqual(loaded.json("calibrationModel"), model);
  assert.equal(loaded.elements.get("#sample-count").textContent, "25");
  assert.equal(loaded.run("isCalibrating"), false);
});

test("no hand cannot capture; geometry fallback and calibrated bounds remain", async () => {
  const a = await app();
  a.run("toggleCalibration(); captureSample()");
  assert.equal(a.run("pendingSamples.length"), 0);
  assert.equal(a.elements.get("#capture-sample").disabled, true);
  assert.equal(a.run("observeHand({landmarks: []})"), null);
  assert.equal(a.run("estimateRawTarget(Array.from({length: 21}, () => ({x: 0.5, y: 0.5})))"), null);
  assert.deepEqual(a.json("predictCalibratedTarget({ raw: {x: 0.4, y: 0.7}, feature: [1] })"), { x: 0.4, y: 0.7 });
  a.run("calibrationModel = {xCoefficients: [2], yCoefficients: [-1]}");
  assert.deepEqual(a.json("predictCalibratedTarget({ raw: {x: 0.4, y: 0.7}, feature: [1] })"), { x: 1, y: 0 });
});

test("synthetic landmarks enable capture; losing the hand disables it", async () => {
  const a = await app();
  a.context.hand = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.7, z: 0 }));
  Object.assign(a.context.hand, {
    0: { x: 0.5, y: 0.85, z: 0 },
    6: { x: 0.47, y: 0.55, z: 0 },
    7: { x: 0.45, y: 0.42, z: 0 },
    8: { x: 0.43, y: 0.3, z: 0 },
  });
  a.run("toggleCalibration(); drawTracking({landmarks: [hand]})");
  assert.equal(a.elements.get("#capture-sample").disabled, false);
  assert.equal(a.elements.get("#pointing-state").textContent, "Calibrating");
  const feature = a.json("latestObservation.feature");
  assert.equal(feature.length, 22);
  assert.ok(feature.every(Number.isFinite));
  a.run("captureSample(); drawTracking({landmarks: []}); captureSample()");
  assert.equal(a.run("pendingSamples.length"), 1);
  assert.equal(a.elements.get("#capture-sample").disabled, true);
  assert.equal(a.elements.get("#pointing-state").textContent, "No hand");
});

test("undo/clear pending retain the active model, including after reload", async () => {
  const model = { ...referenceModel(samples(24)), sampleCount: 24, samples: samples(24), coverageCount: 24 };
  const a = await app(JSON.stringify({ model, pendingSamples: samples(2) }));
  a.run("undoSample()");
  assert.equal(a.run("pendingSamples.length"), 1);
  a.run("clearPendingSamples()");
  assert.equal(a.run("pendingSamples.length"), 0);
  assert.deepEqual(a.json("calibrationModel"), model);
  const loaded = await app(a.data.get(storageKey));
  assert.deepEqual(loaded.json("calibrationModel"), model);
});

test("legacy pending samples load; malformed storage safely falls back", async () => {
  const old = await app(JSON.stringify({ samples: samples(3) }));
  assert.equal(old.run("pendingSamples.length"), 3);
  assert.equal(old.run("calibrationModel"), null);
  const corrupt = await app("{");
  assert.equal(corrupt.run("pendingSamples.length"), 0);
  assert.equal(corrupt.run("calibrationModel"), null);
});

test("unavailable storage keeps in-memory captures and reports failure", async () => {
  const a = await app();
  a.context.input = samples(1);
  a.run('localStorage.setItem = () => { throw new Error("storage denied"); }; toggleCalibration(); latestObservation = {feature: input[0].feature}; captureSample()');
  assert.equal(a.run("pendingSamples.length"), 1);
  assert.match(a.elements.get("#status").textContent, /not saved/);
});

test("cancelled reset and storage failure retain data; confirmed reset clears it", async () => {
  const a = await app(JSON.stringify({ pendingSamples: samples(2) }));
  a.run("window.confirm = () => false; resetCalibration()");
  assert.equal(a.run("pendingSamples.length"), 2);
  a.run('window.confirm = () => true; localStorage.removeItem = () => { throw new Error("storage denied"); }; resetCalibration()');
  assert.equal(a.run("pendingSamples.length"), 2);
  a.context.localStorage.removeItem = (key) => a.data.delete(key);
  a.run("resetCalibration()");
  assert.equal(a.run("pendingSamples.length"), 0);
  assert.equal(a.run("calibrationModel"), null);
  assert.equal(a.data.has(storageKey), false);
});
