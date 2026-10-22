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
  const drawCalls = [];
  const drawingErrors = new Map();
  const drawing = new Proxy({}, { get: (_, method) => () => {
    drawCalls.push(method);
    if (drawingErrors.has(method)) throw drawingErrors.get(method);
  } });
  const context = vm.createContext({
    console: { log() {}, error() {} },
    performance: { now: () => 1 },
    requestAnimationFrame() {},
    cancelAnimationFrame() {},
    navigator: {
      mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) },
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
          addEventListener() {}, getContext: () => drawing, focus() {},
          play: async () => {}, pause() {}, currentTime: 0,
        });
        return elements.get(id);
      },
    },
    FilesetResolver: { forVisionTasks: async () => ({}) },
    HandLandmarker: { createFromOptions: async () => ({
      close() {},
      detectForVideo: () => ({ landmarks: [] }),
    }) },
  });
  vm.runInContext(source, context);
  await context.ready;
  return {
    data, elements, context, drawCalls, drawingErrors,
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

function pointingHand(xOffset = 0) {
  const hand = Array.from({ length: 21 }, () => ({ x: 0.5 + xOffset, y: 0.7, z: 0 }));
  Object.assign(hand, {
    0: { x: 0.5 + xOffset, y: 0.85, z: 0 },
    6: { x: 0.47 + xOffset, y: 0.55, z: 0 },
    7: { x: 0.45 + xOffset, y: 0.42, z: 0 },
    8: { x: 0.43 + xOffset, y: 0.3, z: 0 },
  });
  return hand;
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
  a.context.hand = pointingHand();
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

test("two-hand results use only the first reported hand, without fallback", async () => {
  const a = await app();
  a.context.first = pointingHand();
  a.context.second = pointingHand(0.2);
  a.context.unclear = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const first = a.json("observeHand({landmarks: [first]})");
  assert.deepEqual(a.json("observeHand({landmarks: [first, second]})"), first);
  const second = a.json("observeHand({landmarks: [second]})");
  assert.deepEqual(a.json("observeHand({landmarks: [second, first]})"), second);
  assert.notDeepEqual(first.feature, second.feature);
  a.run("toggleCalibration(); drawTracking({landmarks: [first, second]}); captureSample()");
  assert.equal(a.elements.get("#hand-count").textContent, "2");
  a.run("drawTracking({landmarks: [unclear, second]}); captureSample()");
  assert.equal(a.run("pendingSamples.length"), 1);
  assert.equal(a.run("latestObservation"), null);
  assert.equal(a.run("smoothedPointer"), null);
  assert.equal(a.elements.get("#capture-sample").disabled, true);
  assert.equal(a.elements.get("#pointing-state").textContent, "Unclear");
  assert.equal(a.elements.get("#confidence").textContent, "0%");
  a.run("drawTracking({landmarks: [second]})");
  assert.equal(a.run("smoothedPointer.x"), a.run("latestObservation.raw.x"));
  assert.equal(a.run("smoothedPointer.y"), a.run("latestObservation.raw.y"));
  assert.equal(a.elements.get("#capture-sample").disabled, false);
});

test("mode follows calibration state even between video frames", async () => {
  const a = await app();
  a.context.hand = pointingHand();
  a.run("drawTracking({landmarks: [hand]})");
  assert.equal(a.elements.get("#pointing-state").textContent, "Geometry");
  a.run("toggleCalibration()");
  assert.equal(a.elements.get("#pointing-state").textContent, "Calibrating");
  a.run("calibrationModel = {sampleCount: 24, meanError: 0, maxError: 0}; toggleCalibration()");
  assert.equal(a.elements.get("#pointing-state").textContent, "Calibrated");
  a.run("resetCalibration()");
  assert.equal(a.elements.get("#pointing-state").textContent, "Calibrating");
  a.run("toggleCalibration()");
  assert.equal(a.elements.get("#pointing-state").textContent, "Geometry");
});

test("tracking errors clear stale confidence and pointer and prevent capture", async () => {
  const a = await app();
  a.context.hand = pointingHand();
  a.run("toggleCalibration(); drawTracking({landmarks: [hand]})");
  assert.notEqual(a.elements.get("#confidence").textContent, "0%");
  assert.notEqual(a.run("smoothedPointer"), null);
  a.drawCalls.length = 0;
  a.run('video.currentTime = 1; handLandmarker.detectForVideo = () => { throw new Error("lost tracker"); }; predict(); captureSample()');
  assert.equal(a.run("latestObservation"), null);
  assert.equal(a.run("smoothedPointer"), null);
  assert.equal(a.elements.get("#confidence").textContent, "0%");
  assert.equal(a.elements.get("#pointing-state").textContent, "Error");
  assert.equal(a.elements.get("#hand-count").textContent, "0");
  assert.ok(a.drawCalls.includes("clearRect"), "the stale pointer canvas must be cleared");
  assert.equal(a.elements.get("#capture-sample").disabled, true);
  assert.equal(a.run("pendingSamples.length"), 0);
});

test("drawing failure also stops capture without re-entering drawing", async () => {
  const a = await app();
  a.context.hand = pointingHand();
  a.run("toggleCalibration(); drawTracking({landmarks: [hand]})");
  a.drawCalls.length = 0;
  a.drawingErrors.set("fillRect", new Error("drawing failed"));
  a.run("video.currentTime = 1; handLandmarker.detectForVideo = () => ({landmarks: [hand]}); predict(); captureSample()");
  assert.equal(a.run("latestObservation"), null);
  assert.equal(a.run("smoothedPointer"), null);
  assert.equal(a.run("pendingSamples.length"), 0);
  assert.equal(a.elements.get("#capture-sample").disabled, true);
  assert.equal(a.elements.get("#pointing-state").textContent, "Error");
  assert.equal(a.drawCalls.filter((call) => call === "fillRect").length, 1);
  assert.ok(a.drawCalls.includes("clearRect"));
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
  a.run("stopCamera()");
  await a.run("startTracking()");
  assert.equal(a.run("pendingSamples.length"), 1);
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

test('non-finite or incomplete landmarks are rejected and tracking can recover', async () => {
  const a = await app();
  a.context.hand = pointingHand();
  a.run('toggleCalibration(); drawTracking({landmarks: [hand]})');
  const first = a.json('latestObservation.feature');
  for (const bad of [NaN, Infinity, -Infinity]) {
    a.context.bad = bad;
    a.run('hand[8].x = bad; drawTracking({landmarks: [hand]}); captureSample()');
    assert.equal(a.run('latestObservation'), null);
    assert.equal(a.run('smoothedPointer'), null);
    assert.equal(a.elements.get('#capture-sample').disabled, true);
    assert.equal(a.run('pendingSamples.length'), 0);
  }
  assert.equal(a.run('observeHand({landmarks: [[{x: 0, y: 0}]]})'), null);
  a.context.hand = pointingHand();
  a.run('drawTracking({landmarks: [hand]})');
  assert.deepEqual(a.json('latestObservation.feature'), first);
  assert.equal(a.elements.get('#capture-sample').disabled, false);
  a.context.world = pointingHand();
  a.context.world[8].z = Infinity;
  assert.equal(a.run('observeHand({landmarks: [hand], worldLandmarks: [world]})'), null);
});

test('non-finite calibration samples and finite overflow cannot produce a model', async () => {
  const a = await app();
  for (const bad of [NaN, Infinity, -Infinity, 1e308]) {
    a.context.input = samples(24);
    a.context.input[0].feature[1] = bad;
    assert.equal(a.run('fitModelFromSamples(input)'), null);
  }
  a.context.input = samples(24);
  a.context.input[0].target.x = NaN;
  assert.equal(a.run('fitModelFromSamples(input)'), null);
});

test('a rejected refit leaves the valid model, pending captures and saved bytes unchanged', async () => {
  const a = await app();
  a.context.input = samples(24);
  a.run('calibrationModel = {...fitModelFromSamples(input), samples: input}; saveCalibration(); toggleCalibration()');
  const before = a.json('calibrationModel');
  const stored = a.data.get(storageKey);
  a.context.badFeature = samples(1)[0].feature;
  a.context.badFeature[1] = 1e308;
  a.run('latestObservation = {feature: badFeature}; captureSample()');
  assert.deepEqual(a.json('calibrationModel'), before);
  assert.equal(a.run('pendingSamples.length'), 0);
  assert.equal(a.data.get(storageKey), stored);
  assert.match(a.elements.get('#status').textContent, /not accepted|rejected/i);
});

test('saved null or stale coefficients are rejected without dropping valid pending samples', async () => {
  for (const coefficients of [Array(22).fill(null), [1, 2], Array(23).fill(0)]) {
    const a = await app(JSON.stringify({
      model: { xCoefficients: coefficients, yCoefficients: Array(22).fill(0), sampleCount: 24, meanError: 0, maxError: 0 },
      pendingSamples: samples(2),
    }));
    assert.equal(a.run('calibrationModel'), null);
    assert.equal(a.run('pendingSamples.length'), 2);
    assert.match(a.elements.get('#status').textContent, /invalid|rejected/i);
  }
});

test('invalid pending data does not replace a valid saved model', async () => {
  const a = await app();
  a.context.input = samples(24);
  const model = a.json('({...fitModelFromSamples(input), samples: input})');
  const pending = samples(2);
  pending[0].feature[2] = null;
  const loaded = await app(JSON.stringify({ model, pendingSamples: pending }));
  assert.deepEqual(loaded.json('calibrationModel'), model);
  assert.equal(loaded.run('pendingSamples.length'), 1);
});

test('near-degenerate finite inputs retain finite ridge-fit output', async () => {
  const a = await app();
  a.context.input = samples(24).map((sample, i) => ({...sample, feature: Array.from({length: 22}, (_, j) => j === 0 ? 1 : 0.5 + i * 1e-12)}));
  const model = a.json('fitModelFromSamples(input)');
  assert.ok(model.xCoefficients.every(Number.isFinite));
  assert.ok(model.yCoefficients.every(Number.isFinite));
  assert.ok(Number.isFinite(model.meanError));
});

test('first-fit resume retains samples once and refitting clears stale smoothing', async () => {
  const a = await app(JSON.stringify({ samples: samples(24) }));
  assert.equal(a.run('calibrationModel.sampleCount'), 24);
  assert.equal(a.run('calibrationModel.samples.length'), 24);
  assert.equal(a.run('pendingSamples.length'), 0);
  assert.deepEqual(a.json('getCurrentTarget()'), {x: 0.92, y: 0.92});
  a.context.hand = pointingHand();
  a.run('drawTracking({landmarks: [hand]}); toggleCalibration(); captureSample()');
  assert.equal(a.run('calibrationModel.sampleCount'), 25);
  assert.equal(a.run('smoothedPointer'), null);
});

test('a non-finite prediction is not drawn and a valid observation can recover', async () => {
  const a = await app();
  a.context.hand = pointingHand();
  a.run('calibrationModel = {xCoefficients: Array(22).fill(1e308), yCoefficients: Array(22).fill(0)}; drawTracking({landmarks: [hand]})');
  assert.equal(a.elements.get('#pointing-state').textContent, 'Invalid prediction');
  assert.equal(a.run('latestObservation'), null);
  assert.equal(a.run('smoothedPointer'), null);
  assert.equal(a.elements.get('#capture-sample').disabled, true);
  a.run('calibrationModel = null; drawTracking({landmarks: [hand]})');
  assert.equal(a.elements.get('#pointing-state').textContent, 'Geometry');
});

test('an invalid cached transform is rebuilt from its retained valid training samples', async () => {
  const retained = samples(24);
  const stored = JSON.stringify({model: {xCoefficients: [null], yCoefficients: [null], sampleCount: 24, samples: retained}, pendingSamples: []});
  const a = await app(stored);
  assert.equal(a.run('calibrationModel.sampleCount'), 24);
  assert.deepEqual(a.json('calibrationModel.samples'), retained);
  assert.equal(a.run('pendingSamples.length'), 0);
  assert.match(a.elements.get('#status').textContent, /rebuilt/);
  assert.equal(a.data.get(storageKey), stored);
});

test('intermediate solver overflow is rejected even if final coefficients might appear finite', async () => {
  const a = await app();
  assert.throws(() => a.run('solveLinearSystem([[1e308,1e308],[-1e308,1e308]], [[1,2],[3,4]])'), /unstable/);
});

test('near-threshold finger geometry stays finite, while missing and null coordinates are rejected', async () => {
  const a = await app();
  a.context.hand = pointingHand();
  a.run('hand[8] = {...hand[7], x: hand[7].x + 0.0059}');
  assert.equal(a.run('observeHand({landmarks:[hand]})'), null);
  a.run('hand[8].x = hand[7].x + 0.0061');
  assert.ok(a.json('observeHand({landmarks:[hand]}).feature').every(Number.isFinite));
  a.run('hand[8].z = null');
  assert.equal(a.run('observeHand({landmarks:[hand]})'), null);
});
