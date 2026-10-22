import { test, expect } from '@playwright/test';

const key = 'hands-mode.calibration.v2';
const mockTracker = `
export const FilesetResolver = { forVisionTasks: async () => {
  const state = window.__handsTest;
  if (state.deferModel) await new Promise(resolve => { state.resolveModel = resolve; });
  return {};
}};
export const HandLandmarker = { createFromOptions: async (_, options) => {
  window.__handsTest.options = options;
  if (window.__handsTest.loadError) throw new Error('simulated model failure');
  const state = window.__handsTest;
  if (state.deferDetector) await new Promise(resolve => { state.resolveDetector = resolve; });
  const detector = { closed: false, close() { this.closed = true; }, detectForVideo() {
    state.detections++;
    state.processed = state.frame;
    if (state.failure) throw new Error('simulated tracking failure');
    return structuredClone(state.results);
  }};
  state.detectors.push(detector);
  return detector;
}};`;

// Only the browser boundary is mocked. Production HTML/JS, events, canvas and
// localStorage execute unchanged; no camera permission or real media is used.
async function openApp(page, options = {}) {
  const external = [];
  await page.route('**/*', async (route) => {
    const url = route.request().url();
    if (url === 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs') {
      await route.fulfill({ contentType: 'text/javascript', body: mockTracker,
        headers: { 'Access-Control-Allow-Origin': '*' } });
    } else if (url.startsWith('http://127.0.0.1:18765/')) {
      await route.continue();
    } else {
      external.push(url);
      await route.abort();
    }
  });
  await page.addInitScript((options) => {
    window.__handsTest = { frame: 0, processed: -1, results: { landmarks: [] },
      cameraCalls: [], cameraRequests: [], streams: [], detectors: [], detections: 0,
      failure: false, ...options };
    const state = window.__handsTest;
    const raf = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = (callback) => raf((time) => {
      state.lastCallback = callback;
      callback(time);
    });
    Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', {
      get: () => state.frozenVideoTime ?? performance.now() / 1000,
    });
    HTMLMediaElement.prototype.play = async () => {
      if (state.playError) throw new DOMException('simulated playback failure', 'NotSupportedError');
      if (state.deferPlay) await new Promise(resolve => { state.resolvePlay = resolve; });
    };
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      window.__handsTest.cameraCalls.push(constraints);
      if (state.cameraError) throw new DOMException('simulated camera denial', state.cameraError);
      const create = () => {
        const stream = new MediaStream();
        const track = Object.assign(new EventTarget(), {
          kind: 'video', readyState: 'live', stopCalls: 0,
          stop() { this.stopCalls++; this.readyState = 'ended'; },
        });
        stream.getTracks = () => [track];
        state.streams.push({ stream, track });
        return stream;
      };
      if (state.deferCamera) return new Promise(resolve => state.cameraRequests.push(() => resolve(create())));
      return create();
    };
  }, options);
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  return external;
}

function hand(offset = 0) {
  const result = Array.from({ length: 21 }, () => ({ x: 0.5 + offset, y: 0.7, z: 0 }));
  Object.assign(result, {
    0: { x: 0.5 + offset, y: 0.85, z: 0 },
    6: { x: 0.47 + offset, y: 0.55, z: 0 },
    7: { x: 0.45 + offset, y: 0.42, z: 0 },
    8: { x: 0.43 + offset, y: 0.3, z: 0 },
  });
  return result;
}

async function frame(page, hands) {
  const number = await page.evaluate((landmarks) => {
    const state = window.__handsTest;
    state.results = { landmarks };
    return ++state.frame;
  }, hands);
  await expect.poll(() => page.evaluate(() => window.__handsTest.processed)).toBe(number);
}

const mode = (page) => page.locator('#pointing-state');
const samples = (page) => page.locator('#sample-count');
const capture = (page) => page.getByRole('button', { name: 'Capture Sample', exact: true });
const canvas = (page) => page.locator('#overlay').evaluate((element) => element.toDataURL());

test('real UI: startup, hand selection/loss, calibration, pause/resume and saved reload', async ({ page }) => {
  const external = await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await expect(capture(page)).toBeDisabled();
  await frame(page, [hand()]);
  await expect(mode(page)).toHaveText('Geometry');
  const firstPointer = await canvas(page);
  await frame(page, [hand(), hand(0.2)]);
  expect(await canvas(page)).toBe(firstPointer);
  await expect(page.locator('#hand-count')).toHaveText('2');
  await frame(page, [hand(0.2), hand()]);
  expect(await canvas(page)).not.toBe(firstPointer);
  await frame(page, []);
  await expect(mode(page)).toHaveText('No hand');
  await expect(capture(page)).toBeDisabled();
  await frame(page, [hand()]);
  expect(await canvas(page)).toBe(firstPointer);
  await page.getByRole('button', { name: 'Start Calibration', exact: true }).click();
  for (let i = 1; i <= 24; i++) {
    await frame(page, [hand((i % 5) * 0.02)]);
    await capture(page).click();
    await expect(samples(page)).toHaveText(String(i));
  }
  await page.getByRole('button', { name: 'Pause Calibration', exact: true }).click();
  await expect(mode(page)).toHaveText('Calibrated');
  await expect(capture(page)).toBeDisabled();
  await page.getByRole('button', { name: 'Resume Calibration', exact: true }).click();
  await capture(page).click();
  await expect(samples(page)).toHaveText('25');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(samples(page)).toHaveText('25');
  await expect(capture(page)).toBeDisabled();
  await frame(page, [hand()]);
  await expect(mode(page)).toHaveText('Calibrated');
  const saved = await page.evaluate((key) => JSON.parse(localStorage.getItem(key)), key);
  expect(saved.model.samples).toHaveLength(25);
  expect(saved.pendingSamples).toHaveLength(0);
  expect(await page.evaluate(() => window.__handsTest.options.numHands)).toBe(2);
  expect(await page.evaluate(() => window.__handsTest.cameraCalls)).toEqual([
    { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false },
  ]);
  expect(await page.locator('#webcam').evaluate((video) => video.srcObject.getTracks().length)).toBe(1);
  expect(external).toEqual([]);
  const savedBeforeStop = await page.evaluate(key => localStorage.getItem(key), key);
  await page.getByRole('button', { name: 'Stop Camera', exact: true }).click();
  await expect(mode(page)).toHaveText('Camera stopped');
  await expect(samples(page)).toHaveText('25');
  expect(await page.evaluate(key => localStorage.getItem(key), key)).toBe(savedBeforeStop);
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Stop Camera', exact: true })).toBeVisible();
  await frame(page, [hand()]);
  await expect(mode(page)).toHaveText('Calibrated');
  page.once('dialog', (dialog) => dialog.dismiss());
  await page.getByRole('button', { name: 'Reset & Start Over', exact: true }).click();
  await expect(samples(page)).toHaveText('25');
  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Reset & Start Over', exact: true }).click();
  await expect(samples(page)).toHaveText('0');
  await expect(page.locator('#overlay')).toBeFocused();
  await page.keyboard.press('Space');
  await expect(samples(page)).toHaveText('1');
});

test('focused Pause button responds to Space without recording an accidental sample', async ({ page }, testInfo) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Start Calibration', exact: true }).click();
  await page.getByRole('button', { name: 'Pause Calibration', exact: true }).focus();
  await page.evaluate(() => {
    window.__handsTest.keyEvents = [];
    window.addEventListener('keydown', (event) => {
      window.__handsTest.keyEvents.push({ code: event.code, prevented: event.defaultPrevented });
    });
  });
  await page.keyboard.press('Space');
  // A button's native Space activation must remain available.
  expect(await page.evaluate(() => window.__handsTest.keyEvents)).toEqual([
    { code: 'Space', prevented: false },
  ]);
  await expect(page.getByRole('button', { name: 'Start Calibration', exact: true })).toBeVisible();
  await expect(mode(page)).toHaveText('Geometry');
  await expect(samples(page)).toHaveText('0');
  await expect(capture(page)).toBeDisabled();
  await page.screenshot({ path: testInfo.outputPath('keyboard-pause.png') });
});

test('starting calibration keeps the Space shortcut ready, and Capture remains keyboard-operable', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Start Calibration', exact: true }).click();
  await expect(page.locator('#overlay')).toBeFocused();
  await page.keyboard.press('Space');
  await expect(samples(page)).toHaveText('1');
  await capture(page).focus();
  await page.keyboard.press('Space');
  await expect(samples(page)).toHaveText('2');
});

test('holding the capture shortcut records once, not a burst of duplicate samples', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Start Calibration', exact: true }).click();
  await page.locator('body').click({ position: { x: 1100, y: 500 } });
  await page.keyboard.down('Space');
  await page.keyboard.down('Space');
  await page.keyboard.up('Space');
  await expect(samples(page)).toHaveText('1');
});

test('modified Space shortcuts do not capture samples', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Start Calibration', exact: true }).click();
  for (const shortcut of ['Shift+Space', 'Alt+Space', 'Control+Space', 'Meta+Space']) {
    await page.keyboard.press(shortcut);
  }
  await expect(samples(page)).toHaveText('0');
  await expect(mode(page)).toHaveText('Calibrating');
});

test('tracker error releases the stream; retry preserves calibration samples', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Start Calibration', exact: true }).click();
  await capture(page).click();
  await page.evaluate(() => { window.__handsTest.failure = true; ++window.__handsTest.frame; });
  await expect(mode(page)).toHaveText('Error');
  await expect(page.locator('#confidence')).toHaveText('0%');
  await expect(capture(page)).toBeDisabled();
  await expect(page.locator('#status')).toContainText('Start Camera');
  expect(await page.evaluate(() => window.__handsTest.streams[0].track.readyState)).toBe('ended');
  expect(await page.evaluate(() => window.__handsTest.detectors[0].closed)).toBe(true);
  await page.evaluate(() => { window.__handsTest.failure = false; });
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Stop Camera', exact: true })).toBeVisible();
  await expect(samples(page)).toHaveText('1');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Resume Calibration', exact: true }).click();
  await capture(page).click();
  await expect(samples(page)).toHaveText('2');
});

for (const [name, options, expected] of [
  ['camera denial', { cameraError: 'NotAllowedError' }, 'Camera denied'],
  ['camera security rejection', { cameraError: 'SecurityError' }, 'Camera denied'],
  ['camera missing', { cameraError: 'NotFoundError' }, 'No camera'],
  ['camera busy', { cameraError: 'NotReadableError' }, 'Camera busy'],
  ['model load failure', { loadError: true }, 'Load error'],
]) {
  test(`${name} shows recovery guidance and disables capture`, async ({ page }) => {
    await openApp(page, options);
    await expect(mode(page)).toHaveText(expected);
    await expect(capture(page)).toBeDisabled();
    await expect(page.locator('#calibration-toggle')).toBeDisabled();
    await expect(page.locator('#status')).toContainText('Start Camera');
    await page.evaluate(() => { window.__handsTest.cameraError = null; window.__handsTest.loadError = false; });
    await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop Camera', exact: true })).toBeVisible();
  });
}

test('camera stop releases tracks and stale callbacks cannot resume; repeated restarts stay clean', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  for (let i = 0; i < 3; i++) {
    await frame(page, [hand()]);
    await page.evaluate(() => { window.__handsTest.staleCallback = window.__handsTest.lastCallback; });
    await page.getByRole('button', { name: 'Stop Camera', exact: true }).click();
    await expect(mode(page)).toHaveText('Camera stopped');
    await expect(capture(page)).toBeDisabled();
    expect(await page.locator('#webcam').evaluate(video => video.srcObject)).toBe(null);
    const before = await page.evaluate(() => window.__handsTest.detections);
    await page.evaluate(() => window.__handsTest.staleCallback(performance.now()));
    expect(await page.evaluate(() => window.__handsTest.detections)).toBe(before);
    expect(await page.evaluate(() => window.__handsTest.streams.map(({ track }) => track.stopCalls))).toEqual(Array(i + 1).fill(1));
    expect(await page.evaluate(() => window.__handsTest.detectors.every(detector => detector.closed))).toBe(true);
    await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop Camera', exact: true })).toBeVisible();
  }
  await page.evaluate(() => { window.dispatchEvent(new Event('pagehide')); window.dispatchEvent(new Event('pagehide')); });
  expect(await page.evaluate(() => window.__handsTest.streams.every(({ track }) => track.readyState === 'ended' && track.stopCalls === 1))).toBe(true);
});

test('cancelled camera request releases its late stream without replacing a newer start', async ({ page }) => {
  await openApp(page, { deferCamera: true });
  await expect(mode(page)).toHaveText('Awaiting camera');
  await page.getByRole('button', { name: 'Cancel Camera Start', exact: true }).click();
  await expect(mode(page)).toHaveText('Camera stopped');
  await expect(page.locator('#status')).toContainText('pending');
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__handsTest.cameraRequests.length)).toBe(2);
  await page.evaluate(() => window.__handsTest.cameraRequests[0]());
  await expect.poll(() => page.evaluate(() => window.__handsTest.streams[0].track.readyState)).toBe('ended');
  await expect(mode(page)).toHaveText('Awaiting camera');
  await page.evaluate(() => window.__handsTest.cameraRequests[1]());
  await expect(mode(page)).toHaveText('No hand');
  expect(await page.evaluate(() => window.__handsTest.streams.map(({ track }) => track.readyState))).toEqual(['ended', 'live']);
  await page.getByRole('button', { name: 'Stop Camera', exact: true }).click();
});

test('cancelling model load prevents a later camera request', async ({ page }) => {
  await openApp(page, { deferModel: true });
  await expect(mode(page)).toHaveText('Loading');
  await page.getByRole('button', { name: 'Cancel Camera Start', exact: true }).click();
  await page.evaluate(() => window.__handsTest.resolveModel());
  await expect(mode(page)).toHaveText('Camera stopped');
  expect(await page.evaluate(() => window.__handsTest.cameraCalls)).toEqual([]);
});

test('cancelled video playback cannot revive an old session after restart', async ({ page }) => {
  await openApp(page, { deferPlay: true });
  await expect(mode(page)).toHaveText('Awaiting camera');
  await expect.poll(() => page.evaluate(() => window.__handsTest.streams.length)).toBe(1);
  await page.getByRole('button', { name: 'Cancel Camera Start', exact: true }).click();
  expect(await page.evaluate(() => window.__handsTest.streams[0].track.readyState)).toBe('ended');
  await page.evaluate(() => { window.__handsTest.deferPlay = false; });
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Stop Camera', exact: true })).toBeVisible();
  await frame(page, [hand()]);
  await page.evaluate(() => window.__handsTest.resolvePlay());
  await expect(mode(page)).toHaveText('Geometry');
  expect(await page.evaluate(() => window.__handsTest.streams.map(({ track }) => track.readyState))).toEqual(['ended', 'live']);
  await page.getByRole('button', { name: 'Stop Camera', exact: true }).click();
});


test('cancelled detector initialization closes the late detector', async ({ page }) => {
  await openApp(page, { deferDetector: true });
  await expect.poll(() => page.evaluate(() => Boolean(window.__handsTest.resolveDetector))).toBe(true);
  await page.getByRole('button', { name: 'Cancel Camera Start', exact: true }).click();
  await page.evaluate(() => window.__handsTest.resolveDetector());
  await expect.poll(() => page.evaluate(() => window.__handsTest.detectors[0]?.closed)).toBe(true);
  await expect(mode(page)).toHaveText('Camera stopped');
  expect(await page.evaluate(() => window.__handsTest.cameraCalls)).toEqual([]);
});

test('an ended camera track stops the session and allows restart', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await page.evaluate(() => {
    const track = window.__handsTest.streams[0].track;
    track.readyState = 'ended';
    track.dispatchEvent(new Event('ended'));
  });
  await expect(mode(page)).toHaveText('Camera stopped');
  await expect(page.locator('#status')).toContainText('stream ended');
  expect(await page.evaluate(() => window.__handsTest.detectors[0].closed)).toBe(true);
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Stop Camera', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Stop Camera', exact: true }).click();
});

test('non-finite landmarks cannot poison calibration and the next hand recovers', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await frame(page, [hand()]);
  await page.getByRole('button', {name: 'Start Calibration', exact: true}).click();
  await capture(page).click();
  const stored = await page.evaluate(key => localStorage.getItem(key), key);
  const invalid = hand();
  invalid[8].x = NaN;
  await frame(page, [invalid]);
  await expect(mode(page)).toHaveText('Unclear');
  await expect(page.locator('#confidence')).toHaveText('0%');
  await expect(capture(page)).toBeDisabled();
  await expect(samples(page)).toHaveText('1');
  expect(await page.evaluate(key => localStorage.getItem(key), key)).toBe(stored);
  await frame(page, [hand()]);
  await capture(page).click();
  await expect(samples(page)).toHaveText('2');
});

test('invalid saved transform is reported and replaced only by fresh valid calibration', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  const invalid = JSON.stringify({model: {xCoefficients: Array(22).fill(null), yCoefficients: Array(22).fill(0), sampleCount: 24, meanError: 0, maxError: 0}, pendingSamples: []});
  await page.evaluate(({key, invalid}) => localStorage.setItem(key, invalid), {key, invalid});
  await page.reload({waitUntil: 'domcontentloaded'});
  await expect(page.locator('#status')).toContainText(/invalid/i);
  await frame(page, [hand()]);
  await expect(mode(page)).toHaveText('Geometry');
  expect(await page.evaluate(key => localStorage.getItem(key), key)).toBe(invalid);
  await page.getByRole('button', {name: 'Start Calibration', exact: true}).click();
  for (let i = 0; i < 24; i++) {
    await frame(page, [hand((i % 5) * 0.02)]);
    await capture(page).click();
  }
  await page.getByRole('button', {name: 'Pause Calibration', exact: true}).click();
  await expect(mode(page)).toHaveText('Calibrated');
  const valid = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), key);
  expect(valid.model.xCoefficients.every(Number.isFinite)).toBe(true);
  expect(valid.model.sampleCount).toBe(24);
});


test('interrupted camera input clears stale output and preserves calibration for restart', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Start Calibration', exact: true }).click();
  await capture(page).click();
  const stored = await page.evaluate(key => localStorage.getItem(key), key);
  const drawn = await canvas(page);
  await page.evaluate(() => {
    const state = window.__handsTest;
    state.frozenVideoTime = document.querySelector('#webcam').currentTime;
    state.staleCallback = state.lastCallback;
    const track = state.streams[0].track;
    track.muted = true;
    track.dispatchEvent(new Event('mute'));
  });
  await expect(capture(page)).toBeDisabled();
  await expect(mode(page)).toHaveText('Camera stopped');
  await expect(page.locator('#confidence')).toHaveText('0%');
  await expect(page.locator('#hand-count')).toHaveText('0');
  await expect(page.locator('#status')).toContainText('interrupted');
  expect(await canvas(page)).not.toBe(drawn);
  expect(await page.evaluate(() => window.__handsTest.streams[0].track.readyState)).toBe('ended');
  expect(await page.evaluate(() => window.__handsTest.detectors[0].closed)).toBe(true);
  expect(await page.evaluate(key => localStorage.getItem(key), key)).toBe(stored);
  const detections = await page.evaluate(() => window.__handsTest.detections);
  await page.evaluate(() => window.__handsTest.staleCallback(performance.now()));
  expect(await page.evaluate(() => window.__handsTest.detections)).toBe(detections);
  await page.evaluate(() => { window.__handsTest.frozenVideoTime = null; });
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
  await expect(mode(page)).toHaveText('Geometry');
  await page.evaluate(() => window.__handsTest.streams[0].track.dispatchEvent(new Event('ended')));
  await expect(page.getByRole('button', { name: 'Stop Camera', exact: true })).toBeVisible();
  await expect(samples(page)).toHaveText('1');
  await page.getByRole('button', { name: 'Resume Calibration', exact: true }).click();
  await capture(page).click();
  await expect(samples(page)).toHaveText('2');
});

test('camera input interrupted during playback startup cannot revive the retired session', async ({ page }) => {
  await openApp(page, { deferPlay: true });
  await expect(mode(page)).toHaveText('Awaiting camera');
  await page.evaluate(() => window.__handsTest.streams[0].track.dispatchEvent(new Event('mute')));
  await expect(mode(page)).toHaveText('Camera stopped');
  expect(await page.evaluate(() => window.__handsTest.streams[0].track.readyState)).toBe('ended');
  await page.evaluate(() => { window.__handsTest.deferPlay = false; });
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Stop Camera', exact: true })).toBeVisible();
  await frame(page, [hand()]);
  await page.evaluate(() => window.__handsTest.resolvePlay());
  await expect(mode(page)).toHaveText('Geometry');
  expect(await page.evaluate(() => window.__handsTest.streams.map(({ track }) => track.readyState))).toEqual(['ended', 'live']);
});


test('video playback rejection releases resources and retry uses a fresh stream', async ({ page }) => {
  await openApp(page, { playError: true });
  await expect(mode(page)).toHaveText('Error');
  await expect(capture(page)).toBeDisabled();
  await expect(page.locator('#confidence')).toHaveText('0%');
  expect(await page.locator('#webcam').evaluate(video => video.srcObject)).toBe(null);
  expect(await page.evaluate(() => window.__handsTest.streams[0].track.readyState)).toBe('ended');
  expect(await page.evaluate(() => window.__handsTest.detectors[0].closed)).toBe(true);
  await page.evaluate(() => { window.__handsTest.playError = false; });
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click();
  await frame(page, [hand()]);
  await expect(mode(page)).toHaveText('Geometry');
  expect(await page.evaluate(() => window.__handsTest.streams.map(({ track }) => track.readyState))).toEqual(['ended', 'live']);
});
