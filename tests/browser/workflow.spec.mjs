import { test, expect } from '@playwright/test';

const key = 'hands-mode.calibration.v2';
const mockTracker = `
export const FilesetResolver = { forVisionTasks: async () => ({}) };
export const HandLandmarker = { createFromOptions: async (_, options) => {
  window.__handsTest.options = options;
  if (window.__handsTest.loadError) throw new Error('simulated model failure');
  return { detectForVideo: () => {
    const state = window.__handsTest;
    state.processed = state.frame;
    if (state.failure) throw new Error('simulated tracking failure');
    return structuredClone(state.results);
  }};
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
  await page.addInitScript(({ cameraError, loadError }) => {
    window.__handsTest = { frame: 0, processed: -1, results: { landmarks: [] },
      cameraCalls: [], loadError, failure: false };
    Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', {
      get: () => performance.now() / 1000,
    });
    HTMLMediaElement.prototype.play = async () => {};
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      window.__handsTest.cameraCalls.push(constraints);
      if (cameraError) throw new DOMException('simulated camera denial', cameraError);
      return new MediaStream();
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
  expect(await page.locator('#webcam').evaluate((video) => video.srcObject.getTracks().length)).toBe(0);
  expect(external).toEqual([]);
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

test('tracker error stops capture; reload recovers without losing calibration samples', async ({ page }) => {
  await openApp(page);
  await expect(mode(page)).toHaveText('No hand');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Start Calibration', exact: true }).click();
  await capture(page).click();
  await page.evaluate(() => { window.__handsTest.failure = true; ++window.__handsTest.frame; });
  await expect(mode(page)).toHaveText('Error');
  await expect(page.locator('#confidence')).toHaveText('0%');
  await expect(capture(page)).toBeDisabled();
  await expect(page.locator('#status')).toContainText('Reload the page');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(mode(page)).toHaveText('No hand');
  await expect(samples(page)).toHaveText('1');
  await frame(page, [hand()]);
  await page.getByRole('button', { name: 'Resume Calibration', exact: true }).click();
  await capture(page).click();
  await expect(samples(page)).toHaveText('2');
});

for (const [name, options, expected] of [
  ['camera denial', { cameraError: 'NotAllowedError' }, 'Camera denied'],
  ['model load failure', { loadError: true }, 'Load error'],
]) {
  test(`${name} shows recovery guidance and disables capture`, async ({ page }) => {
    await openApp(page, options);
    await expect(mode(page)).toHaveText(expected);
    await expect(capture(page)).toBeDisabled();
    await expect(page.locator('#calibration-toggle')).toBeDisabled();
    await expect(page.locator('#status')).toContainText(/reload/i);
  });
}
