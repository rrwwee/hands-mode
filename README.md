# Hands Mode

Hands Mode is a small browser-based prototype that estimates where a person is pointing on a screen. It combines a geometric pointing estimate with calibration for a particular user, webcam position, screen and physical setup.

The project explores a simple question: can interaction with a computer feel more direct and spatial than moving a conventional pointer? It is an experimental adaptive-interface prototype, not a finished input method or a claim of adaptive intelligence.

## How it works

1. [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js) finds hand landmarks in each webcam frame.
2. The prototype extends the direction of the index finger to produce an initial geometric estimate of the pointed-at location.
3. During calibration, the user points at targets distributed across the screen and captures samples.
4. Once there are enough samples, the application fits a regularised linear mapping from landmark-derived features to screen coordinates. Later captures refit that mapping.
5. Calibration samples, the fitted model and fit diagnostics are stored locally in the browser so the setup can be resumed.

The visual output shows the estimated finger ray, the estimated screen point, calibration targets and a compact set of diagnostics.

## Run locally

This is a static site with no build step.

```sh
git clone https://github.com/rrwwee/hands-mode.git
cd hands-mode
python3 -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000) in a current desktop browser. Stop the server with <kbd>Ctrl</kbd>+<kbd>C</kbd>.

Opening `index.html` directly is not supported: webcam access and JavaScript modules require a secure browser context. `localhost` is treated as secure by current browsers.

## Requirements

- A current desktop version of Chrome, Edge, Firefox or Safari with JavaScript enabled.
- A webcam and browser permission to use it.
- An internet connection when the page loads. The MediaPipe JavaScript, WebAssembly files and hand-landmark model are fetched from external CDNs.
- A stable camera and screen position while calibrating and using the fitted model.

Browser and hardware combinations have not been tested comprehensively.

## Tests

With Node.js 22 or later installed, run:

```sh
node --test tests/*.test.mjs
```

The tests simulate browser APIs and hand observations without using a webcam,
network connection or GPU. They compare the shared X/Y calibration solve against
the original independent-axis calculation and exercise capture, refit, reload,
pending-sample controls and storage failures. They do not measure real-world
pointing accuracy or replace browser/device testing.

To run the real-browser workflow tests with synthetic landmarks:

```sh
npm ci --ignore-scripts
npx playwright install --only-shell chromium
npm run test:browser
```

These tests use an isolated headless browser with a local server and mocked
MediaPipe/camera input. They do not request camera permission or download the
hand model. CI runs them alongside the Node tests.

## Controls and calibration

1. Load the page and allow webcam access when the browser asks.
2. Hold up one hand and point with your index finger. Before calibration, the green marker uses the geometric estimate.
3. Select **Start Calibration**. Point at the pink target, then press <kbd>Space</kbd> or select **Capture Sample**.
4. Repeat across the target positions. The calibrated mapping becomes active after 24 captures and is refitted after later captures.
5. Select **Pause Calibration** when finished.

Starting or resuming calibration focuses the pointing area. Space captures one
sample per press there. Use Tab to reach the controls; while a button has focus,
Space activates that button instead of capturing a sample.

**Pause Calibration** pauses sample capture; the camera and pointer keep running.
**Stop Camera** releases this app's stream, stops tracking and keeps calibration.
Use **Start Camera** to restart, including after a tracking or camera error.
During startup, **Cancel Camera Start** cancels this app's pending start. A browser
permission prompt may remain open; if its stream arrives later, the app releases
it without resuming tracking. Leaving the page also releases the active stream.

**Undo Pending** removes only the most recent sample that has not yet been included in a fitted model. **Clear Pending** removes all such pending samples. Neither action changes the active fitted model.

**Reset & Start Over** is the destructive option. After confirmation, it removes the active model and all stored calibration samples, clears the in-memory calibration state and starts a fresh calibration session.

**Copy Diagnostics** copies aggregate model information—sample counts, coverage and fit errors—to the clipboard. It does not copy webcam frames.

Non-finite observations and failed numerical fits are rejected. A rejected capture
keeps the active model and saved data. Invalid saved transforms are reported and
rebuilt from valid retained samples when possible; otherwise prediction uses
geometry until calibration succeeds. Finite fit diagnostics still do not establish
real-world pointing accuracy.

## Privacy

- Webcam frames are processed client-side by MediaPipe in the browser.
- This application does not intentionally upload, log or persist webcam images or video. Its code contains no frame-upload endpoint.
- Landmark-derived calibration features, target coordinates, the fitted model and diagnostics are stored in browser `localStorage` under `hands-mode.calibration.v2`.
- **Reset & Start Over** removes that `localStorage` entry.
- MediaPipe code and WebAssembly assets are downloaded from jsDelivr, and the hand-landmark model is downloaded from Google-hosted storage.
- The application itself adds no analytics. MediaPipe maintains its own [privacy notice](https://github.com/google-ai-edge/mediapipe#privacy-notice), including information about performance and usage metrics in MediaPipe Tasks.

## Current limitations

- Pointing accuracy depends strongly on lighting, camera position, hand visibility, screen geometry and how consistently the user points.
- Calibration is specific to the current user and physical setup. Moving the camera or screen can make a saved model inaccurate.
- The mapping is a small regularised linear fit over hand-derived features; it does not continuously learn or reason about user intent.
- The model is fitted and evaluated on the same captured samples, so the displayed fit error is a training diagnostic rather than an independent accuracy measurement.
- Only the first detected hand is used for the pointing estimate, even though the tracker can report up to two hands.
- The prototype draws an estimated point; it does not control the operating-system cursor or click interface elements.
- There is no automated cross-browser, cross-device or accessibility test suite.

## Demo media

> **Screenshot/GIF placeholder:** no suitable media is currently included. Add a clear capture of the calibrated interface here. The recommended recording sequence is listed in the release checklist below.

For a useful demo, record the app at a desktop viewport with the HUD visible:

1. Begin with 23 pending calibration samples already captured.
2. Show a pointing hand moving the geometric estimate.
3. Start or resume calibration and capture the highlighted target once, causing the model to fit.
4. Pause calibration.
5. Move the pointing hand between two or three screen positions while the HUD shows **Mode: Calibrated**, sample count, coverage and fit error.

Avoid including unrelated windows, browser account details or other personal information in the capture.

## Dependencies and attribution

The application has one runtime software dependency:

- [MediaPipe Tasks Vision `0.10.35`](https://www.npmjs.com/package/@mediapipe/tasks-vision), loaded as an ES module and WebAssembly bundle from jsDelivr.

It also downloads Google’s MediaPipe Hand Landmarker model at runtime. MediaPipe is a Google project and is distributed under the [Apache License 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE). Third-party code and model assets remain subject to their own terms.

## License

The original code in this repository is available under the [MIT License](./LICENSE).
