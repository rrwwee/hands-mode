import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  timeout: 20_000,
  expect: { timeout: 4_000 },
  use: {
    baseURL: 'http://127.0.0.1:18765',
    browserName: 'chromium',
    channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
    headless: true,
    viewport: { width: 1280, height: 900 },
    permissions: [],
    launchOptions: { args: ['--disable-gpu'] },
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'python3 -m http.server 18765 --bind 127.0.0.1',
    url: 'http://127.0.0.1:18765',
    reuseExistingServer: false,
  },
});
