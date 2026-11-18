# HybridANC Desktop Frontend

React, TypeScript and Vite interface prototype for HybridANC.

## Run

```bash
cd apps/desktop/frontend
npm install
npm run dev
```

Open the URL shown by Vite, normally `http://localhost:5173`.

## Build and lint

```bash
npm run build
npm run lint
npm run preview
```

## Mock-data status

This frontend uses realistic mock device, waveform and ANC metrics only.
It does not capture live audio, generate anti-noise or claim that simulated
ANC is running on physical hardware. Physical ANC output remains visibly disabled.
