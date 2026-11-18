# HybridANC Native Bridge

The native bridge is reserved for a future Electron/Tauri or JUCE-to-Python
host boundary. The initial integration deliberately uses HTTP/SSE from the
React frontend to the localhost Python service.

## Current Integration

- Python engine: `prototype/python/backend_service.py`
- API contract: `docs/API_CONTRACT.md`
- Frontend HTTP adapter: `apps/desktop/frontend/src/services/audioService.ts`
- Frontend service selection:
  - no `VITE_HYBRIDANC_API_URL`: mock service
  - `VITE_HYBRIDANC_API_URL=http://127.0.0.1:8765`: HTTP service

## Rules

- Bind Python only to `127.0.0.1`.
- Never transmit raw audio frames through JSON or SSE.
- Keep DSP independent of HTTP classes.
- Stop capture/playback if the frontend disconnects or the backend is stopped.
- Never expose physical ANC without measured calibration and native safety.
