# Trifecta Voice Lab

A progressive web application that unifies three browser-native ML stacks:

- **Whisper ASR** (via [Transformers.js](https://github.com/xenova/transformers.js)) for low-latency speech ingestion.
- **WebLLM** (via [`@mlc-ai/web-llm`](https://github.com/mlc-ai/web-llm)) for in-tab reasoning with quantized small models.
- **Piper TTS** (via [`onnxruntime-web`](https://onnxruntime.ai/docs/get-started/with-javascript.html)) for fully local speech synthesis.

The UI preloads all three models, surfaces their status, and then runs a hands-free loop: *record speech → transcribe → build an LLM response → render the answer with Piper*. Everything stays inside the browser and the app installs as a PWA.

Live RMS tracking, a 6-band equalizer, and silence detection mirror the microphone HUD shipped in `/home/robit/Respositories/Cygnus-App-Replit-1`, so the recorder auto-pauses a little over a second after you stop speaking. Whisper and Piper models can be swapped from dropdowns before each turn (again mirroring the Cygnus demo’s UX).

## Getting started

```bash
npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173`). The first load fetches the Whisper, WebLLM, and Piper artefacts (~30 MB in total), so let the status cards reach **Ready** before recording.

To create a production build:

```bash
npm run build
npm run preview
```

The PWA manifest, icons, and service worker are generated automatically (`vite-plugin-pwa`).

## Project structure

```
src/
  components/         # Status cards, recorder CTA, conversation log
  hooks/              # Wrap Whisper, WebLLM, Piper workers + recorder/RMS plumbing
  lib/piper.ts        # Port of the Piper text/phoneme/audio pipeline
  utils/              # Audio helpers and IndexedDB model cache
  workers/            # Dedicated workers for Whisper ASR + Piper TTS
public/
  onnx-runtime/       # ORT WASM binaries copied from the Piper demo
  tts-model/          # en_US Piper model + config
  icons/              # PWA icons (generated)
```

## How it works

1. **Model boot** – `useWhisperModel`, `useWebLLM`, and `usePiperModel` each spin up their worker/runtime immediately and stream fine-grained progress into shared `ModelStatusCard` components.
2. **Ingress** – `useAudioRecorder` wraps `MediaRecorder`, normalises microphone blobs to mono 16 kHz buffers, continuously tracks RMS/EQ energy, and auto-stops once 1.3 s of silence is detected.
3. **Reasoning** – Whisper output seeds the rolling conversation history (plus a concise system prompt). `useWebLLM` keeps a warm `MLCEngine` and calls `chat.completions.create` per turn.
4. **Egress** – Piper runs inside a worker with the original model-cache/on-device phonemizer pipeline. A merged waveform is returned, normalised, trimmed, and played in the UI as soon as it lands.
5. **Model controls** – The Whisper selector swaps between Tiny/Base/Small checkpoints before recording, while the Piper selector surfaces the first 60 LibriTTS voices (all 904 are ready once the worker is up).
5. **Offline install** – `vite-plugin-pwa` precaches the app shell and the large WASM artefacts (limit bumped to 35 MB) so the experience keeps working when re-opened.

## Notes & troubleshooting

- The bundled Piper WASM files (in `public/onnx-runtime`) and model weights (~24 MB) were copied directly from `/home/robit/Respositories/piper-tts-web-demo/public`.
- Whisper Tiny EN is loaded by default for responsiveness. Update `MODEL_ID` inside `src/workers/whisper-worker.ts` if you need multilingual or larger variants.
- WebLLM currently targets `Llama-3.2-1B-Instruct-q4f32_1-MLC` for manageable VRAM. Swap the `DEFAULT_MODEL` constant in `useWebLLM.ts` if you prefer another prebuilt model.
- Browsers without Audio Worklets or MediaRecorder (older Safari/iOS) will blocks microphone capture. The UI surfaces these errors via the alert banner.
- Build time warnings about large chunks are expected—the models and WASM runtimes are hefty. They are documented inside the Vite logs.

## Next ideas

1. Allow speaker/voice and rate selection by exposing more of the Piper worker options.
2. Stream WebLLM tokens incrementally and feed Piper chunk-by-chunk for lower perceived latency.
3. Persist conversation history (or transcribed clips) to IndexedDB to review previous sessions offline.
