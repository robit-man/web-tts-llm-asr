/// <reference lib="webworker" />

import { env, pipeline } from "@xenova/transformers";

type WhisperRequest =
  | { type: "init" }
  | { type: "transcribe"; id: number; audio: Float32Array };

env.allowLocalModels = false;

const ctx: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
const MODEL_ID = "Xenova/whisper-tiny.en";

let transcriberPromise: Promise<any> | null = null;
let transcriber: any | null = null;

async function ensureTranscriber() {
  if (transcriber) return transcriber;
  if (!transcriberPromise) {
    ctx.postMessage({
      type: "status",
      model: "whisper",
      state: "loading",
      message: "Initializing Whisper (tiny.en)...",
    });

    transcriberPromise = pipeline("automatic-speech-recognition", MODEL_ID, {
      quantized: true,
      progress_callback: (progress: {
        progress?: number;
        text?: string;
        url?: string;
        file?: string;
      }) => {
        ctx.postMessage({
          type: "status",
          model: "whisper",
          state: "loading",
          message: progress.text ?? "Downloading Whisper assets...",
          detail: progress.url ?? progress.file,
          progress:
            progress.progress && progress.progress > 0
              ? Math.min(progress.progress, 1)
              : undefined,
        });
      },
    })
      .then((instance) => {
        transcriber = instance;
        ctx.postMessage({
          type: "status",
          model: "whisper",
          state: "ready",
          message: "Whisper ready",
        });
        return instance;
      })
      .catch((error) => {
        ctx.postMessage({
          type: "status",
          model: "whisper",
          state: "error",
          message: (error as Error).message ?? "Unable to load Whisper",
        });
        throw error;
      });
  }
  return transcriberPromise;
}

ctx.addEventListener(
  "message",
  async (event: MessageEvent<WhisperRequest>) => {
    const data = event.data;

    if (data.type === "init") {
      await ensureTranscriber().catch(() => {});
      return;
    }

    if (data.type === "transcribe") {
      try {
        const model = await ensureTranscriber();
        const result = await model(data.audio, {
          chunk_length_s: 30,
          stride_length_s: 5,
          language: "en",
          task: "transcribe",
          return_timestamps: true,
          force_full_sequences: false,
        });

        ctx.postMessage({
          type: "transcription",
          id: data.id,
          text: result.text,
          chunks: result.chunks,
        });
      } catch (error) {
        ctx.postMessage({
          type: "error",
          model: "whisper",
          id: data.id,
          message:
            (error as Error).message ?? "Unable to run Whisper transcription",
        });
      }
    }
  },
);
