/// <reference lib="webworker" />

import { env, pipeline } from "@xenova/transformers";

type TranscriptChunk = { text: string; timestamp: [number, number | null] };

type WhisperRequest =
  | { type: "init"; model?: string }
  | { type: "set-model"; model: string }
  | { type: "transcribe"; id: number; audio: Float32Array };

type StatusMessage = {
  type: "status";
  model: "whisper";
  state: "loading" | "ready" | "error";
  message?: string;
  detail?: string;
  progress?: number;
};

type UpdateMessage = {
  type: "update";
  id: number;
  text: string;
  chunks: TranscriptChunk[];
};

type CompleteMessage = {
  type: "transcription";
  id: number;
  text: string;
  chunks: TranscriptChunk[];
};

type ErrorMessage = {
  type: "error";
  model: "whisper";
  id?: number;
  message: string;
};

env.allowLocalModels = false;

const ctx: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
const DEFAULT_MODEL_ID = "Xenova/whisper-tiny";

let transcriberPromise: Promise<any> | null = null;
let transcriber: any | null = null;
let selectedModelId = DEFAULT_MODEL_ID;
let effectiveModelId = resolveModelId(DEFAULT_MODEL_ID);

async function ensureTranscriber() {
  if (transcriber) return transcriber;
  if (!transcriberPromise) {
    const modelToLoad = effectiveModelId;
    ctx.postMessage({
      type: "status",
      model: "whisper",
      state: "loading",
      message: `Initializing Whisper (${modelToLoad})...`,
    } satisfies StatusMessage);

    transcriberPromise = pipeline("automatic-speech-recognition", effectiveModelId, {
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
        message: progress.text ?? `Downloading ${effectiveModelId}…`,
          detail: progress.url ?? progress.file,
          progress:
            progress.progress && progress.progress > 0
              ? Math.min(progress.progress, 1)
              : undefined,
        } satisfies StatusMessage);
      },
    })
      .then((instance) => {
        transcriber = instance;
        ctx.postMessage({
          type: "status",
          model: "whisper",
          state: "ready",
        message: `${effectiveModelId} ready`,
        } satisfies StatusMessage);
        return instance;
      })
      .catch((error) => {
        ctx.postMessage({
          type: "status",
          model: "whisper",
          state: "error",
          message: (error as Error).message ?? "Unable to load Whisper",
        } satisfies StatusMessage);
        throw error;
      });
  }
  return transcriberPromise;
}

ctx.addEventListener("message", async (event: MessageEvent<WhisperRequest>) => {
  const data = event.data;

  if (data.type === "init") {
    if (data.model) {
      setSelectedModelId(data.model);
    }
    ensureModelUpToDate();
    await ensureTranscriber().catch(() => {});
    return;
  }

  if (data.type === "set-model") {
    if (data.model) {
      setSelectedModelId(data.model);
      ensureModelUpToDate(true);
      await ensureTranscriber().catch(() => {});
    }
    return;
  }

  if (data.type === "transcribe") {
    try {
      const result = await transcribeAudio(data.id, data.audio);
      if (!result) {
        ctx.postMessage({
          type: "error",
          model: "whisper",
          id: data.id,
          message: "No transcription result",
        } satisfies ErrorMessage);
        return;
      }

      ctx.postMessage({
        type: "transcription",
        id: data.id,
        text: result.text,
        chunks: result.chunks,
      } satisfies CompleteMessage);
    } catch (error) {
      ctx.postMessage({
        type: "error",
        model: "whisper",
        id: data.id,
        message: (error as Error).message ?? "Unable to run Whisper transcription",
      } satisfies ErrorMessage);
    }
  }
});

async function transcribeAudio(id: number, audio: Float32Array) {
  const isDistilWhisper = selectedModelId.startsWith("distil-whisper/");
  ensureModelUpToDate();

  const transcriberInstance = await ensureTranscriber();
  if (!transcriberInstance) {
    return null;
  }

  const timePrecision =
    transcriberInstance.processor.feature_extractor.config.chunk_length /
    transcriberInstance.model.config.max_source_positions;

  const chunksToProcess: {
    tokens: number[];
    finalised: boolean;
  }[] = [
    {
      tokens: [],
      finalised: false,
    },
  ];

  function chunkCallback(chunk: { tokens: number[]; is_last?: boolean }) {
    const last = chunksToProcess[chunksToProcess.length - 1];
    Object.assign(last, chunk);
    last.finalised = true;
    if (!chunk.is_last) {
      chunksToProcess.push({
        tokens: [],
        finalised: false,
      });
    }
  }

  function callbackFunction(item: [{ output_token_ids: number[] }]) {
    const last = chunksToProcess[chunksToProcess.length - 1];
    last.tokens = [...item[0].output_token_ids];
    const data = transcriberInstance.tokenizer._decode_asr(chunksToProcess, {
      time_precision: timePrecision,
      return_timestamps: true,
      force_full_sequences: false,
    });
    ctx.postMessage({
      type: "update",
      id,
      text: data[0],
      chunks: data[1].chunks as TranscriptChunk[],
    } satisfies UpdateMessage);
  }

  const output = await transcriberInstance(audio, {
    top_k: 0,
    do_sample: false,
    chunk_length_s: isDistilWhisper ? 20 : 30,
    stride_length_s: isDistilWhisper ? 3 : 5,
    language: "en",
    task: "transcribe",
    return_timestamps: true,
    force_full_sequences: false,
    callback_function: callbackFunction,
    chunk_callback: chunkCallback,
  }).catch((error: unknown) => {
    ctx.postMessage({
      type: "error",
      model: "whisper",
      id,
      message: (error as Error).message ?? "Transcription failed",
    } satisfies ErrorMessage);
    return null;
  });

  return output as { text: string; chunks: TranscriptChunk[] } | null;
}

function resolveModelId(base: string) {
  if (base.startsWith("distil-whisper/")) {
    return base;
  }
  if (base.endsWith(".en")) {
    return base;
  }
  return `${base}.en`;
}

function setSelectedModelId(model: string) {
  selectedModelId = model;
}

function ensureModelUpToDate(force = false) {
  const resolved = resolveModelId(selectedModelId);
  if (force || resolved !== effectiveModelId) {
    effectiveModelId = resolved;
    if (transcriber && typeof transcriber.dispose === "function") {
      try {
        transcriber.dispose();
      } catch (error) {
        console.warn("Error disposing Whisper model", error);
      }
    }
    transcriber = null;
    transcriberPromise = null;
  }
}
