/// <reference lib="webworker" />

import { env, pipeline } from "@xenova/transformers";

type TranscriptChunk = { text: string; timestamp: [number, number | null] };

type WhisperRequest =
  | { type: "init"; model?: string }
  | { type: "set-model"; model: string }
  | { type: "transcribe"; id: number; audio: Float32Array }
  | { type: "transcribe-batch"; id: number; audio: Float32Array }
  | { type: "start-stream"; streamId: number }
  | { type: "push-stream-chunk"; streamId: number; audio: Float32Array }
  | { type: "end-stream"; streamId: number };

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
  stream?: boolean;
  text: string;
  chunks: TranscriptChunk[];
};

type CompleteMessage = {
  type: "transcription";
  id: number;
  stream?: boolean;
  text: string;
  chunks: TranscriptChunk[];
};

type ErrorMessage = {
  type: "error";
  model: "whisper";
  id?: number;
  stream?: boolean;
  message: string;
};

env.allowLocalModels = false;

const ctx: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
const DEFAULT_MODEL_ID = "Xenova/whisper-tiny";
const STREAM_SAMPLE_RATE = 16000;
const STREAM_TAIL_SECONDS = 4;
const STREAM_MIN_CHUNK_SAMPLES = STREAM_SAMPLE_RATE * 0.5; // 0.5 seconds minimum

let transcriberPromise: Promise<any> | null = null;
let transcriber: any | null = null;
let selectedModelId = DEFAULT_MODEL_ID;
let effectiveModelId = resolveModelId(DEFAULT_MODEL_ID);

type StreamState = {
  active: boolean;
  id: number;
  chunks: Float32Array[];
  processing: boolean;
  rerun: boolean;
  finalize: boolean;
  transcript: string;
  tail: Float32Array | null;
};

const streamState: StreamState = {
  active: false,
  id: 0,
  chunks: [],
  processing: false,
  rerun: false,
  finalize: false,
  transcript: "",
  tail: null,
};

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

  if (data.type === "transcribe-batch") {
    try {
      const result = await transcribeAudio(data.id, data.audio, { stream: false });
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
        message: (error as Error).message ?? "Unable to run batch transcription",
        } satisfies ErrorMessage);
    }
  }

  if (data.type === "start-stream") {
    streamState.active = true;
    streamState.id = data.streamId;
    streamState.chunks = [];
    streamState.processing = false;
    streamState.rerun = false;
    streamState.finalize = false;
    streamState.transcript = "";
    streamState.tail = null;
    return;
  }

  if (data.type === "push-stream-chunk") {
    if (!streamState.active || data.streamId !== streamState.id) {
      return;
    }
    streamState.chunks.push(data.audio);
    scheduleStreamProcess(false);
    return;
  }

  if (data.type === "end-stream") {
    if (!streamState.active || data.streamId !== streamState.id) {
      return;
    }
    streamState.finalize = true;
    scheduleStreamProcess(true);
    return;
  }
});

async function transcribeAudio(
  id: number,
  audio: Float32Array,
  options?: {
    stream?: boolean;
    suppressUpdate?: boolean;
    suppressComplete?: boolean;
  },
) {
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
    if (options?.suppressUpdate) {
      return;
    }
    const data = transcriberInstance.tokenizer._decode_asr(chunksToProcess, {
      time_precision: timePrecision,
      return_timestamps: true,
      force_full_sequences: false,
    });
    ctx.postMessage({
      type: "update",
      id,
      stream: options?.stream ?? false,
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
      stream: options?.stream ?? false,
      message: (error as Error).message ?? "Transcription failed",
    } satisfies ErrorMessage);
    return null;
  });

  if (!output) {
    return null;
  }

  if (!options?.suppressComplete) {
    ctx.postMessage({
      type: "transcription",
      id,
      stream: options?.stream ?? false,
      text: output.text,
      chunks: output.chunks as TranscriptChunk[],
    } satisfies CompleteMessage);
  }

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

function scheduleStreamProcess(forceFinal: boolean) {
  if (!streamState.active) {
    return;
  }
  if (forceFinal) {
    streamState.finalize = true;
  }
  if (streamState.processing) {
    streamState.rerun = true;
    return;
  }
  streamState.processing = true;
  void processStream().finally(() => {
    streamState.processing = false;
    if (streamState.rerun || (streamState.finalize && streamState.active)) {
      streamState.rerun = false;
      scheduleStreamProcess(false);
    }
  });
}

async function processStream() {
  if (!streamState.active) {
    return;
  }
  const batch = concatWithTail(streamState.tail, streamState.chunks);
  if (!batch.length) {
    return;
  }

  const finalize = streamState.finalize;
  const totalNewSamples = streamState.chunks.reduce((sum, chunk) => sum + chunk.length, 0);

  if (!finalize && totalNewSamples < STREAM_MIN_CHUNK_SAMPLES) {
    streamState.chunks = [batch];
    return;
  }
  if (finalize) {
    const result = await transcribeAudio(streamState.id, batch, {
      stream: true,
      suppressUpdate: false, // Enable token-level updates
      suppressComplete: true,
    });
    if (result) {
      appendStreamTranscript(result);
      ctx.postMessage({
        type: "transcription",
        id: streamState.id,
        stream: true,
        text: streamState.transcript,
        chunks: result.chunks as TranscriptChunk[],
      } satisfies CompleteMessage);
    }
    streamState.active = false;
    streamState.finalize = false;
    streamState.chunks = [];
    streamState.tail = null;
  } else {
    const result = await transcribeAudio(streamState.id, batch, {
      stream: true,
      suppressUpdate: false, // Enable token-level updates
      suppressComplete: true,
    });
    if (result) {
      appendStreamTranscript(result);
      ctx.postMessage({
        type: "update",
        id: streamState.id,
        stream: true,
        text: streamState.transcript,
        chunks: result.chunks as TranscriptChunk[],
      } satisfies UpdateMessage);
      streamState.tail = getTail(batch);
    }
  }
}

function concatWithTail(tail: Float32Array | null, chunks: Float32Array[]) {
  const total =
    (tail ? tail.length : 0) + chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  if (total === 0) {
    return new Float32Array(0);
  }
  const result = new Float32Array(total);
  let offset = 0;
  if (tail) {
    result.set(tail, offset);
    offset += tail.length;
  }
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  chunks.length = 0;
  return result;
}

function getTail(audio: Float32Array) {
  const tailSamples = Math.min(audio.length, STREAM_SAMPLE_RATE * STREAM_TAIL_SECONDS);
  if (tailSamples === 0) {
    return null;
  }
  return audio.slice(audio.length - tailSamples);
}

function appendStreamTranscript(result: { text: string }) {
  const text = result.text?.trim();
  if (!text) {
    return;
  }
  streamState.transcript = `${streamState.transcript} ${text}`.trim();
}
