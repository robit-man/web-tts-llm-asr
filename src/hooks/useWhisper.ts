import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { audioBufferToFloat32 } from "../utils/audio";
import type { ModelStatus } from "../types/models";

type TranscriptChunk = { text: string; timestamp: [number, number | null] };

type WhisperWorkerMessage =
  | {
      type: "status";
      model: "whisper";
      state: ModelStatus["state"];
      message?: string;
      detail?: string;
      progress?: number;
    }
  | {
      type: "update";
      id: number;
      stream?: boolean;
      text: string;
      chunks: TranscriptChunk[];
    }
  | {
      type: "transcription";
      id: number;
      stream?: boolean;
      text: string;
      chunks: TranscriptChunk[];
    }
  | {
      type: "error";
      id?: number;
      stream?: boolean;
      model: "whisper";
      message: string;
    };

const DEFAULT_MODEL = "Xenova/whisper-tiny";

export function useWhisperModel(initialModel = DEFAULT_MODEL) {
  const [status, setStatus] = useState<ModelStatus>({
    model: "whisper",
    label: "Whisper ASR",
    state: "loading",
    message: "Preparing worker...",
  });
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [modelId, setModelId] = useState(initialModel);
  const [partialText, setPartialText] = useState("");
  const [finalText, setFinalText] = useState("");
  const [chunks, setChunks] = useState<TranscriptChunk[]>([]);
  const [hookError, setHookError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestId = useRef(0);
  const pending = useRef<
    Map<
      number,
      {
        resolve: (
          value: {
            text: string;
            chunks: { text: string; timestamp: [number, number | null] }[];
          },
        ) => void;
        reject: (reason?: unknown) => void;
      }
    >
  >(new Map());
  const streamResolvers = useRef<
    Map<
      number,
      {
        resolve: (value: { text: string; chunks: TranscriptChunk[] }) => void;
        reject: (reason?: unknown) => void;
      }
    >
  >(new Map());
  const streamActiveRef = useRef(false);
  const currentStreamIdRef = useRef(0);

  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/whisper-worker.ts", import.meta.url),
      {
        type: "module",
      },
    );
    workerRef.current = worker;
    worker.postMessage({ type: "init", model: initialModel });

    const handleMessage = (event: MessageEvent<WhisperWorkerMessage>) => {
      const message = event.data;

      if (message.type === "status" && message.model === "whisper") {
        setStatus((prev) => ({
          ...prev,
          state: message.state,
          message: message.message,
          detail: message.detail,
          progress: message.progress,
        }));
        return;
      }

      if (message.type === "update") {
        setPartialText(message.text);
        setChunks(message.chunks);
        return;
      }

      if (message.type === "transcription") {
        if (message.stream) {
          setPartialText(message.text);
          setFinalText(message.text);
          setChunks(message.chunks);
          const resolver = streamResolvers.current.get(message.id);
          if (resolver) {
            resolver.resolve({ text: message.text, chunks: message.chunks });
            streamResolvers.current.delete(message.id);
          }
          setIsTranscribing(false);
          return;
        }
        const handlers = pending.current.get(message.id);
        if (handlers) {
          handlers.resolve({
            text: message.text,
            chunks: message.chunks,
          });
          pending.current.delete(message.id);
        }
        setPartialText(message.text);
        setFinalText(message.text);
        setChunks(message.chunks);
        if (pending.current.size === 0) {
          setIsTranscribing(false);
        }
        return;
      }

      if (message.type === "error" && message.model === "whisper") {
        if (message.id !== undefined) {
          const handlers = pending.current.get(message.id);
          if (handlers) {
            handlers.reject(new Error(message.message));
            pending.current.delete(message.id);
          }
          if (pending.current.size === 0) {
            setIsTranscribing(false);
          }
        } else {
          setStatus((prev) => ({
            ...prev,
            state: "error",
            message: message.message,
          }));
        }
        if (message.stream) {
          const resolver = streamResolvers.current.get(message.id ?? currentStreamIdRef.current);
          if (resolver) {
            resolver.reject(new Error(message.message));
            streamResolvers.current.delete(message.id ?? currentStreamIdRef.current);
          }
        }
        setHookError(message.message);
      }
    };

    worker.addEventListener("message", handleMessage);
    return () => {
      worker.removeEventListener("message", handleMessage);
      worker.terminate();
      pending.current.forEach((handlers) =>
        handlers.reject(new Error("Worker terminated")),
      );
      pending.current.clear();
    };
  }, [initialModel]);

  const transcribe = useCallback(async (buffer: AudioBuffer) => {
    if (!workerRef.current) {
      throw new Error("Whisper worker is not ready");
    }

    const audio = audioBufferToFloat32(buffer);
    const id = requestId.current++;
    setIsTranscribing(true);
    setHookError(null);
    setPartialText("");
    setFinalText("");
    setChunks([]);

    return await new Promise<{
      text: string;
      chunks: { text: string; timestamp: [number, number | null] }[];
    }>(
      (resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        workerRef.current?.postMessage(
          {
            type: "transcribe",
            id,
            audio,
          },
          [audio.buffer],
        );
      },
    );
  }, []);

  const transcribeBatch = useCallback(async (blob: Blob) => {
    if (!workerRef.current) {
      throw new Error("Whisper worker is not ready");
    }

    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new AudioContext({ sampleRate: 16000 });
    let audioBuffer: AudioBuffer;
    try {
      audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    } finally {
      await audioContext.close();
    }

    const audio = audioBufferToFloat32(audioBuffer);
    const id = requestId.current++;
    setIsTranscribing(true);
    setHookError(null);
    setPartialText("");
    setFinalText("");
    setChunks([]);

    return await new Promise<{
      text: string;
      chunks: { text: string; timestamp: [number, number | null] }[];
    }>(
      (resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        workerRef.current?.postMessage(
          {
            type: "transcribe-batch",
            id,
            audio,
          },
          [audio.buffer],
        );
      },
    );
  }, []);

  const startStream = useCallback(async () => {
    if (!workerRef.current) {
      throw new Error("Whisper worker is not ready");
    }
    const streamId = currentStreamIdRef.current + 1;
    currentStreamIdRef.current = streamId;
    streamActiveRef.current = true;
    setIsTranscribing(true);
    workerRef.current.postMessage({ type: "start-stream", streamId });
  }, []);

  const pushStreamChunk = useCallback((chunk: Float32Array) => {
    if (!workerRef.current || !streamActiveRef.current) {
      return;
    }
    const streamId = currentStreamIdRef.current;
    workerRef.current.postMessage(
      {
        type: "push-stream-chunk",
        streamId,
        audio: chunk,
      },
      [chunk.buffer],
    );
  }, []);

  const finishStream = useCallback(async () => {
    if (!workerRef.current || !streamActiveRef.current) {
      return null;
    }
    const streamId = currentStreamIdRef.current;
    const resultPromise = new Promise<{ text: string; chunks: TranscriptChunk[] }>((resolve, reject) => {
      streamResolvers.current.set(streamId, { resolve, reject });
    });
    workerRef.current.postMessage({ type: "end-stream", streamId });
    const result = await resultPromise.finally(() => {
      streamActiveRef.current = false;
      streamResolvers.current.delete(streamId);
      setIsTranscribing(false);
    });
    return result;
  }, []);

  const setModel = useCallback(
    (nextModel: string) => {
      if (!workerRef.current) return;
      if (nextModel === modelId) return;
      setModelId(nextModel);
      setStatus((prev) => ({
        ...prev,
        state: "loading",
        message: `Loading ${nextModel}…`,
        progress: undefined,
      }));
      workerRef.current.postMessage({ type: "set-model", model: nextModel });
    },
    [modelId],
  );

  return useMemo(
    () => ({
      status,
      isTranscribing,
      transcribe,
      transcribeBatch,
      model: modelId,
      setModel,
      partialText,
      finalText,
      chunks,
      error: hookError,
      startStream,
      pushStreamChunk,
      finishStream,
    }),
    [
      chunks,
      finalText,
      hookError,
      isTranscribing,
      modelId,
      partialText,
      startStream,
      pushStreamChunk,
      finishStream,
      setModel,
      status,
      transcribe,
      transcribeBatch,
    ],
  );
}
