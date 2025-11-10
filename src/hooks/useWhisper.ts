import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { audioBufferToFloat32 } from "../utils/audio";
import type { ModelStatus } from "../types/models";

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
      type: "transcription";
      id: number;
      text: string;
      chunks: { text: string }[];
    }
  | {
      type: "error";
      id?: number;
      model: "whisper";
      message: string;
    };

export function useWhisperModel() {
  const [status, setStatus] = useState<ModelStatus>({
    model: "whisper",
    label: "Whisper ASR",
    state: "loading",
    message: "Preparing worker...",
  });
  const [isTranscribing, setIsTranscribing] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const requestId = useRef(0);
  const pending = useRef<
    Map<
      number,
      {
        resolve: (value: { text: string; chunks: { text: string }[] }) => void;
        reject: (reason?: unknown) => void;
      }
    >
  >(new Map());

  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/whisper-worker.ts", import.meta.url),
      {
        type: "module",
      },
    );
    workerRef.current = worker;
    worker.postMessage({ type: "init" });

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

      if (message.type === "transcription") {
        const handlers = pending.current.get(message.id);
        if (handlers) {
          handlers.resolve({
            text: message.text,
            chunks: message.chunks,
          });
          pending.current.delete(message.id);
        }
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
  }, []);

  const transcribe = useCallback(async (buffer: AudioBuffer) => {
    if (!workerRef.current) {
      throw new Error("Whisper worker is not ready");
    }

    const audio = audioBufferToFloat32(buffer);
    const id = requestId.current++;
    setIsTranscribing(true);

    return await new Promise<{ text: string; chunks: { text: string }[] }>(
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

  return useMemo(
    () => ({
      status,
      isTranscribing,
      transcribe,
    }),
    [isTranscribing, status, transcribe],
  );
}
