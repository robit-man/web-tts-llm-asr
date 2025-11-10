import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ModelStatus } from "../types/models";

type PiperWorkerMessage =
  | {
      type: "status";
      model: "piper";
      state: ModelStatus["state"];
      message?: string;
      voices?: { id: number; name: string }[];
    }
  | {
    type: "speech";
    id: number;
    audio: Blob;
  }
  | {
    type: "error";
    id?: number;
    model: "piper";
    message: string;
  };

export function usePiperModel() {
  const [status, setStatus] = useState<ModelStatus>({
    model: "piper",
    label: "Piper TTS",
    state: "loading",
    message: "Starting Piper worker...",
  });
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState<{ id: number; name: string }[]>([]);
  const [voiceId, setVoiceId] = useState<number>(0);
  const workerRef = useRef<Worker | null>(null);
  const requestId = useRef(0);
  const pending = useRef<
    Map<number, { resolve: (value: string) => void; reject: (reason?: unknown) => void }>
  >(new Map());

  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/piper-worker.ts", import.meta.url),
      { type: "module" },
    );
    workerRef.current = worker;
    worker.postMessage({ type: "init" });

    const handleMessage = (event: MessageEvent<PiperWorkerMessage>) => {
      const message = event.data;

      if (message.type === "status" && message.model === "piper") {
        setStatus((prev) => ({
          ...prev,
          state: message.state,
          message: message.message,
        }));
        if (message.voices && message.voices.length) {
          setVoices(message.voices);
        }
        return;
      }

      if (message.type === "speech") {
        const handlers = pending.current.get(message.id);
        if (handlers) {
          const url = URL.createObjectURL(message.audio);
          handlers.resolve(url);
          pending.current.delete(message.id);
        }
        if (pending.current.size === 0) {
          setIsSpeaking(false);
        }
        return;
      }

      if (message.type === "error" && message.model === "piper") {
        if (message.id !== undefined) {
          const handlers = pending.current.get(message.id);
          if (handlers) {
            handlers.reject(new Error(message.message));
            pending.current.delete(message.id);
          }
          if (pending.current.size === 0) {
            setIsSpeaking(false);
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

  useEffect(() => {
    if (voices.length > 0) {
      if (!voices.some((voice) => voice.id === voiceId)) {
        setVoiceId(voices[0].id);
      }
    }
  }, [voiceId, voices]);

  const speak = useCallback(
    async (text: string) => {
      if (!workerRef.current) {
        throw new Error("Piper worker not ready");
      }
      const id = requestId.current++;
      setIsSpeaking(true);

      return await new Promise<string>((resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        workerRef.current?.postMessage({ type: "speak", id, text, voice: voiceId });
      });
    },
    [voiceId],
  );

  return useMemo(
    () => ({
      status,
      speak,
      isSpeaking,
      voices,
      voiceId,
      setVoiceId,
    }),
    [isSpeaking, speak, status, voiceId, voices],
  );
}
