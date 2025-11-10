import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as webllm from "@mlc-ai/web-llm";
import type { ModelStatus } from "../types/models";

const DEFAULT_MODEL = "Llama-3.2-1B-Instruct-q4f32_1-MLC";

export function useWebLLM(modelId = DEFAULT_MODEL) {
  const [status, setStatus] = useState<ModelStatus>({
    model: "webllm",
    label: "WebLLM Reasoner",
    state: "loading",
    message: "Booting WebGPU runtime...",
  });
  const [isResponding, setIsResponding] = useState(false);
  const engineRef = useRef<webllm.MLCEngineInterface | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const engine = await webllm.CreateMLCEngine(
          modelId,
          {
            initProgressCallback: (report) => {
              if (cancelled) return;
              setStatus({
                model: "webllm",
                label: "WebLLM Reasoner",
                state: "loading",
                message: report.text,
                progress: report.progress,
              });
            },
            logLevel: "WARN",
          },
          {
            context_window_size: 2048,
          },
        );

        if (!cancelled) {
          engineRef.current = engine;
          setStatus({
            model: "webllm",
            label: "WebLLM Reasoner",
            state: "ready",
            message: `${modelId} ready`,
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatus({
            model: "webllm",
            label: "WebLLM Reasoner",
            state: "error",
            message:
              (error as Error).message ?? "Unable to initialize WebLLM runtime",
          });
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  const generate = useCallback(
    async (messages: webllm.ChatCompletionMessageParam[]) => {
      if (!engineRef.current) {
        throw new Error("WebLLM runtime is not ready yet");
      }

      setIsResponding(true);
      try {
        const reply = await engineRef.current.chat.completions.create({
          messages,
          temperature: 0.7,
          max_tokens: 256,
        });
        const text = reply.choices?.[0]?.message?.content ?? "";
        return text.trim();
      } finally {
        setIsResponding(false);
      }
    },
    [],
  );

  return useMemo(
    () => ({
      status,
      generate,
      isResponding,
    }),
    [generate, isResponding, status],
  );
}
