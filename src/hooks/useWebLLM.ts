import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as webllm from "@mlc-ai/web-llm";
import type { ModelStatus } from "../types/models";

// WebGPU type declarations
interface GPU {
  requestAdapter(): Promise<GPUAdapter | null>;
}

interface GPUAdapter {
  requestDevice(): Promise<unknown>;
}

declare global {
  interface Navigator {
    gpu?: GPU;
  }
}

const DEFAULT_MODEL = "Llama-3.2-1B-Instruct-q4f32_1-MLC";
const ENGINE_CACHE_KEY = "__trifecta_webllm_engine__";
const OLLAMA_BASE_URL = "http://localhost:11434";

export interface OllamaModel {
  name: string;
  size: number;
  modified_at: string;
}

type EngineStore = {
  engine: webllm.MLCEngineInterface | null;
  promise: Promise<webllm.MLCEngineInterface> | null;
  modelId: string | null;
};

function getEngineStore(): EngineStore {
  const globalScope = globalThis as typeof globalThis & {
    [ENGINE_CACHE_KEY]?: EngineStore;
  };
  if (!globalScope[ENGINE_CACHE_KEY]) {
    globalScope[ENGINE_CACHE_KEY] = {
      engine: null,
      promise: null,
      modelId: null,
    };
  }
  return globalScope[ENGINE_CACHE_KEY]!;
}

async function obtainEngine(
  modelId: string,
  progress: (report: webllm.InitProgressReport) => void,
) {
  const store = getEngineStore();

  if (store.engine && store.modelId === modelId) {
    return store.engine;
  }

  if (store.promise) {
    return store.promise;
  }

  const promise = webllm
    .CreateMLCEngine(
      modelId,
      {
        initProgressCallback: progress,
        logLevel: "WARN",
      },
      {
        context_window_size: 2048,
      },
    )
    .then((engine) => {
      const cache = getEngineStore();
      cache.engine = engine;
      cache.modelId = modelId;
      cache.promise = null;
      return engine;
    })
    .catch((error) => {
      const cache = getEngineStore();
      cache.promise = null;
      cache.engine = null;
      cache.modelId = null;
      throw error;
    });

  store.promise = promise;
  return promise;
}

async function fetchOllamaModels(): Promise<OllamaModel[]> {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }
    const data = await response.json();
    return data.models || [];
  } catch (error) {
    console.error("[Ollama] Failed to fetch models:", error);
    return [];
  }
}

export function useWebLLM(initialModel = DEFAULT_MODEL) {
  const [status, setStatus] = useState<ModelStatus>({
    model: "webllm",
    label: "WebLLM Reasoner",
    state: "loading",
    message: "Booting WebGPU runtime...",
  });
  const [isResponding, setIsResponding] = useState(false);
  const [partialResponse, setPartialResponse] = useState("");
  const [modelId, setModelId] = useState(initialModel);
  const [useOllama, setUseOllama] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const engineRef = useRef<webllm.MLCEngineInterface | null>(null);

  // Fetch Ollama models when Ollama mode is enabled
  useEffect(() => {
    if (!useOllama) return;

    const fetchModels = async () => {
      setStatus({
        model: "webllm",
        label: "Ollama Reasoner",
        state: "loading",
        message: "Fetching Ollama models...",
      });

      const models = await fetchOllamaModels();
      setOllamaModels(models);

      if (models.length === 0) {
        setStatus({
          model: "webllm",
          label: "Ollama Reasoner",
          state: "error",
          message: "No Ollama models found",
          detail: "Ensure Ollama is running at http://localhost:11434",
        });
      } else {
        setStatus({
          model: "webllm",
          label: "Ollama Reasoner",
          state: "ready",
          message: `${models.length} model${models.length > 1 ? 's' : ''} available`,
          detail: `Connected to Ollama at localhost:11434`,
        });
      }
    };

    fetchModels();
  }, [useOllama]);

  useEffect(() => {
    // Skip WebLLM loading if using Ollama
    if (useOllama) return;

    let cancelled = false;

    const load = async () => {
      try {
        // Check WebGPU support before attempting to load
        if (!navigator.gpu) {
          throw new Error(
            "WebGPU is not supported in this browser. Please use Chrome/Edge 113+ with WebGPU enabled."
          );
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          throw new Error(
            "WebGPU adapter not found. Ensure WebGPU flags are enabled in chrome://flags"
          );
        }

        // Log GPU info for diagnostics
        const adapterInfo = await (adapter as any).requestAdapterInfo?.();
        console.log("[WebLLM] WebGPU Adapter Info:", {
          vendor: adapterInfo?.vendor || "unknown",
          architecture: adapterInfo?.architecture || "unknown",
          device: adapterInfo?.device || "unknown",
          description: adapterInfo?.description || "unknown",
        });

        const engine = await obtainEngine(modelId, (report) => {
          if (cancelled) return;
          setStatus({
            model: "webllm",
            label: "WebLLM Reasoner",
            state: "loading",
            message: report.text,
            progress: report.progress,
          });
        });

        if (!cancelled) {
          engineRef.current = engine;
          const deviceInfo = adapterInfo?.description || "GPU";
          setStatus({
            model: "webllm",
            label: "WebLLM Reasoner",
            state: "ready",
            message: `${modelId} ready`,
            detail: `Using WebGPU on ${deviceInfo}`,
          });
          console.log(`[WebLLM] Model loaded successfully on ${deviceInfo}`);
        }
      } catch (error) {
        if (!cancelled) {
          const errorMessage = (error as Error).message;
          setStatus({
            model: "webllm",
            label: "WebLLM Reasoner",
            state: "error",
            message: errorMessage ?? "Unable to initialize WebLLM runtime",
            detail: errorMessage.includes("WebGPU")
              ? "Check chrome://flags for WebGPU settings"
              : undefined,
          });
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [modelId, useOllama]);

  const generate = useCallback(
    async (messages: webllm.ChatCompletionMessageParam[]) => {
      setIsResponding(true);
      setPartialResponse("");

      try {
        if (useOllama) {
          // Use Ollama endpoint (OpenAI-compatible API)
          const response = await fetch(`${OLLAMA_BASE_URL}/v1/chat/completions`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              model: modelId,
              messages,
              temperature: 0.7,
              max_tokens: 1024,
              stream: true,
            }),
          });

          if (!response.ok) {
            throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error("Failed to get response reader");
          }

          const decoder = new TextDecoder();
          let fullText = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n").filter((line) => line.trim().startsWith("data: "));

            for (const line of lines) {
              const data = line.replace(/^data: /, "");
              if (data === "[DONE]") break;

              try {
                const parsed = JSON.parse(data);
                const delta = parsed.choices?.[0]?.delta?.content ?? "";
                if (delta) {
                  fullText += delta;
                  setPartialResponse(fullText);
                }
              } catch {
                // Skip invalid JSON lines
              }
            }
          }

          return fullText.trim();
        } else {
          // Use WebLLM
          if (!engineRef.current) {
            throw new Error("WebLLM runtime is not ready yet");
          }

          const completion = await engineRef.current.chat.completions.create({
            messages,
            temperature: 0.7,
            max_tokens: 1024,
            stream: true,
          });

          let fullText = "";
          for await (const chunk of completion) {
            const delta = chunk.choices?.[0]?.delta?.content ?? "";
            if (delta) {
              fullText += delta;
              setPartialResponse(fullText);
            }
          }
          return fullText.trim();
        }
      } finally {
        setIsResponding(false);
        setPartialResponse("");
      }
    },
    [useOllama, modelId],
  );

  const setModel = useCallback((newModelId: string) => {
    if (newModelId === modelId) return;
    setModelId(newModelId);
    engineRef.current = null;
    const store = getEngineStore();
    store.engine = null;
    store.modelId = null;
    store.promise = null;
  }, [modelId]);

  const toggleOllama = useCallback((enabled: boolean) => {
    setUseOllama(enabled);
    if (!enabled) {
      // Clear Ollama models when switching back to WebLLM
      setOllamaModels([]);
    }
  }, []);

  return useMemo(
    () => ({
      status,
      generate,
      isResponding,
      partialResponse,
      model: modelId,
      setModel,
      useOllama,
      toggleOllama,
      ollamaModels,
    }),
    [generate, isResponding, modelId, partialResponse, setModel, status, useOllama, toggleOllama, ollamaModels],
  );
}
