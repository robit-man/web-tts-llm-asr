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

export interface GPUAdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  powerPreference: GPUPowerPreference;
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

  // Initialize useOllama from localStorage
  const [useOllama, setUseOllama] = useState(() => {
    const saved = localStorage.getItem("trifecta_use_ollama");
    return saved === "true";
  });

  // WebGPU state
  const [gpuPowerPreference, setGpuPowerPreference] = useState<GPUPowerPreference>(() => {
    const saved = localStorage.getItem("trifecta_gpu_power_preference");
    return (saved as GPUPowerPreference) || "high-performance";
  });
  const [availableGPUs, setAvailableGPUs] = useState<GPUAdapterInfo[]>([]);
  const [currentGPU, setCurrentGPU] = useState<GPUAdapterInfo | null>(null);
  const [webgpuSupported, setWebgpuSupported] = useState<boolean>(true);

  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const engineRef = useRef<webllm.MLCEngineInterface | null>(null);
  const ollamaFetchedRef = useRef(false);

  // Save Ollama toggle state to localStorage
  useEffect(() => {
    localStorage.setItem("trifecta_use_ollama", String(useOllama));
  }, [useOllama]);

  // Save GPU power preference to localStorage
  useEffect(() => {
    localStorage.setItem("trifecta_gpu_power_preference", gpuPowerPreference);
  }, [gpuPowerPreference]);

  // Detect available GPUs
  const detectGPUs = useCallback(async () => {
    if (!navigator.gpu) {
      setWebgpuSupported(false);
      return [];
    }

    const detectedGPUs: GPUAdapterInfo[] = [];
    const preferences: GPUPowerPreference[] = ["high-performance", "low-power"];

    for (const pref of preferences) {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: pref });
        if (adapter) {
          const info = await (adapter as any).requestAdapterInfo?.();
          const gpuInfo: GPUAdapterInfo = {
            vendor: info?.vendor || "Unknown",
            architecture: info?.architecture || "Unknown",
            device: info?.device || "Unknown",
            description: info?.description || `GPU (${pref})`,
            powerPreference: pref,
          };
          // Avoid duplicates by checking description
          if (!detectedGPUs.some(gpu => gpu.description === gpuInfo.description)) {
            detectedGPUs.push(gpuInfo);
          }
        }
      } catch (error) {
        console.warn(`[WebGPU] Failed to detect ${pref} adapter:`, error);
      }
    }

    setAvailableGPUs(detectedGPUs);
    return detectedGPUs;
  }, []);

  // Fetch Ollama models when Ollama mode is enabled
  useEffect(() => {
    if (!useOllama) {
      ollamaFetchedRef.current = false;
      return;
    }

    // Only fetch if we haven't fetched yet
    if (ollamaFetchedRef.current) return;

    const fetchModels = async () => {
      setStatus({
        model: "webllm",
        label: "Ollama Reasoner",
        state: "loading",
        message: "Fetching Ollama models...",
      });

      const models = await fetchOllamaModels();
      setOllamaModels(models);
      ollamaFetchedRef.current = true;

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
          setWebgpuSupported(false);
          throw new Error(
            "WebGPU is not supported in this browser. Please use Chrome/Edge 113+ with WebGPU enabled."
          );
        }

        // Detect available GPUs
        setStatus({
          model: "webllm",
          label: "WebLLM Reasoner",
          state: "loading",
          message: "Detecting GPUs...",
        });

        const gpus = await detectGPUs();
        console.log("[WebLLM] Detected GPUs:", gpus);

        // Request adapter with selected power preference
        const adapter = await navigator.gpu.requestAdapter({
          powerPreference: gpuPowerPreference
        });

        if (!adapter) {
          throw new Error(
            "WebGPU adapter not found. Ensure WebGPU flags are enabled in chrome://flags"
          );
        }

        // Get and store GPU info
        const adapterInfo = await (adapter as any).requestAdapterInfo?.();
        const gpuInfo: GPUAdapterInfo = {
          vendor: adapterInfo?.vendor || "Unknown",
          architecture: adapterInfo?.architecture || "Unknown",
          device: adapterInfo?.device || "Unknown",
          description: adapterInfo?.description || "Unknown GPU",
          powerPreference: gpuPowerPreference,
        };
        setCurrentGPU(gpuInfo);

        console.log("[WebLLM] Using WebGPU Adapter:", {
          ...gpuInfo,
          powerPreference: gpuPowerPreference,
        });

        setStatus({
          model: "webllm",
          label: "WebLLM Reasoner",
          state: "loading",
          message: `Loading model on ${gpuInfo.description}...`,
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
          setStatus({
            model: "webllm",
            label: "WebLLM Reasoner",
            state: "ready",
            message: `${modelId} ready`,
            detail: `${gpuInfo.description} (${gpuPowerPreference})`,
          });
          console.log(`[WebLLM] Model loaded successfully on ${gpuInfo.description}`);
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
              ? "Check chrome://flags for WebGPU settings or try different GPU"
              : undefined,
          });
          setCurrentGPU(null);
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [modelId, useOllama, gpuPowerPreference, detectGPUs]);

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

  const changeGPUPreference = useCallback((preference: GPUPowerPreference) => {
    setGpuPowerPreference(preference);
    // Clear engine to force reload
    engineRef.current = null;
    const store = getEngineStore();
    store.engine = null;
    store.modelId = null;
    store.promise = null;
  }, []);

  const retryWebGPUInit = useCallback(() => {
    if (useOllama) return;
    // Clear engine to force reload
    engineRef.current = null;
    const store = getEngineStore();
    store.engine = null;
    store.modelId = null;
    store.promise = null;
    // Trigger reload by updating model (even if same)
    setModelId((prev) => prev);
  }, [useOllama]);

  const refreshOllamaModels = useCallback(async () => {
    if (!useOllama) return;

    setStatus({
      model: "webllm",
      label: "Ollama Reasoner",
      state: "loading",
      message: "Refreshing Ollama models...",
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
  }, [useOllama]);

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
      refreshOllamaModels,
      // WebGPU controls
      webgpuSupported,
      gpuPowerPreference,
      changeGPUPreference,
      availableGPUs,
      currentGPU,
      detectGPUs,
      retryWebGPUInit,
    }),
    [
      generate,
      isResponding,
      modelId,
      partialResponse,
      setModel,
      status,
      useOllama,
      toggleOllama,
      ollamaModels,
      refreshOllamaModels,
      webgpuSupported,
      gpuPowerPreference,
      changeGPUPreference,
      availableGPUs,
      currentGPU,
      detectGPUs,
      retryWebGPUInit,
    ],
  );
}
