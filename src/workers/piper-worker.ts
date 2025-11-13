/// <reference lib="webworker" />

import { PiperTTS, RawAudio, TextSplitterStream } from "../lib/piper";

type PiperRequest =
  | { type: "init" }
  | { type: "speak"; id: number; text: string; voice?: number; speed?: number }
  | { type: "load_custom_model"; onnxUrl: string; configUrl: string; modelName: string };

const ctx: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;

let tts: PiperTTS | null = null;
let isLoading = false;
let voiceCache: { id: number; name: string; originalId?: number; isCustom?: boolean }[] = [];
let defaultVoices: { id: number; name: string }[] = [];
let customTTS: PiperTTS | null = null;

async function ensureModel() {
  if (tts || isLoading) {
    return;
  }

  isLoading = true;
  ctx.postMessage({
    type: "status",
    model: "piper",
    state: "loading",
    message: "Loading Piper voice...",
  });

  try {
    const modelPath = `${import.meta.env.BASE_URL}tts-model/en_US-libritts_r-medium.onnx`;
    const configPath = `${import.meta.env.BASE_URL}tts-model/en_US-libritts_r-medium.onnx.json`;
    tts = await PiperTTS.from_pretrained(modelPath, configPath);
    defaultVoices = tts.getSpeakers();
    voiceCache = [...defaultVoices];
    ctx.postMessage({
      type: "status",
      model: "piper",
      state: "ready",
      message: "Piper voice ready",
      voices: voiceCache,
    });
  } catch (error) {
    ctx.postMessage({
      type: "status",
      model: "piper",
      state: "error",
      message: (error as Error).message ?? "Unable to load Piper voice",
    });
  } finally {
    isLoading = false;
  }
}

function normalizePeak(f32: Float32Array<ArrayBufferLike>, target = 0.9) {
  if (!f32.length) return;
  let max = 1e-9;
  for (let i = 0; i < f32.length; i++) {
    max = Math.max(max, Math.abs(f32[i]));
  }
  const gain = Math.min(4, target / max);
  if (gain < 1) {
    for (let i = 0; i < f32.length; i++) {
      f32[i] *= gain;
    }
  }
}

function trimSilence(
  f32: Float32Array<ArrayBufferLike>,
  thresh = 0.002,
  minSamples = 480,
): Float32Array<ArrayBufferLike> {
  let start = 0;
  let end = f32.length - 1;

  while (start < end && Math.abs(f32[start]) < thresh) start++;
  while (end > start && Math.abs(f32[end]) < thresh) end--;

  start = Math.max(0, start - minSamples);
  end = Math.min(f32.length, end + minSamples);

  return f32.slice(start, end);
}

function mergeChunks(chunks: RawAudio[]) {
  if (!chunks.length) {
    return null;
  }

  const sampleRate = chunks[0].sampling_rate;
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.audio.length, 0);
  const waveform = new Float32Array(totalLength);
  let offset = 0;

  for (const { audio } of chunks) {
    waveform.set(audio, offset);
    offset += audio.length;
  }

  normalizePeak(waveform, 0.9);
  const trimmed = trimSilence(
    waveform,
    0.002,
    Math.floor(sampleRate * 0.02),
  );

  return new RawAudio(trimmed, sampleRate);
}

ctx.addEventListener("message", async (event: MessageEvent<PiperRequest>) => {
  const data = event.data;

  if (data.type === "init") {
    await ensureModel();
    if (tts && voiceCache.length === 0) {
      voiceCache = tts.getSpeakers();
      ctx.postMessage({
        type: "status",
        model: "piper",
        state: "ready",
        message: "Piper voice ready",
        voices: voiceCache,
      });
    }
    return;
  }

  if (!tts) {
    await ensureModel();
  }

  if (data.type === "load_custom_model") {
    try {
      isLoading = true;
      ctx.postMessage({
        type: "status",
        model: "piper",
        state: "loading",
        message: "Loading custom model...",
      });

      // Load custom model separately
      customTTS = await PiperTTS.from_pretrained(data.onnxUrl, data.configUrl);
      const customVoices = customTTS.getSpeakers();

      // Get the current max ID from voiceCache
      const maxId = voiceCache.length > 0 ? Math.max(...voiceCache.map(v => v.id)) : -1;

      // Add only the first speaker from the custom model with the filename as the name
      const newVoice = {
        id: maxId + 1,
        name: data.modelName,
        originalId: customVoices[0].id,
        isCustom: true,
      };

      // Append to existing voiceCache
      voiceCache = [...voiceCache, newVoice];

      ctx.postMessage({
        type: "custom_model_loaded",
        voices: voiceCache,
      });
    } catch (error) {
      ctx.postMessage({
        type: "error",
        model: "piper",
        message: (error as Error).message ?? "Unable to load custom model",
      });
    } finally {
      isLoading = false;
    }
    return;
  }

  if (!tts) {
    ctx.postMessage({
      type: "status",
      model: "piper",
      state: "error",
      message: "Piper model not ready",
    });
    return;
  }

  if (data.type === "speak") {
    const { text, voice = 0, speed = 1, id } = data;
    const streamer = new TextSplitterStream();
    streamer.push(text);
    streamer.close();

    const chunks: RawAudio[] = [];
    try {
      // Find the voice in cache to determine if it's custom
      const voiceInfo = voiceCache.find(v => v.id === voice);
      const isCustomVoice = voiceInfo?.isCustom || false;
      const speakerId = isCustomVoice && voiceInfo?.originalId !== undefined
        ? voiceInfo.originalId
        : voice;

      // Use the appropriate TTS model
      const activeTTS = isCustomVoice && customTTS ? customTTS : tts;

      if (!activeTTS) {
        throw new Error("TTS model not ready");
      }

      const stream = activeTTS.stream(streamer, {
        speakerId: speakerId,
        lengthScale: 1 / Math.max(speed, 0.5),
      });

      for await (const entry of stream) {
        chunks.push(entry.audio);
      }

      const merged = mergeChunks(chunks);
      if (!merged) {
        throw new Error("Unable to render speech");
      }

      ctx.postMessage(
        {
          type: "speech",
          id,
          audio: merged.toBlob(),
        },
        [],
      );
    } catch (error) {
      ctx.postMessage({
        type: "error",
        id,
        model: "piper",
        message: (error as Error).message ?? "Unable to generate speech",
      });
    }
  }
});
