import type { InferenceSession, Tensor } from "onnxruntime-web";
import { cleanTextForTTS, chunkText } from "../utils/textCleaner";

export class TextSplitterStream implements AsyncIterable<string> {
  private chunks: string[] = [];

  private chunkText(text: string) {
    const cleanedText = cleanTextForTTS(text);
    return chunkText(cleanedText);
  }

  push(text: string) {
    const sentences = this.chunkText(text) || [text];
    this.chunks.push(...sentences);
  }

  close() {}

  async *[Symbol.asyncIterator]() {
    for (const chunk of this.chunks) {
      yield chunk;
    }
  }
}

export class RawAudio {
  public audio: Float32Array<ArrayBufferLike>;
  public sampling_rate: number;

  constructor(
    audio: Float32Array<ArrayBufferLike>,
    samplingRate: number,
  ) {
    this.audio = audio;
    this.sampling_rate = samplingRate;
  }

  get length() {
    return this.audio.length;
  }

  toBlob() {
    const buffer = this.encodeWAV(this.audio, this.sampling_rate);
    return new Blob([buffer], { type: "audio/wav" });
  }

  private encodeWAV(samples: Float32Array, sampleRate: number) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    this.writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    this.writeString(view, 8, "WAVE");
    this.writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    this.writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);
    this.floatTo16BitPCM(view, 44, samples);

    return buffer;
  }

  private writeString(view: DataView, offset: number, string: string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  private floatTo16BitPCM(output: DataView, offset: number, input: Float32Array) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }
}

type VoiceConfig = {
  phoneme_type: string;
  phoneme_id_map: Record<string, number>;
  num_speakers: number;
  speaker_id_map?: Record<string, number>;
  espeak?: { voice?: string };
  audio: { sample_rate: number };
};

type StreamOptions = {
  speakerId?: number;
  lengthScale?: number;
  noiseScale?: number;
  noiseWScale?: number;
};

export class PiperTTS {
  private voiceConfig: VoiceConfig | null;
  private session: InferenceSession | null;

  constructor(
    voiceConfig: VoiceConfig | null = null,
    session: InferenceSession | null = null,
  ) {
    this.voiceConfig = voiceConfig;
    this.session = session;
  }

  static async from_pretrained(modelPath: string, configPath: string) {
    const ort = await import("onnxruntime-web");
    const { cachedFetch } = await import("../utils/modelCache");

    ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}onnx-runtime/`;

    const [modelResponse, configResponse] = await Promise.all([
      cachedFetch(modelPath),
      cachedFetch(configPath),
    ]);

    const [modelBuffer, voiceConfig] = await Promise.all([
      modelResponse.arrayBuffer(),
      configResponse.json(),
    ]);

    const session = await ort.InferenceSession.create(
      new Uint8Array(modelBuffer),
      {
        executionProviders: [
        {
          name: "wasm",
        },
        ],
      },
    );

    return new PiperTTS(voiceConfig as VoiceConfig, session);
  }

  private async textToPhonemes(text: string) {
    if (!this.voiceConfig) {
      return [];
    }

    if (this.voiceConfig.phoneme_type === "text") {
      return [Array.from(text.normalize("NFD"))];
    }

    const { phonemize } = await import("phonemizer");
    const voice = this.voiceConfig.espeak?.voice || "en-us";
    const phonemes = await phonemize(text, voice);

    let phonemeText: string;
    if (typeof phonemes === "string") {
      phonemeText = phonemes;
    } else if (Array.isArray(phonemes)) {
      phonemeText = phonemes.join(" ");
    } else if (phonemes && typeof phonemes === "object") {
      phonemeText =
        (phonemes as { text?: string; phonemes?: string }).text ||
        (phonemes as { phonemes?: string }).phonemes ||
        String(phonemes);
    } else {
      phonemeText = String(phonemes || text);
    }

    const sentences = phonemeText
      .split(/[.!?]+/)
      .filter((s) => s.trim().length > 0);
    return sentences.map((sentence) =>
      Array.from(sentence.trim().normalize("NFD")),
    );
  }

  private phonemesToIds(textPhonemes: string[][]) {
    if (!this.voiceConfig || !this.voiceConfig.phoneme_id_map) {
      throw new Error("Phoneme ID map not available");
    }

    const idMap = this.voiceConfig.phoneme_id_map;
    const BOS = "^";
    const EOS = "$";
    const PAD = "_";

    const phonemeIds: number[] = [];

    for (const sentencePhonemes of textPhonemes) {
      phonemeIds.push(idMap[BOS]);
      phonemeIds.push(idMap[PAD]);

      for (const phoneme of sentencePhonemes) {
        if (phoneme in idMap) {
          phonemeIds.push(idMap[phoneme]);
          phonemeIds.push(idMap[PAD]);
        }
      }
      phonemeIds.push(idMap[EOS]);
    }

    return phonemeIds;
  }

  async *stream(
    textStreamer: TextSplitterStream,
    options: StreamOptions = {},
  ): AsyncGenerator<{ text: string; audio: RawAudio }> {
    if (!this.session || !this.voiceConfig) {
      throw new Error("TTS not initialized");
    }

    const {
      speakerId = 0,
      lengthScale = 1.0,
      noiseScale = 0.667,
      noiseWScale = 0.8,
    } = options;
    const ort = await import("onnxruntime-web");

    for await (const text of textStreamer) {
      if (!text.trim()) continue;

      const textPhonemes = await this.textToPhonemes(text);
      const phonemeIds = this.phonemesToIds(textPhonemes);

      const inputs: Record<string, Tensor> = {
        input: new ort.Tensor(
          "int64",
          new BigInt64Array(phonemeIds.map((id) => BigInt(id))),
          [1, phonemeIds.length],
        ),
        input_lengths: new ort.Tensor(
          "int64",
          BigInt64Array.from([BigInt(phonemeIds.length)]),
          [1],
        ),
        scales: new ort.Tensor(
          "float32",
          Float32Array.from([noiseScale, lengthScale, noiseWScale]),
          [3],
        ),
      };

      if (this.voiceConfig.num_speakers > 1) {
        inputs["sid"] = new ort.Tensor(
          "int64",
          BigInt64Array.from([BigInt(speakerId)]),
          [1],
        );
      }

      const results = await this.session.run(inputs);
      const audioOutput = results.output;
      const audioData = audioOutput.data as Float32Array;
      const sampleRate = this.voiceConfig.audio.sample_rate;
      const finalAudioData = new Float32Array(audioData);

      yield {
        text,
        audio: new RawAudio(finalAudioData, sampleRate),
      };
    }
  }

  getSpeakers() {
    if (!this.voiceConfig || this.voiceConfig.num_speakers <= 1) {
      return [{ id: 0, name: "Voice 1" }];
    }

    const speakerIdMap = this.voiceConfig.speaker_id_map || {};
    return Object.entries(speakerIdMap)
      .sort(([, a], [, b]) => a - b)
      .map(([originalId, id]) => ({
        id,
        name: `Voice ${id + 1}`,
        originalId,
      }));
  }
}
