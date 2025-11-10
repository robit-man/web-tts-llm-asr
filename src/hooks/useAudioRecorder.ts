import { useCallback, useRef, useState } from "react";
import { TARGET_SAMPLE_RATE } from "../utils/audio";

const DEFAULT_EQ_BANDS = 6;
const DEFAULT_SILENCE_MS = 1300;
const DEFAULT_MIN_RECORDING_MS = 600;
const DEFAULT_MAX_RECORDING_MS = 15000;

type FloatAnalyserArray = Float32Array<ArrayBuffer>;
type ByteAnalyserArray = Uint8Array<ArrayBuffer>;

export interface LevelSnapshot {
  rms: number;
  eq: number[];
  isActive: boolean;
}

interface UseAudioRecorderOptions {
  onComplete: (buffer: AudioBuffer) => Promise<void> | void;
  onLevels?: (snapshot: LevelSnapshot) => void;
  eqBands?: number;
  silenceDurationMs?: number;
  minRecordingMs?: number;
  maxRecordingMs?: number;
}

export function useAudioRecorder({
  onComplete,
  onLevels,
  eqBands = DEFAULT_EQ_BANDS,
  silenceDurationMs = DEFAULT_SILENCE_MS,
  minRecordingMs = DEFAULT_MIN_RECORDING_MS,
  maxRecordingMs = DEFAULT_MAX_RECORDING_MS,
}: UseAudioRecorderOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [recorderError, setRecorderError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const analyserDataRef = useRef<FloatAnalyserArray | null>(null);
  const frequencyDataRef = useRef<ByteAnalyserArray | null>(null);
  const monitorRafRef = useRef<number | null>(null);
  const noiseFloorRef = useRef(0.001);
  const recordingStartedAtRef = useRef<number | null>(null);
  const accumulatedSilenceRef = useRef(0);
  const speechActiveRef = useRef(false);
  const silenceTriggeredRef = useRef(false);
  const stopReasonRef = useRef<"manual" | "silence" | "external">("manual");

  const eqStateRef = useRef<number[]>(Array(eqBands).fill(4));

  const resetLevels = useCallback(() => {
    eqStateRef.current = Array(eqBands).fill(4);
    onLevels?.({ rms: 0, eq: eqStateRef.current, isActive: false });
  }, [eqBands, onLevels]);

  const stopMonitor = useCallback(() => {
    if (monitorRafRef.current !== null) {
      cancelAnimationFrame(monitorRafRef.current);
      monitorRafRef.current = null;
    }
  }, []);

  const cleanup = useCallback(async () => {
    stopMonitor();
    if (audioContextRef.current) {
      await audioContextRef.current.close().catch(() => undefined);
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    analyserDataRef.current = null;
    frequencyDataRef.current = null;
    mediaRecorderRef.current = null;
    chunksRef.current = [];
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    resetLevels();
    recordingStartedAtRef.current = null;
    accumulatedSilenceRef.current = 0;
    speechActiveRef.current = false;
    silenceTriggeredRef.current = false;
    stopReasonRef.current = "manual";
  }, [resetLevels, stopMonitor]);

  const updateLevels = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return null;

    if (!analyserDataRef.current || analyserDataRef.current.length !== analyser.fftSize) {
      analyserDataRef.current = new Float32Array(analyser.fftSize) as FloatAnalyserArray;
    }
    if (!frequencyDataRef.current || frequencyDataRef.current.length !== analyser.frequencyBinCount) {
      frequencyDataRef.current = new Uint8Array(analyser.frequencyBinCount) as ByteAnalyserArray;
    }

    const timeBuffer = analyserDataRef.current as Float32Array<ArrayBuffer>;
    const freqBuffer = frequencyDataRef.current as Uint8Array<ArrayBuffer>;
    analyser.getFloatTimeDomainData(timeBuffer);
    analyser.getByteFrequencyData(freqBuffer);

    let sumSquares = 0;
    for (let i = 0; i < timeBuffer.length; i += 1) {
      const sample = timeBuffer[i];
      sumSquares += sample * sample;
    }
    const rms = Math.sqrt(sumSquares / timeBuffer.length);

    // Update adaptive noise floor
    const smoothing = rms > noiseFloorRef.current ? 0.05 : 0.005;
    noiseFloorRef.current =
      noiseFloorRef.current * (1 - smoothing) + rms * smoothing || noiseFloorRef.current;

    const dynamicThreshold = Math.max(noiseFloorRef.current * 1.8, 0.0008);
    const silenceThreshold = Math.max(dynamicThreshold * 0.6, noiseFloorRef.current * 1.2);
    const frameDurationMs = (analyser.fftSize / analyser.context.sampleRate) * 1000;
    const recorderActive = mediaRecorderRef.current?.state === "recording";
    const isActive = recorderActive && (speechActiveRef.current || rms > dynamicThreshold);

    const nyquist = analyser.context.sampleRate / 2;
    const binCount = analyser.frequencyBinCount;
    const nextLevels: number[] = [];
    for (let band = 0; band < eqBands; band += 1) {
      const startHz = (band / eqBands) * nyquist;
      const endHz = ((band + 1) / eqBands) * nyquist;
      const minIndex = Math.max(0, Math.floor((startHz / nyquist) * binCount));
      const maxIndex = Math.min(binCount - 1, Math.floor((endHz / nyquist) * binCount));
      let sum = 0;
      let samples = 0;
      for (let i = minIndex; i <= maxIndex; i += 1) {
        sum += freqBuffer[i];
        samples += 1;
      }
      const average = samples > 0 ? sum / samples : 0;
      const normalized = Math.min(1, average / 255);
      const target = isActive ? 20 + normalized * 80 : 6 + normalized * 24;
      const previous = eqStateRef.current[band] ?? 4;
      const smoothed = previous * 0.65 + target * 0.35;
      nextLevels.push(smoothed);
    }
    eqStateRef.current = nextLevels;
    onLevels?.({ rms, eq: nextLevels, isActive });

    return {
      rms,
      dynamicThreshold,
      silenceThreshold,
      frameDurationMs,
      recorderActive,
    };
  }, [eqBands, onLevels]);

  const startMonitor = useCallback(() => {
    stopMonitor();
    const tick = () => {
      const metrics = updateLevels();
      const now = performance.now();
      const recorderActive = metrics?.recorderActive ?? false;

      if (metrics && recorderActive) {
        const { rms, dynamicThreshold, silenceThreshold, frameDurationMs } = metrics;

        if (rms > dynamicThreshold) {
          speechActiveRef.current = true;
          accumulatedSilenceRef.current = 0;
          noiseFloorRef.current = Math.min(noiseFloorRef.current * 0.9 + rms * 0.1, 0.02);
          silenceTriggeredRef.current = false;
        } else {
          if (!speechActiveRef.current) {
            noiseFloorRef.current = noiseFloorRef.current * 0.92 + rms * 0.08;
          } else {
            accumulatedSilenceRef.current += frameDurationMs;
            const startedAt = recordingStartedAtRef.current ?? now;
            const elapsed = now - startedAt;
            if (
              !silenceTriggeredRef.current &&
              accumulatedSilenceRef.current >= silenceDurationMs &&
              rms < silenceThreshold &&
              elapsed > minRecordingMs
            ) {
              silenceTriggeredRef.current = true;
              speechActiveRef.current = false;
              accumulatedSilenceRef.current = 0;
              stopReasonRef.current = "silence";
              mediaRecorderRef.current?.stop();
              monitorRafRef.current = null;
              return;
            }
          }
        }

        const startedAt = recordingStartedAtRef.current ?? now;
        if (now - startedAt > maxRecordingMs) {
          silenceTriggeredRef.current = true;
          speechActiveRef.current = false;
          stopReasonRef.current = "silence";
          mediaRecorderRef.current?.stop();
          monitorRafRef.current = null;
          return;
        }
      }

      monitorRafRef.current = requestAnimationFrame(tick);
    };
    monitorRafRef.current = requestAnimationFrame(tick);
  }, [maxRecordingMs, minRecordingMs, silenceDurationMs, stopMonitor, updateLevels]);

  const start = useCallback(async () => {
    if (isRecording) return;
    setRecorderError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      const audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.2;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyserRef.current = analyser;
      audioContextRef.current = audioContext;
      recordingStartedAtRef.current = performance.now();
      accumulatedSilenceRef.current = 0;
      speechActiveRef.current = false;
      silenceTriggeredRef.current = false;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        stopMonitor();
        const reason = stopReasonRef.current;
        try {
          if (reason === "silence") {
            if (!chunksRef.current.length) {
              throw new Error("No audio captured.");
            }
            const blob = new Blob(chunksRef.current, { type: "audio/webm" });
            const arrayBuffer = await blob.arrayBuffer();
            const decodeContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
            const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer.slice(0));
            await onComplete(audioBuffer);
            await decodeContext.close();
          }
        } catch (error) {
          if (reason === "silence") {
            setRecorderError(
              (error as Error).message ?? "Could not decode audio data",
            );
          }
        } finally {
          stopReasonRef.current = "manual";
          setIsRecording(false);
          await cleanup();
        }
      };

      recorder.start();
      setIsRecording(true);
      startMonitor();
    } catch (error) {
      setRecorderError(
        (error as Error).message ?? "Microphone permissions denied",
      );
      await cleanup();
    }
  }, [cleanup, isRecording, onComplete, startMonitor, stopMonitor]);

  const stop = useCallback(
    (reason: "manual" | "silence" | "external" = "manual") => {
      if (mediaRecorderRef.current && isRecording) {
        stopReasonRef.current = reason;
        mediaRecorderRef.current.stop();
      }
    },
    [isRecording],
  );

  return {
    isRecording,
    start,
    stop,
    recorderError,
  };
}
