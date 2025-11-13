import { useCallback, useEffect, useRef, useState } from "react";

export interface LevelSnapshot {
  rms: number;
  eq: number[];
  isActive: boolean;
}

interface UseAudioRecorderOptions {
  onRecordingComplete: (blob: Blob | null) => void;
  onLevels?: (snapshot: LevelSnapshot) => void;
  onAudioChunk?: (chunk: Float32Array) => void;
  eqBands?: number;
  enableVAD?: boolean;
  minRecordingMs?: number;
  maxRecordingMs?: number;
  silenceDurationMs?: number;
}

type StopReason = 'manual' | 'silence' | 'max_duration';

const DEFAULT_EQ_BANDS = 6;
const DEFAULT_MIN_RECORDING_MS = 450;
const DEFAULT_MAX_RECORDING_MS = 15000;
const DEFAULT_SILENCE_DURATION_MS = 900; // Reduced from 1200ms - faster cutoff with better detection
const MIN_FREQUENCY = 120;
const MAX_FREQUENCY = 5200;
const EQ_DECAY = 0.6;
const EQ_THRESHOLD = 0.08;

const SUPPORTED_MIME_TYPES = ['audio/webm', 'audio/mp4', 'audio/ogg', 'audio/wav', 'audio/aac'];

function getSupportedMimeType(): string | undefined {
  for (const type of SUPPORTED_MIME_TYPES) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return undefined;
}

export function useAudioRecorder({
  onRecordingComplete,
  onLevels,
  onAudioChunk,
  eqBands = DEFAULT_EQ_BANDS,
  enableVAD = true,
  minRecordingMs = DEFAULT_MIN_RECORDING_MS,
  maxRecordingMs = DEFAULT_MAX_RECORDING_MS,
  silenceDurationMs = DEFAULT_SILENCE_DURATION_MS,
}: UseAudioRecorderOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [recorderError, setRecorderError] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const analyserSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const monitorRafRef = useRef<number | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);

  const noiseFloorRef = useRef(0.0004); // Lower initial threshold for quieter speech
  const backgroundNoiseRef = useRef(0.0004); // Track background noise separately
  const smoothedEnergyRef = useRef(0); // Smoothed energy for stable detection
  const recordingStartedAtRef = useRef<number>(0);
  const lastSpeechTimeRef = useRef<number>(0);
  const speechActiveRef = useRef(false);
  const silenceTriggeredRef = useRef(false);
  const accumulatedSilenceRef = useRef(0);
  const speechFramesRef = useRef(0); // Count consecutive speech frames
  const eqUpdateTimeRef = useRef(0);
  const eqLevelsRef = useRef<number[]>(Array(eqBands).fill(6));
  const stopReasonRef = useRef<StopReason>('manual');

  const stopMonitor = useCallback(() => {
    if (monitorRafRef.current !== null) {
      console.log('[Monitor] ⏹️  Stopping monitoring loop, RAF ID:', monitorRafRef.current);
      console.trace('[Monitor] stopMonitor called from:');
      cancelAnimationFrame(monitorRafRef.current);
      monitorRafRef.current = null;
    }
  }, []);

  const releaseAudioResources = useCallback(() => {
    stopMonitor();
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    analyserSourceRef.current?.disconnect();
    analyserSourceRef.current = null;
    analyserRef.current = null;
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => undefined);
      audioContextRef.current = null;
    }
  }, [stopMonitor]);

  const cleanup = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    releaseAudioResources();
    recorderRef.current = null;
    chunksRef.current = [];
    audioChunksRef.current = [];
    speechActiveRef.current = false;
    silenceTriggeredRef.current = false;
    noiseFloorRef.current = 0.0004; // Lower threshold for quieter speech
    backgroundNoiseRef.current = 0.0004;
    smoothedEnergyRef.current = 0;
    speechFramesRef.current = 0;
    stopReasonRef.current = 'manual';
  }, [releaseAudioResources]);

  const initializeAnalyser = useCallback(
    async (stream: MediaStream) => {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
        console.log('[Analyser] Created new AudioContext');
      }

      const ctx = audioContextRef.current;
      console.log('[Analyser] AudioContext state before resume:', ctx.state);

      // Ensure AudioContext is running
      if (ctx.state === 'suspended' || ctx.state === 'closed') {
        try {
          await ctx.resume();
          console.log('[Analyser] AudioContext resumed, new state:', ctx.state);
        } catch (error) {
          console.error('[Analyser] Failed to resume AudioContext:', error);
          // If resume fails, recreate the AudioContext
          audioContextRef.current = new AudioContext();
          console.log('[Analyser] Created new AudioContext after resume failure');
        }
      }

      // Verify context is now running
      if (audioContextRef.current.state !== 'running') {
        console.warn('[Analyser] AudioContext not running, attempting to start:', audioContextRef.current.state);
        try {
          await audioContextRef.current.resume();
        } catch (error) {
          console.error('[Analyser] Still failed to resume AudioContext:', error);
        }
      }

      analyserSourceRef.current?.disconnect();
      if (processorRef.current) {
        processorRef.current.disconnect();
      }

      const source = audioContextRef.current.createMediaStreamSource(stream);
      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.4;
      analyser.minDecibels = -90;
      analyser.maxDecibels = -10;
      source.connect(analyser);

      console.log('[Analyser] Created new analyser, AudioContext state:', audioContextRef.current.state);

      // Set up ScriptProcessorNode for raw audio capture if callback is provided
      if (onAudioChunk) {
        const bufferSize = 4096;
        const processor = audioContextRef.current.createScriptProcessor(bufferSize, 1, 1);

        processor.onaudioprocess = (event) => {
          const inputData = event.inputBuffer.getChannelData(0);
          const sampleRate = event.inputBuffer.sampleRate;

          // Resample to 16kHz if necessary
          if (sampleRate === 16000) {
            onAudioChunk(new Float32Array(inputData));
          } else {
            const targetSampleRate = 16000;
            const ratio = sampleRate / targetSampleRate;
            const newLength = Math.floor(inputData.length / ratio);
            const resampled = new Float32Array(newLength);

            for (let i = 0; i < newLength; i++) {
              const srcIndex = i * ratio;
              const srcIndexFloor = Math.floor(srcIndex);
              const srcIndexCeil = Math.min(srcIndexFloor + 1, inputData.length - 1);
              const fraction = srcIndex - srcIndexFloor;
              resampled[i] = inputData[srcIndexFloor] * (1 - fraction) + inputData[srcIndexCeil] * fraction;
            }

            onAudioChunk(resampled);
          }
        };

        source.connect(processor);
        processor.connect(audioContextRef.current.destination);
        processorRef.current = processor;
      }

      analyserRef.current = analyser;
      analyserSourceRef.current = source;
    },
    [onAudioChunk],
  );

  const stop = useCallback((reason: StopReason = 'manual') => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state !== 'recording') {
      console.log('[Recorder] stop() called but not recording:', {
        hasRecorder: !!recorder,
        state: recorder?.state,
        reason
      });
      return;
    }

    console.log('[Recorder] Stopping recording:', {
      reason,
      chunksBeforeStop: chunksRef.current.length,
      recorderState: recorder.state
    });

    stopReasonRef.current = reason;
    speechActiveRef.current = false;
    silenceTriggeredRef.current = false;

    // Don't kill monitoring or state - let the 'stop' event handler manage it
    recorder.stop();
  }, []);

  const startMonitoringLevels = useCallback(() => {
    // Monitor continuously while stream exists, not just while recording (like Cygnus)
    if (!analyserRef.current) {
      console.error('[Monitoring] Cannot start - no analyser!');
      stopMonitor();
      return;
    }

    console.log('[Monitoring] Starting monitoring loop', {
      hasAnalyser: !!analyserRef.current,
      hasAudioContext: !!audioContextRef.current,
      audioContextState: audioContextRef.current?.state,
      streamActive: streamRef.current?.active,
    });

    const analyser = analyserRef.current;
    const timeBuffer = new Float32Array(analyser.fftSize);
    const frequencyBuffer = new Uint8Array(analyser.frequencyBinCount);
    let lastHealthCheck = performance.now();
    let loopCount = 0;

    const detect = () => {
      // Wrap entire detect in try-catch to prevent loop from dying
      try {
        loopCount++;

        // Log every 60 frames (~1 second) to prove loop is running
        if (loopCount % 60 === 0) {
          console.log(`[VAD] Loop active: ${loopCount} iterations, RAF ID:`, monitorRafRef.current);
        }

        if (!analyserRef.current) {
          console.error('[VAD] ❌ analyserRef became null at iteration', loopCount);
          monitorRafRef.current = null;
          return;
        }

        // Keep AudioContext alive - resume if suspended
        if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
          console.warn('[VAD] AudioContext suspended during monitoring, resuming...');
          audioContextRef.current.resume().catch((error) => {
            console.error('[VAD] Failed to resume AudioContext:', error);
          });
        }

        analyser.getFloatTimeDomainData(timeBuffer);
        analyser.getByteFrequencyData(frequencyBuffer);

      let sumSquares = 0;
      for (let i = 0; i < timeBuffer.length; i += 1) {
        const sample = timeBuffer[i];
        sumSquares += sample * sample;
      }
      const rms = Math.sqrt(sumSquares / timeBuffer.length);
      const now = performance.now();

      // Periodic health check (every 5 seconds)
      if (now - lastHealthCheck > 5000) {
        lastHealthCheck = now;
        console.log('[VAD] Health check:', {
          audioContextState: audioContextRef.current?.state,
          analyserExists: !!analyserRef.current,
          streamActive: streamRef.current?.active,
          isRecording: recorderRef.current?.state === 'recording',
          monitoringActive: monitorRafRef.current !== null,
          rms: rms.toFixed(4),
        });
      }

      // Debug: Check if we're getting actual audio data
      if (recorderRef.current?.state === 'recording' && rms === 0 && now - recordingStartedAtRef.current > 1000) {
        const sampleSum = timeBuffer.reduce((sum, val) => sum + Math.abs(val), 0);
        if (sampleSum === 0) {
          console.error('[VAD] No audio data detected! AudioContext:', {
            state: audioContextRef.current?.state,
            sampleRate: analyser.context.sampleRate,
            streamActive: streamRef.current?.active,
            streamTracks: streamRef.current?.getAudioTracks().map(t => ({ id: t.id, enabled: t.enabled, muted: t.muted, readyState: t.readyState }))
          });
        }
      }

      // Calculate spectral energy in speech frequency range (300-3400Hz)
      const nyquist = analyser.context.sampleRate / 2;
      const binCount = analyser.frequencyBinCount;
      const speechMinHz = 300;
      const speechMaxHz = 3400;
      const speechMinBin = Math.floor((speechMinHz / nyquist) * binCount);
      const speechMaxBin = Math.ceil((speechMaxHz / nyquist) * binCount);

      let speechEnergy = 0;
      let totalEnergy = 0;
      for (let i = 0; i < binCount; i++) {
        const magnitude = frequencyBuffer[i] / 255; // Normalize to 0-1
        totalEnergy += magnitude;
        if (i >= speechMinBin && i <= speechMaxBin) {
          speechEnergy += magnitude;
        }
      }
      const speechRatio = totalEnergy > 0 ? speechEnergy / totalEnergy : 0;

      // Calculate zero-crossing rate (helps distinguish speech from noise)
      let zeroCrossings = 0;
      for (let i = 1; i < timeBuffer.length; i++) {
        if ((timeBuffer[i] >= 0 && timeBuffer[i - 1] < 0) ||
            (timeBuffer[i] < 0 && timeBuffer[i - 1] >= 0)) {
          zeroCrossings++;
        }
      }
      const zcr = zeroCrossings / timeBuffer.length;

      // Smooth energy using exponential moving average
      const energyAlpha = 0.3;
      smoothedEnergyRef.current = energyAlpha * rms + (1 - energyAlpha) * smoothedEnergyRef.current;

      // Adaptive thresholds based on background noise and speech characteristics
      const dynamicThreshold = Math.max(backgroundNoiseRef.current * 2.0, 0.0004);
      const silenceThreshold = Math.max(backgroundNoiseRef.current * 1.3, 0.0003);

      // Speech detection requires: high energy + speech-like spectrum + appropriate ZCR
      const hasEnergy = smoothedEnergyRef.current > dynamicThreshold;
      const hasSpeechSpectrum = speechRatio > 0.25; // At least 25% energy in speech range
      const hasReasonableZCR = zcr > 0.02 && zcr < 0.4; // Typical speech range
      const isSpeechLike = hasEnergy && (hasSpeechSpectrum || hasReasonableZCR);

      const frameDurationMs = (analyser.fftSize / analyser.context.sampleRate) * 1000;
      const isRecorderRecording = recorderRef.current?.state === 'recording';
      const isCurrentlyActive = isRecorderRecording && (speechActiveRef.current || isSpeechLike);

      // Update EQ visualization
      if (now - eqUpdateTimeRef.current > 90) {
        eqUpdateTimeRef.current = now;
        if (isRecorderRecording) {
          const nyquist = analyser.context.sampleRate / 2;
          const binCount = analyser.frequencyBinCount;
          let peakNormalized = 0;
          const previousLevels = eqLevelsRef.current;
          const nextLevels = Array.from({ length: eqBands }, (_, index) => {
            const bandStartHz = MIN_FREQUENCY + ((MAX_FREQUENCY - MIN_FREQUENCY) / eqBands) * index;
            const bandEndHz = MIN_FREQUENCY + ((MAX_FREQUENCY - MIN_FREQUENCY) / eqBands) * (index + 1);
            const minIndex = Math.max(0, Math.floor((bandStartHz / nyquist) * binCount));
            const maxIndex = Math.min(binCount - 1, Math.floor((bandEndHz / nyquist) * binCount));
            let sum = 0;
            let samples = 0;
            for (let i = minIndex; i <= maxIndex; i += 1) {
              const magnitude = frequencyBuffer[i];
              if (Number.isFinite(magnitude)) {
                sum += magnitude;
                samples += 1;
              }
            }
            const avgMagnitude = samples > 0 ? sum / samples : 0;
            const normalized = Math.min(1, avgMagnitude / 255);
            peakNormalized = Math.max(peakNormalized, normalized);
            const targetHeight = isCurrentlyActive ? 18 + normalized * 90 : 6 + normalized * 28;
            const previousHeight = previousLevels[index] ?? 6;
            const smoothed = previousHeight * EQ_DECAY + targetHeight * (1 - EQ_DECAY);
            return Math.max(6, smoothed);
          });

          eqLevelsRef.current = nextLevels;
          onLevels?.({
            rms,
            eq: nextLevels,
            isActive: peakNormalized > EQ_THRESHOLD,
          });
        } else {
          eqLevelsRef.current = eqLevelsRef.current.map(() => 4);
          onLevels?.({
            rms,
            eq: eqLevelsRef.current,
            isActive: false,
          });
        }
      }

      // VAD logic with hysteresis and improved noise adaptation
      if (enableVAD) {
        if (isSpeechLike) {
          // Speech detected - increment speech frame counter
          speechFramesRef.current++;

          // Require 3 consecutive speech frames to activate (prevents false triggers)
          if (speechFramesRef.current >= 3 || speechActiveRef.current) {
            if (!speechActiveRef.current) {
              console.log('[VAD] Speech onset detected:', {
                energy: smoothedEnergyRef.current.toFixed(4),
                threshold: dynamicThreshold.toFixed(4),
                speechRatio: speechRatio.toFixed(3),
                zcr: zcr.toFixed(3),
              });
            }
            speechActiveRef.current = true;
            lastSpeechTimeRef.current = now;
            accumulatedSilenceRef.current = 0;
            silenceTriggeredRef.current = false;

            // Adapt noise floor more slowly during speech
            backgroundNoiseRef.current = Math.min(
              backgroundNoiseRef.current * 0.98 + smoothedEnergyRef.current * 0.02,
              0.015
            );
          }
        } else {
          // No speech detected
          speechFramesRef.current = 0;

          // Update background noise estimate when not speaking
          if (!speechActiveRef.current) {
            backgroundNoiseRef.current = backgroundNoiseRef.current * 0.95 + smoothedEnergyRef.current * 0.05;
            // Cap background noise to prevent runaway adaptation
            backgroundNoiseRef.current = Math.min(backgroundNoiseRef.current, 0.01);
          }

          // Check for silence during active speech
          if (speechActiveRef.current) {
            accumulatedSilenceRef.current += frameDurationMs;
            const recordingElapsed = now - recordingStartedAtRef.current;

            // More sophisticated silence check: low energy AND low speech characteristics
            const isTrueSilence =
              smoothedEnergyRef.current < silenceThreshold &&
              speechRatio < 0.15; // Very little speech-range energy

            if (
              !silenceTriggeredRef.current &&
              accumulatedSilenceRef.current >= silenceDurationMs &&
              isTrueSilence &&
              recordingElapsed > minRecordingMs &&
              recorderRef.current?.state === 'recording'
            ) {
              console.log('[VAD] Silence detected:', {
                energy: smoothedEnergyRef.current.toFixed(4),
                threshold: silenceThreshold.toFixed(4),
                speechRatio: speechRatio.toFixed(3),
                silenceDuration: accumulatedSilenceRef.current.toFixed(0),
              });
              silenceTriggeredRef.current = true;
              speechActiveRef.current = false;
              accumulatedSilenceRef.current = 0;
              speechFramesRef.current = 0;
              stop('silence');
            }
          }
        }

        // Max duration check
        if (
          recorderRef.current?.state === 'recording' &&
          recordingStartedAtRef.current &&
          now - recordingStartedAtRef.current > maxRecordingMs
        ) {
          stop('max_duration');
        }
      }
      } catch (error) {
        console.error('[VAD] Error in detect loop:', error);
        // Don't let the loop die - continue monitoring
      }

      monitorRafRef.current = requestAnimationFrame(detect);
    };

    stopMonitor();
    monitorRafRef.current = requestAnimationFrame(detect);
  }, [eqBands, enableVAD, maxRecordingMs, minRecordingMs, onLevels, silenceDurationMs, stop, stopMonitor]);

  const start = useCallback(async () => {
    console.log('[useAudioRecorder.start] 🎤 start() called');
    setRecorderError(null);

    try {
      const mimeType = getSupportedMimeType();
      if (!mimeType) {
        throw new Error('This browser does not support any required audio codecs.');
      }

      if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
        throw new Error('This browser does not support audio recording.');
      }

      if (!streamRef.current) {
        console.log('[useAudioRecorder.start] Requesting microphone access...');
        try {
          streamRef.current = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true }
          });
          console.log('[useAudioRecorder.start] ✅ Got new MediaStream:', {
            id: streamRef.current.id,
            active: streamRef.current.active,
            tracks: streamRef.current.getAudioTracks().length
          });
        } catch (error) {
          if ((error as DOMException).name === 'NotAllowedError') {
            throw new Error('Microphone access denied. Please enable and try again.');
          }
          throw new Error('Unable to access microphone. Check device settings.');
        }
      } else {
        console.log('[useAudioRecorder.start] Reusing existing stream');
      }

      // Only initialize analyser if we don't have one or if AudioContext is closed
      if (!analyserRef.current || !audioContextRef.current || audioContextRef.current.state === 'closed') {
        console.log('[Recorder] Initializing new analyser');
        await initializeAnalyser(streamRef.current);
      } else {
        // Just ensure AudioContext is running
        console.log('[Recorder] Reusing existing analyser, ensuring AudioContext is running');
        if (audioContextRef.current.state !== 'running') {
          await audioContextRef.current.resume();
          console.log('[Recorder] Resumed AudioContext, state:', audioContextRef.current.state);
        }
      }

      // Only create recorder once and reuse it (like Cygnus)
      if (!recorderRef.current) {
        const recorder = new MediaRecorder(streamRef.current, { mimeType });

        recorder.addEventListener('dataavailable', (event) => {
          console.log('[Recorder] dataavailable event:', {
            size: event.data.size,
            type: event.data.type,
            totalChunks: chunksRef.current.length + 1
          });
          if (event.data && event.data.size > 0) {
            chunksRef.current.push(event.data);
          }
        });

        recorder.addEventListener('stop', () => {
          const reason = stopReasonRef.current;
          const shouldProcess = reason === 'silence' || reason === 'max_duration';

          console.log('[Recorder] stop event:', {
            reason,
            shouldProcess,
            chunksCount: chunksRef.current.length,
            totalSize: chunksRef.current.reduce((sum, chunk) => sum + chunk.size, 0)
          });

          // Update state AFTER recorder has actually stopped
          setIsRecording(false);

          if (!shouldProcess) {
            // Manually stopped - don't process, don't notify callback
            console.log('[Recorder] Manual stop - not processing');
            chunksRef.current = [];
            stopReasonRef.current = 'manual';
            return;
          }

          // VAD stopped - process the audio
          const hasAudio = chunksRef.current.length > 0;
          const blob = hasAudio
            ? new Blob(chunksRef.current, { type: recorderRef.current?.mimeType || mimeType })
            : null;

          console.log('[Recorder] Created blob:', {
            hasAudio,
            blobSize: blob?.size,
            blobType: blob?.type
          });

          chunksRef.current = [];
          stopReasonRef.current = 'manual';
          onRecordingComplete(blob);
        });

        recorderRef.current = recorder;
      }

      const recorder = recorderRef.current;

      // Don't start if already recording
      if (recorder.state === 'recording') {
        return;
      }

      chunksRef.current = [];
      // Reset VAD state for new recording
      backgroundNoiseRef.current = Math.max(backgroundNoiseRef.current, 0.0004);
      smoothedEnergyRef.current = 0;
      recordingStartedAtRef.current = performance.now();
      lastSpeechTimeRef.current = recordingStartedAtRef.current;
      speechActiveRef.current = false;
      silenceTriggeredRef.current = false;
      accumulatedSilenceRef.current = 0;
      speechFramesRef.current = 0;

      console.log('[Recorder] Starting recording:', {
        state: recorder.state,
        mimeType: recorder.mimeType,
        stream: streamRef.current?.active,
        audioContext: audioContextRef.current?.state,
        hasAnalyser: !!analyserRef.current,
        monitoringActive: monitorRafRef.current !== null
      });

      recorder.start();
      setIsRecording(true);

      // ALWAYS restart monitoring to ensure it's running
      console.log('[Recorder] Starting/restarting monitoring loop');
      startMonitoringLevels();

      console.log('[useAudioRecorder.start] ✅ Recording started successfully');
    } catch (error) {
      console.error('[useAudioRecorder.start] ❌ Error during start:', error);
      setRecorderError(error instanceof Error ? error.message : 'Unable to start recording.');
      cleanup();
    }
  }, [cleanup, initializeAnalyser, onRecordingComplete, startMonitoringLevels]);

  // Cleanup on unmount - stop stream and release resources (like Cygnus line 655-672)
  useEffect(() => {
    return () => {
      if (recorderRef.current?.state === 'recording') {
        recorderRef.current.stop();
      }
      cleanup();
    };
  }, [cleanup]);

  return {
    isRecording,
    start,
    stop,
    recorderError,
  };
}
