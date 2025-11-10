import { useCallback, useRef, useState } from "react";
import { TARGET_SAMPLE_RATE } from "../utils/audio";

export function useAudioRecorder(
  onComplete: (buffer: AudioBuffer) => Promise<void> | void,
) {
  const [isRecording, setIsRecording] = useState(false);
  const [recorderError, setRecorderError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const cleanup = useCallback(() => {
    mediaRecorderRef.current = null;
    chunksRef.current = [];
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  const start = useCallback(async () => {
    if (isRecording) return;
    setRecorderError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        setIsRecording(false);
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const arrayBuffer = await blob.arrayBuffer();
          const audioContext = new AudioContext({
            sampleRate: TARGET_SAMPLE_RATE,
          });
          const audioBuffer = await audioContext.decodeAudioData(
            arrayBuffer.slice(0),
          );
          await onComplete(audioBuffer);
          await audioContext.close();
        } catch (error) {
          setRecorderError(
            (error as Error).message ?? "Could not decode audio data",
          );
        } finally {
          cleanup();
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      setRecorderError(
        (error as Error).message ?? "Microphone permissions denied",
      );
      cleanup();
    }
  }, [cleanup, isRecording, onComplete]);

  const stop = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
  }, [isRecording]);

  return {
    isRecording,
    start,
    stop,
    recorderError,
  };
}
