import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import "./App.css";
import ModelStatusCard from "./components/ModelStatusCard";
import RecorderButton from "./components/RecorderButton";
import ConversationLog from "./components/ConversationLog";
import AudioVisualizer from "./components/AudioVisualizer";
import { useAudioRecorder, type LevelSnapshot } from "./hooks/useAudioRecorder";
import { useWhisperModel } from "./hooks/useWhisper";
import { useWebLLM } from "./hooks/useWebLLM";
import { usePiperModel } from "./hooks/usePiper";
import { TARGET_SAMPLE_RATE } from "./utils/audio";
import type { ConversationTurn, ModelStatus } from "./types/models";

const SYSTEM_PROMPT =
  "You are the voice of a hands-free companion. Keep replies short (max three sentences), helpful, and speak in the first person.";

const STORAGE_KEYS = {
  WHISPER_MODEL: "whisper_model",
  WEBLLM_MODEL: "webllm_model",
  PIPER_VOICE: "piper_voice",
} as const;

const statusSort = (a: ModelStatus, b: ModelStatus) =>
  ["whisper", "webllm", "piper"].indexOf(a.model) -
  ["whisper", "webllm", "piper"].indexOf(b.model);

const WHISPER_OPTIONS = [
  { id: "Xenova/whisper-tiny", label: "Whisper Tiny (~75 MB)" },
  { id: "Xenova/whisper-base", label: "Whisper Base (~142 MB)" },
  { id: "Xenova/whisper-small", label: "Whisper Small (~462 MB)" },
];

const WEBLLM_OPTIONS = [
  { id: "Llama-3.2-1B-Instruct-q4f32_1-MLC", label: "Llama 3.2 1B (1.1 GB)", group: "Recommended" },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama 3.2 1B - Low Memory (879 MB)", group: "Recommended" },
  { id: "Llama-3.2-3B-Instruct-q4f32_1-MLC", label: "Llama 3.2 3B (2.9 GB)", group: "Recommended" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", label: "Llama 3.2 3B - Low Memory (2.3 GB)", group: "Recommended" },
  { id: "Phi-3.5-mini-instruct-q4f16_1-MLC", label: "Phi 3.5 Mini (2.7 GB)", group: "Recommended" },
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 0.5B (512 MB)", group: "Lightweight" },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 1.5B (1.2 GB)", group: "Lightweight" },
  { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 3B (2.2 GB)", group: "Lightweight" },
  { id: "SmolLM2-1.7B-Instruct-q4f16_1-MLC", label: "SmolLM2 1.7B (1.2 GB)", group: "Lightweight" },
  { id: "SmolLM2-360M-Instruct-q4f16_1-MLC", label: "SmolLM2 360M (300 MB)", group: "Lightweight" },
  { id: "gemma-2-2b-it-q4f16_1-MLC", label: "Gemma 2 2B (1.7 GB)", group: "Alternative" },
  { id: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC", label: "TinyLlama 1.1B (831 MB)", group: "Alternative" },
  { id: "Hermes-3-Llama-3.2-3B-q4f16_1-MLC", label: "Hermes 3 Llama 3.2 3B (2.3 GB)", group: "Alternative" },
];

function App() {
  // Load saved model preferences from localStorage
  const savedWhisperModel = localStorage.getItem(STORAGE_KEYS.WHISPER_MODEL) || undefined;
  const savedLlmModel = localStorage.getItem(STORAGE_KEYS.WEBLLM_MODEL) || undefined;
  const savedVoiceId = localStorage.getItem(STORAGE_KEYS.PIPER_VOICE);

  const whisper = useWhisperModel(savedWhisperModel);
  const llm = useWebLLM(savedLlmModel);
  const piper = usePiperModel();

  const {
    status: whisperStatus,
    isTranscribing,
    transcribe,
    startStream,
    pushStreamChunk,
    finishStream,
    model: whisperModel,
    setModel: setWhisperModel,
    partialText,
    finalText,
    error: whisperError,
  } = whisper;
  const {
    status: llmStatus,
    generate,
    isResponding,
    partialResponse,
    model: llmModel,
    setModel: setLlmModel,
  } = llm;
  const {
    status: piperStatus,
    speak,
    isSpeaking,
    voices,
    voiceId,
    setVoiceId,
  } = piper;

  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const turnsRef = useRef<ConversationTurn[]>([]);
  const [alert, setAlert] = useState<string | null>(null);
  const [spokenUrl, setSpokenUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [manualInput, setManualInput] = useState("");
  const [levelSnapshot, setLevelSnapshot] = useState<LevelSnapshot>({
    rms: 0,
    eq: Array(6).fill(6),
    isActive: false,
  });
  const [ingressMode, setIngressMode] = useState<"llm" | "tts">("llm");
  const [streamingAssistantTurn, setStreamingAssistantTurn] = useState<ConversationTurn | null>(null);
  const whisperPrimedRef = useRef(false);
  const whisperPrimingRef = useRef(false);

  const statuses = useMemo(() => {
    return [whisperStatus, llmStatus, piperStatus].slice().sort(statusSort);
  }, [llmStatus, piperStatus, whisperStatus]);

  const ready = statuses.every((status) => status.state === "ready");
  const displayedVoices = voices.slice(0, 60);
  const selectedVoiceLabel =
    voices.find((voice) => voice.id === voiceId)?.name ?? `Voice ${voiceId + 1}`;
  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);

  // Save model preferences to localStorage when they change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.WHISPER_MODEL, whisperModel);
  }, [whisperModel]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.WEBLLM_MODEL, llmModel);
  }, [llmModel]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.PIPER_VOICE, String(voiceId));
  }, [voiceId]);

  // Restore saved voice ID when voices are loaded
  useEffect(() => {
    if (savedVoiceId && voices.length > 0 && voiceId === 0) {
      const savedId = parseInt(savedVoiceId, 10);
      if (!isNaN(savedId) && voices.some(v => v.id === savedId)) {
        setVoiceId(savedId);
      }
    }
  }, [voices, savedVoiceId, voiceId, setVoiceId]);

  const updateSpeechUrl = useCallback((url: string | null) => {
    setSpokenUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return url;
    });
  }, []);

  const resetConversation = useCallback(() => {
    setTurns([]);
    turnsRef.current = [];
    setStreamingAssistantTurn(null);
    setAlert(null);
    updateSpeechUrl(null);
  }, [updateSpeechUrl]);

  useEffect(() => {
    return () => {
      if (spokenUrl) {
        URL.revokeObjectURL(spokenUrl);
      }
    };
  }, [spokenUrl]);

  const estimateTokens = useCallback((text: string): number => {
    return Math.ceil(text.length / 4);
  }, []);

  const pruneConversationForContext = useCallback(
    (history: ConversationTurn[], maxContextTokens = 1536): ConversationTurn[] => {
      const systemPromptTokens = estimateTokens(SYSTEM_PROMPT);
      let availableTokens = maxContextTokens - systemPromptTokens - 1024;

      const pruned: ConversationTurn[] = [];
      for (let i = history.length - 1; i >= 0; i--) {
        const turn = history[i];
        const turnTokens = estimateTokens(turn.content);
        if (availableTokens - turnTokens < 0 && pruned.length >= 2) {
          break;
        }
        availableTokens -= turnTokens;
        pruned.unshift(turn);
      }
      return pruned;
    },
    [estimateTokens],
  );

  const processMessage = useCallback(
    async (userMessage: string) => {
      const trimmed = userMessage.trim();
      if (!trimmed) {
        setAlert("I didn't hear anything. Try again.");
        return;
      }
      const userTurn: ConversationTurn = {
        role: "user",
        content: trimmed,
        timestamp: Date.now(),
      };

      const historyWithUser = [...turnsRef.current, userTurn];
      setTurns(historyWithUser);
      turnsRef.current = historyWithUser;

      const prunedHistory = pruneConversationForContext(historyWithUser);
      const llmMessages: ChatCompletionMessageParam[] = [
        { role: "system", content: SYSTEM_PROMPT },
        ...prunedHistory.map((turn) => ({
          role: turn.role,
          content: turn.content,
        })),
      ];

      const streamingTurn: ConversationTurn = {
        role: "assistant",
        content: "",
        timestamp: Date.now() + 1,
      };
      setStreamingAssistantTurn(streamingTurn);

      const reply = await generate(llmMessages);
      if (!reply) {
        setStreamingAssistantTurn(null);
        throw new Error("The language model did not return a response.");
      }

      const assistantTurn: ConversationTurn = {
        role: "assistant",
        content: reply,
        timestamp: Date.now() + 1,
      };

      setStreamingAssistantTurn(null);
      const historyWithAssistant = [...historyWithUser, assistantTurn];
      setTurns(historyWithAssistant);
      turnsRef.current = historyWithAssistant;

      const speechUrl = await speak(reply);
      updateSpeechUrl(speechUrl);
    },
    [generate, speak, updateSpeechUrl, pruneConversationForContext],
  );

  const runManualLoop = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content) {
        return;
      }
      setAlert(null);
      setIsProcessing(true);
      try {
        if (ingressMode === "llm") {
          await processMessage(content);
        } else {
          const url = await speak(content);
          updateSpeechUrl(url);
        }
        setManualInput("");
      } catch (error) {
        setAlert(
          (error as Error).message ??
            (ingressMode === "llm"
              ? "Unable to process your typed request."
              : "Unable to synthesize speech."),
        );
      } finally {
        setIsProcessing(false);
      }
    },
    [ingressMode, processMessage, speak, updateSpeechUrl],
  );

  const handleAudioChunk = useCallback((chunk: Float32Array) => {
    pushStreamChunk(chunk);
  }, [pushStreamChunk]);

  const handleRecordingComplete = useCallback(async (_blob: Blob) => {
    setAlert(null);
    setIsProcessing(true);
    try {
      const result = await finishStream();
      if (result && result.text.trim()) {
        await processMessage(result.text.trim());
      } else {
        setAlert("I didn't hear anything. Try again.");
      }
    } catch (error) {
      setAlert((error as Error).message ?? "Unable to transcribe audio.");
    } finally {
      setIsProcessing(false);
    }
  }, [finishStream, processMessage]);

  const recorder = useAudioRecorder({
    onRecordingComplete: handleRecordingComplete,
    onLevels: setLevelSnapshot,
    onAudioChunk: handleAudioChunk,
    enableVAD: true,
    silenceDurationMs: 1200,
  });
  const loopBusy =
    isProcessing ||
    isTranscribing ||
    isResponding ||
    isSpeaking ||
    recorder.isRecording;

  const handleStartRecording = useCallback(async () => {
    try {
      await startStream();
      await recorder.start();
    } catch (error) {
      setAlert((error as Error).message ?? "Unable to access microphone.");
    }
  }, [recorder, startStream]);

  const handleStopRecording = useCallback(() => {
    recorder.stop();
  }, [recorder]);

  useEffect(() => {
    if (
      whisperStatus.state === "ready" &&
      !whisperPrimedRef.current &&
      !whisperPrimingRef.current
    ) {
      if (typeof AudioBuffer === "undefined") return;
      whisperPrimingRef.current = true;
      const silentBuffer = new AudioBuffer({
        length: Math.max(1, Math.floor(TARGET_SAMPLE_RATE * 0.8)),
        numberOfChannels: 1,
        sampleRate: TARGET_SAMPLE_RATE,
      });
      transcribe(silentBuffer)
        .catch(() => {})
        .finally(() => {
          whisperPrimedRef.current = true;
          whisperPrimingRef.current = false;
        });
    }
  }, [transcribe, whisperStatus.state]);

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <p className="eyebrow">Local Voice Assistant</p>
          <h1>Voice, Intelligence, Speech</h1>
        </div>
        <div className="badge">{ready ? "Ready" : "Loading..."}</div>
      </header>

      <section className="status-grid">
        {statuses.map((status) => (
          <ModelStatusCard key={status.model} status={status}>
            {status.model === "whisper" && (
              <div className="model-control">
                <label htmlFor="whisper-model">Model</label>
                <select
                  id="whisper-model"
                  value={whisperModel}
                  onChange={(event) => setWhisperModel(event.target.value)}
                  disabled={recorder.isRecording || loopBusy}
                >
                  {WHISPER_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                  {!WHISPER_OPTIONS.some((option) => option.id === whisperModel) && (
                    <option value={whisperModel}>{whisperModel}</option>
                  )}
                </select>
              </div>
            )}
            {status.model === "webllm" && (
              <div className="model-control">
                <label htmlFor="webllm-model">Model</label>
                <select
                  id="webllm-model"
                  value={llmModel}
                  onChange={(event) => setLlmModel(event.target.value)}
                  disabled={recorder.isRecording || loopBusy}
                >
                  {WEBLLM_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                  {!WEBLLM_OPTIONS.some((option) => option.id === llmModel) && (
                    <option value={llmModel}>{llmModel}</option>
                  )}
                </select>
              </div>
            )}
            {status.model === "piper" && (
              <div className="model-control">
                <label htmlFor="piper-voice">Voice</label>
                <select
                  id="piper-voice"
                  value={String(voiceId)}
                  onChange={(event) => setVoiceId(Number(event.target.value))}
                  disabled={!ready || recorder.isRecording || loopBusy}
                >
                  {displayedVoices.length > 0 ? (
                    displayedVoices.map((voice) => (
                      <option key={voice.id} value={voice.id}>
                        {voice.name}
                      </option>
                    ))
                  ) : (
                    <option value="0" disabled>
                      Loading voices…
                    </option>
                  )}
                  {displayedVoices.every((voice) => voice.id !== voiceId) && (
                    <option value={voiceId}>{selectedVoiceLabel}</option>
                  )}
                </select>
              </div>
            )}
          </ModelStatusCard>
        ))}
      </section>

      <div className="ingress-row">
        <section className="ingress-panel">
          <div className="ingress-panel__copy">
            <h2>Voice Input</h2>
            <p>
              Tap to record. VAD will auto-detect when you finish speaking.
            </p>
          </div>
          <div className="ingress-panel__controls">
            <RecorderButton
              isRecording={recorder.isRecording}
              disabled={!ready || loopBusy}
              onStart={() => {
                void handleStartRecording();
              }}
              onStop={() => {
                void handleStopRecording();
              }}
            />
            <AudioVisualizer
              rms={levelSnapshot.rms}
              eqLevels={levelSnapshot.eq}
              isActive={levelSnapshot.isActive}
            />
            <div className="ingress-panel__transcript">
              <div>
                <p className="ingress-panel__label">Partial</p>
                <p className="ingress-panel__text">{partialText || "\u00A0"}</p>
              </div>
              <div>
                <p className="ingress-panel__label">Final</p>
                <p className="ingress-panel__text">{finalText || "\u00A0"}</p>
              </div>
              {whisperError && <p className="transcript-error">{whisperError}</p>}
            </div>
          </div>
        </section>

        <section className="textual-ingress">
        <div>
          <h2>Text Input</h2>
          <p>
            Type your message or text to speak directly.
          </p>
        </div>
        <form
          className="textual-ingress__form"
          onSubmit={async (event) => {
            event.preventDefault();
            if (!manualInput.trim() || !ready || loopBusy) {
              return;
            }
            await runManualLoop(manualInput.trim());
          }}
        >
          <textarea
            rows={3}
            placeholder="Type a prompt or instruction…"
            value={manualInput}
            onChange={(event) => setManualInput(event.target.value)}
            disabled={!ready || loopBusy}
          />
          <div className="textual-ingress__controls">
            <label className="textual-ingress__toggle">
              <input
                type="checkbox"
                checked={ingressMode === "tts"}
                onChange={(event) => setIngressMode(event.target.checked ? "tts" : "llm")}
                disabled={!ready || loopBusy}
              />
              <span>Send directly to Piper TTS</span>
            </label>
            <button type="submit" disabled={!ready || loopBusy || !manualInput.trim()}>
              {ingressMode === "tts" ? "Speak text" : "Send to loop"}
            </button>
          </div>
        </form>
        </section>
      </div>

      {alert && <div className="alert">{alert}</div>}
      {recorder.recorderError && (
        <div className="alert alert--warning">{recorder.recorderError}</div>
      )}
      {loopBusy && (
        <div className="status-pill">
          Processing...
        </div>
      )}

      {turns.length > 0 && (
        <div className="conversation-controls">
          <button
            onClick={resetConversation}
            disabled={loopBusy}
            className="reset-button"
          >
            Clear Conversation
          </button>
        </div>
      )}

      <ConversationLog
        turns={streamingAssistantTurn && partialResponse
          ? [...turns, { ...streamingAssistantTurn, content: partialResponse }]
          : turns
        }
      />

      {spokenUrl && (
        <div className="player">
          <p>Latest Piper rendering</p>
          <audio controls autoPlay src={spokenUrl} />
        </div>
      )}

      <footer className="app__footer">
        <p>
          Whisper ASR • WebLLM • Piper TTS — All processing runs locally in your browser
        </p>
      </footer>
    </div>
  );
}

export default App;
