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

const statusSort = (a: ModelStatus, b: ModelStatus) =>
  ["whisper", "webllm", "piper"].indexOf(a.model) -
  ["whisper", "webllm", "piper"].indexOf(b.model);

const WHISPER_OPTIONS = [
  { id: "Xenova/whisper-tiny", label: "Whisper Tiny (~75 MB)" },
  { id: "Xenova/whisper-base", label: "Whisper Base (~142 MB)" },
  { id: "Xenova/whisper-small", label: "Whisper Small (~462 MB)" },
];

function App() {
  const whisper = useWhisperModel();
  const llm = useWebLLM();
  const piper = usePiperModel();

  const {
    status: whisperStatus,
    isTranscribing,
    transcribe,
    model: whisperModel,
    setModel: setWhisperModel,
    partialText,
    finalText,
    chunks,
    error: whisperError,
    setManualTranscript,
  } = whisper;
  const { status: llmStatus, generate, isResponding } = llm;
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
  const whisperPrimedRef = useRef(false);
  const whisperPrimingRef = useRef(false);

  const statuses = useMemo(() => {
    return [whisperStatus, llmStatus, piperStatus].slice().sort(statusSort);
  }, [llmStatus, piperStatus, whisperStatus]);

  const ready = statuses.every((status) => status.state === "ready");
  const whisperChoice =
    WHISPER_OPTIONS.find((option) => option.id === whisperModel) ??
    { id: whisperModel, label: whisperModel };
  const displayedVoices = voices.slice(0, 60);
  const selectedVoiceLabel =
    voices.find((voice) => voice.id === voiceId)?.name ?? `Voice ${voiceId + 1}`;
  const loopBusy =
    isProcessing ||
    isTranscribing ||
    isResponding ||
    isSpeaking;

  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);

  const updateSpeechUrl = useCallback((url: string | null) => {
    setSpokenUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return url;
    });
  }, []);

  useEffect(() => {
    return () => {
      if (spokenUrl) {
        URL.revokeObjectURL(spokenUrl);
      }
    };
  }, [spokenUrl]);

  const processMessage = useCallback(
    async (userMessage: string) => {
      const trimmed = userMessage.trim();
      const fallback = trimmed.length > 0 ? trimmed : "(no intelligible speech captured)";
      const userTurn: ConversationTurn = {
        role: "user",
        content: fallback,
        timestamp: Date.now(),
      };

      const historyWithUser = [...turnsRef.current, userTurn];
      setTurns(historyWithUser);
      turnsRef.current = historyWithUser;

      const llmMessages: ChatCompletionMessageParam[] = [
        { role: "system", content: SYSTEM_PROMPT },
        ...historyWithUser.map((turn) => ({
          role: turn.role,
          content: turn.content,
        })),
      ];

      const reply = await generate(llmMessages);
      if (!reply) {
        throw new Error("The language model did not return a response.");
      }

      const assistantTurn: ConversationTurn = {
        role: "assistant",
        content: reply,
        timestamp: Date.now() + 1,
      };

      const historyWithAssistant = [...historyWithUser, assistantTurn];
      setTurns(historyWithAssistant);
      turnsRef.current = historyWithAssistant;

      const speechUrl = await speak(reply);
      updateSpeechUrl(speechUrl);
    },
    [generate, speak, updateSpeechUrl],
  );

  const runManualLoop = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content) {
        return;
      }
      setManualTranscript(content);
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
    [ingressMode, processMessage, setManualTranscript, speak, updateSpeechUrl],
  );

  const runLoop = useCallback(
    async (audioBuffer: AudioBuffer) => {
      setAlert(null);
      setIsProcessing(true);

      try {
        const transcript = await transcribe(audioBuffer);
        let userMessage = transcript.text?.trim();
        const silentFallback = "(no intelligible speech captured)";

        if (!userMessage || userMessage.length === 0) {
          userMessage = silentFallback;
        }
        await processMessage(userMessage);
      } catch (error) {
        setAlert(
          (error as Error).message ?? "Something went wrong in the pipeline.",
        );
      } finally {
        setIsProcessing(false);
      }
    },
    [processMessage, transcribe],
  );

  const recorder = useAudioRecorder({
    onComplete: runLoop,
    onLevels: setLevelSnapshot,
  });

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
          <p className="eyebrow">Progressive Web Demo</p>
          <h1>Browser-native speech, reasoning, and speech.</h1>
          <p className="lede">
            Whisper ingests your microphone, WebLLM reasons locally, and Piper
            answers back without leaving this tab.
          </p>
        </div>
        <div className="badge">{ready ? "All engines ready" : "Preparing..."}</div>
      </header>

      <section className="status-grid">
        {statuses.map((status) => (
          <ModelStatusCard key={status.model} status={status} />
        ))}
      </section>

      <section className="ingress-panel">
        <div>
          <h2>Verbal ingress</h2>
          <p>
            Tap record, speak freely, then watch the ASR → LLM → TTS chain fire.
            For best results, keep utterances under 10 seconds.
          </p>
        </div>
        <RecorderButton
          isRecording={recorder.isRecording}
          disabled={!ready || loopBusy}
          onStart={recorder.start}
          onStop={() => recorder.stop("manual")}
        />
      </section>

      <section className="textual-ingress">
        <div>
          <h2>Textual ingress</h2>
          <p>
            Prefer typing? Send crafted prompts directly into the same Whisper → LLM → Piper pipeline or flip the
            toggle to speak the text immediately with Piper.
          </p>
        </div>
        <div className="textual-ingress__visual">
          <AudioVisualizer
            rms={levelSnapshot.rms}
            eqLevels={levelSnapshot.eq}
            isActive={levelSnapshot.isActive}
          />
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
      <section className="transcript-panel">
        <div className="transcript-panel__row">
          <h3>Partial transcript</h3>
          <p>{partialText || (isTranscribing ? "Listening…" : "—")}</p>
        </div>
        <div className="transcript-panel__row">
          <h3>Final transcript</h3>
          <p>{finalText || "—"}</p>
          <span className="transcript-panel__meta">
            {chunks.length > 0 ? `${chunks.length} segment${chunks.length === 1 ? "" : "s"}` : "No segments yet"}
          </span>
        </div>
        {whisperError && <p className="transcript-error">{whisperError}</p>}
      </section>

      <AudioVisualizer
        rms={levelSnapshot.rms}
        eqLevels={levelSnapshot.eq}
        isActive={levelSnapshot.isActive}
      />

      <div className="model-controls">
        <div className="model-control">
          <label htmlFor="whisper-model">Whisper model</label>
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
          <p className="model-hint">{whisperChoice.label}</p>
        </div>

        <div className="model-control">
          <label htmlFor="piper-voice">Piper voice</label>
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
          <p className="model-hint">{selectedVoiceLabel}</p>
        </div>
      </div>

      {alert && <div className="alert">{alert}</div>}
      {recorder.recorderError && (
        <div className="alert alert--warning">{recorder.recorderError}</div>
      )}
      {loopBusy && (
        <div className="status-pill">
          Listening, transcribing, reasoning, or speaking…
        </div>
      )}

      <ConversationLog turns={turns} />

      {spokenUrl && (
        <div className="player">
          <p>Latest Piper rendering</p>
          <audio controls autoPlay src={spokenUrl} />
        </div>
      )}

      <footer className="app__footer">
        <p>
          Models: Whisper (Transformers.js), WebLLM ({llmStatus.message}), and
          Piper TTS (onnxruntime-web). All computation stays client-side.
        </p>
      </footer>
    </div>
  );
}

export default App;
