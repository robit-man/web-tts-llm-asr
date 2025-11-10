import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import "./App.css";
import ModelStatusCard from "./components/ModelStatusCard";
import RecorderButton from "./components/RecorderButton";
import ConversationLog from "./components/ConversationLog";
import { useAudioRecorder } from "./hooks/useAudioRecorder";
import { useWhisperModel } from "./hooks/useWhisper";
import { useWebLLM } from "./hooks/useWebLLM";
import { usePiperModel } from "./hooks/usePiper";
import type { ConversationTurn, ModelStatus } from "./types/models";

const SYSTEM_PROMPT =
  "You are the voice of a hands-free companion. Keep replies short (max three sentences), helpful, and speak in the first person.";

const statusSort = (a: ModelStatus, b: ModelStatus) =>
  ["whisper", "webllm", "piper"].indexOf(a.model) -
  ["whisper", "webllm", "piper"].indexOf(b.model);

function App() {
  const whisper = useWhisperModel();
  const llm = useWebLLM();
  const piper = usePiperModel();

  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const turnsRef = useRef<ConversationTurn[]>([]);
  const [alert, setAlert] = useState<string | null>(null);
  const [spokenUrl, setSpokenUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const statuses = useMemo(() => {
    return [whisper.status, llm.status, piper.status].slice().sort(statusSort);
  }, [llm.status, piper.status, whisper.status]);

  const ready = statuses.every((status) => status.state === "ready");
  const loopBusy =
    isProcessing ||
    whisper.isTranscribing ||
    llm.isResponding ||
    piper.isSpeaking;

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

  const runLoop = useCallback(
    async (audioBuffer: AudioBuffer) => {
      setAlert(null);
      setIsProcessing(true);

      try {
        const transcript = await whisper.transcribe(audioBuffer);
        const userMessage = transcript.text?.trim();

        if (!userMessage) {
          setAlert("I could not detect any speech. Try again.");
          return;
        }

        const userTurn: ConversationTurn = {
          role: "user",
          content: userMessage,
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

        const reply = await llm.generate(llmMessages);
        if (!reply) {
          setAlert("The language model did not return a response.");
          return;
        }

        const assistantTurn: ConversationTurn = {
          role: "assistant",
          content: reply,
          timestamp: Date.now() + 1,
        };

        const historyWithAssistant = [...historyWithUser, assistantTurn];
        setTurns(historyWithAssistant);
        turnsRef.current = historyWithAssistant;

        const speechUrl = await piper.speak(reply);
        updateSpeechUrl(speechUrl);
      } catch (error) {
        setAlert(
          (error as Error).message ?? "Something went wrong in the pipeline.",
        );
      } finally {
        setIsProcessing(false);
      }
    },
    [llm, piper, updateSpeechUrl, whisper],
  );

  const recorder = useAudioRecorder(runLoop);

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
          onStop={recorder.stop}
        />
      </section>

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
          Models: Whisper (Transformers.js), WebLLM ({llm.status.message}), and
          Piper TTS (onnxruntime-web). All computation stays client-side.
        </p>
      </footer>
    </div>
  );
}

export default App;
