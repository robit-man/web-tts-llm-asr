export type ModelKey = "whisper" | "webllm" | "piper";

export type ModelState = "idle" | "loading" | "ready" | "error";

export interface ModelStatus {
  model: ModelKey;
  label: string;
  state: ModelState;
  message?: string;
  detail?: string;
  progress?: number;
}

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}
