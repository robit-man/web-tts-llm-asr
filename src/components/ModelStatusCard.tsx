import type { ModelStatus } from "../types/models";
import { Cpu, Mic, Waves } from "lucide-react";
import "./ModelStatusCard.css";

const iconMap = {
  whisper: Mic,
  webllm: Cpu,
  piper: Waves,
} as const;

type Props = {
  status: ModelStatus;
  children?: React.ReactNode;
};

export function ModelStatusCard({ status, children }: Props) {
  const Icon = iconMap[status.model];
  const stateLabel = {
    ready: "Ready",
    loading: "Loading",
    error: "Error",
    idle: "Idle",
  }[status.state];

  return (
    <div className="status-card">
      <div className="status-card__header">
        <div className="status-card__icon">
          <Icon size={20} />
        </div>
        <div>
          <p className="status-card__title">{status.label}</p>
          <p className={`status-card__state status-card__state--${status.state}`}>
            {stateLabel}
          </p>
        </div>
      </div>
      <p className="status-card__message">{status.message}</p>
      {status.detail && (
        <p className="status-card__detail" title={status.detail}>
          {status.detail}
        </p>
      )}
      {status.state === "loading" && typeof status.progress === "number" && (
        <div className="status-card__progress">
          <div
            className="status-card__progress-bar"
            style={{ width: `${Math.round(status.progress * 100)}%` }}
          />
        </div>
      )}
      {children}
    </div>
  );
}

export default ModelStatusCard;
