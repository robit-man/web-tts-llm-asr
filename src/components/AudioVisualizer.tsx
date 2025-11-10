import type { FC } from "react";
import "./AudioVisualizer.css";

interface AudioVisualizerProps {
  rms: number;
  eqLevels: number[];
  isActive: boolean;
}

const formatDb = (rms: number) => {
  if (rms <= 0) return "-inf dB";
  const db = 20 * Math.log10(rms);
  return `${db.toFixed(1)} dB`;
};

export const AudioVisualizer: FC<AudioVisualizerProps> = ({ rms, eqLevels, isActive }) => {
  const normalized = Math.min(1, rms / 0.05);
  return (
    <div className="audio-viz">
      <div className="audio-viz__rms">
        <div
          className={`audio-viz__pulse ${isActive ? "audio-viz__pulse--active" : ""}`}
          style={{ transform: `scale(${0.8 + normalized * 0.4})` }}
        />
        <div className="audio-viz__rms-label">{formatDb(rms)}</div>
        <div className="audio-viz__status">{isActive ? "Voice detected" : "Listening"}</div>
      </div>
      <div className="audio-viz__eq">
        {eqLevels.map((level, index) => (
          <div key={index} className="audio-viz__bar">
            <div
              className="audio-viz__bar-fill"
              style={{ height: `${Math.min(100, level)}%` }}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default AudioVisualizer;
