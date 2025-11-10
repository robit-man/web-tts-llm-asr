import { Mic, Square } from "lucide-react";

type Props = {
  isRecording: boolean;
  disabled?: boolean;
  onStart: () => void;
  onStop: () => void;
};

export function RecorderButton({
  isRecording,
  disabled,
  onStart,
  onStop,
}: Props) {
  const label = isRecording ? "Stop capture" : "Start listening";

  const handleClick = () => {
    if (isRecording) {
      onStop();
    } else {
      onStart();
    }
  };

  return (
    <button
      className={`recorder-button ${isRecording ? "recorder-button--active" : ""}`}
      onClick={handleClick}
      disabled={disabled}
    >
      {isRecording ? <Square size={18} /> : <Mic size={18} />}
      <span>{label}</span>
    </button>
  );
}

export default RecorderButton;
