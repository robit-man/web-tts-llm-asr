import type { ConversationTurn } from "../types/models";

type Props = {
  turns: ConversationTurn[];
};

export function ConversationLog({ turns }: Props) {
  if (!turns.length) {
    return (
      <div className="conversation-log conversation-log--empty">
        <p>Ask something out loud to begin the loop.</p>
      </div>
    );
  }

  return (
    <div className="conversation-log">
      {turns.map((turn) => (
        <div
          key={turn.timestamp}
          className={`conversation-log__item conversation-log__item--${turn.role}`}
        >
          <p className="conversation-log__role">
            {turn.role === "user" ? "You" : "Assistant"}
          </p>
          <p className="conversation-log__content">{turn.content}</p>
        </div>
      ))}
    </div>
  );
}

export default ConversationLog;
