import type { TranscriptEntry } from "../lib/transcript";

interface CompactTranscriptProps {
  entries: TranscriptEntry[];
  isMuted: boolean;
}

export function CompactTranscript({
  entries,
  isMuted,
}: CompactTranscriptProps) {
  const visibleEntries = entries.slice(-6);

  return (
    <section className="transcript" aria-label="Conversation transcript">
      {visibleEntries.length === 0 ? (
        <p className="transcript-empty">
          {isMuted
            ? "Unmute when you are ready to speak."
            : "Say something to begin."}
        </p>
      ) : (
        <ol
          className="transcript-list"
          aria-live="polite"
          aria-relevant="additions text"
        >
          {visibleEntries.map((entry) => (
            <li
              className={`transcript-line transcript-line--${entry.speaker}`}
              key={entry.key}
            >
              <span className="transcript-speaker">
                {entry.speaker === "user" ? "You" : "Muse"}
              </span>
              <span className={entry.final ? undefined : "transcript-interim"}>
                {entry.text}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
