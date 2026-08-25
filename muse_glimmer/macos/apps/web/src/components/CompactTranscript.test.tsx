import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { TranscriptEntry } from "../lib/transcript";
import { CompactTranscript } from "./CompactTranscript";

function entry(
  index: number,
  speaker: "user" | "agent" = "user",
): TranscriptEntry {
  return {
    key: `${speaker}-${index}`,
    segmentId: String(index),
    participantIdentity: speaker,
    speaker,
    text: `Line ${index}`,
    final: index % 2 === 0,
    order: index,
  };
}

describe("compact transcript", () => {
  it("gives useful empty guidance for microphone state", () => {
    const { rerender } = render(
      <CompactTranscript entries={[]} isMuted={false} />,
    );
    expect(screen.getByText("Say something to begin.")).toBeVisible();

    rerender(<CompactTranscript entries={[]} isMuted />);
    expect(
      screen.getByText("Unmute when you are ready to speak."),
    ).toBeVisible();
  });

  it("renders only the six newest entries with speaker labels and interim styling", () => {
    render(
      <CompactTranscript
        entries={Array.from({ length: 8 }, (_, index) =>
          entry(index, index % 2 ? "agent" : "user"),
        )}
        isMuted={false}
      />,
    );

    expect(screen.queryByText("Line 1")).not.toBeInTheDocument();
    expect(screen.getByText("Line 2")).toBeVisible();
    expect(screen.getByText("Line 7")).toHaveClass("transcript-interim");
    expect(screen.getAllByText("Muse")).toHaveLength(3);
    expect(screen.getAllByText("You")).toHaveLength(3);
  });
});
