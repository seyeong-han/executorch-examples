import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MuseAvatar } from "./MuseAvatar";

describe("Muse avatar", () => {
  it("describes the current voice state to assistive technology", () => {
    render(<MuseAvatar animation="listening" reducedMotion={false} />);

    expect(
      screen.getByRole("img", { name: /Muse Glimmer/i }),
    ).toHaveAccessibleDescription(
      "An abstract blue voice companion, currently listening.",
    );
  });

  it("holds the visual animation at idle when reduced motion is requested", () => {
    const { container } = render(
      <MuseAvatar animation="happy" reducedMotion />,
    );

    expect(container.firstElementChild).toHaveAttribute(
      "data-animation",
      "idle",
    );
    expect(container.firstElementChild).toHaveAttribute(
      "data-reduced-motion",
      "true",
    );
    expect(screen.getByRole("img")).toHaveAccessibleDescription(
      "An abstract blue voice companion, currently speaking.",
    );
  });
});
