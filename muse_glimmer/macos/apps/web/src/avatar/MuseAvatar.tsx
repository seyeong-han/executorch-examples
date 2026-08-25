import { useId } from "react";

import type { MuseAnimation } from "../lib/agentPresentation";

interface MuseAvatarProps {
  animation: MuseAnimation;
  reducedMotion: boolean;
}

const animationLabels: Record<MuseAnimation, string> = {
  idle: "ready",
  listening: "listening",
  thinking: "thinking",
  working: "connecting",
  happy: "speaking",
};

export function MuseAvatar({ animation, reducedMotion }: MuseAvatarProps) {
  const titleId = useId();
  const descriptionId = useId();
  const visibleAnimation = reducedMotion ? "idle" : animation;

  return (
    <div
      className="muse-avatar"
      data-animation={visibleAnimation}
      data-reduced-motion={reducedMotion || undefined}
    >
      <svg
        className="muse-avatar-art"
        viewBox="0 0 360 360"
        role="img"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <title id={titleId}>Muse Glimmer</title>
        <desc id={descriptionId}>
          An abstract blue voice companion, currently{" "}
          {animationLabels[animation]}.
        </desc>
        <defs>
          <linearGradient
            id="muse-body"
            x1="72"
            y1="44"
            x2="286"
            y2="315"
            gradientUnits="userSpaceOnUse"
          >
            <stop stopColor="#53b8ff" />
            <stop offset="0.46" stopColor="#0668e1" />
            <stop offset="1" stopColor="#004398" />
          </linearGradient>
          <radialGradient
            id="muse-glow"
            cx="0"
            cy="0"
            r="1"
            gradientTransform="translate(137 112) rotate(52) scale(198)"
          >
            <stop stopColor="#ffffff" stopOpacity="0.72" />
            <stop offset="0.34" stopColor="#bde3ff" stopOpacity="0.3" />
            <stop offset="1" stopColor="#0668e1" stopOpacity="0" />
          </radialGradient>
          <filter id="muse-shadow" x="-30%" y="-30%" width="160%" height="170%">
            <feDropShadow
              dx="0"
              dy="18"
              stdDeviation="18"
              floodColor="#064b96"
              floodOpacity="0.22"
            />
          </filter>
        </defs>
        <g className="muse-orbit" aria-hidden="true">
          <circle cx="180" cy="180" r="145" />
          <path d="M50 182c36-28 76-42 130-42s94 14 130 42" />
        </g>
        <g className="muse-character" filter="url(#muse-shadow)">
          <path
            className="muse-ear muse-ear-left"
            d="M92 123C58 111 37 126 40 151c3 27 31 45 63 40Z"
          />
          <path
            className="muse-ear muse-ear-right"
            d="M268 123c34-12 55 3 52 28-3 27-31 45-63 40Z"
          />
          <path
            className="muse-body"
            fill="url(#muse-body)"
            d="M180 63c70 0 116 47 116 116 0 35-8 69-25 95-19 30-49 45-91 45s-72-15-91-45c-17-26-25-60-25-95 0-69 46-116 116-116Z"
          />
          <path
            className="muse-body-glow"
            fill="url(#muse-glow)"
            d="M180 63c70 0 116 47 116 116 0 35-8 69-25 95-19 30-49 45-91 45s-72-15-91-45c-17-26-25-60-25-95 0-69 46-116 116-116Z"
          />
          <g className="muse-face">
            <rect
              className="muse-eye muse-eye-left"
              x="126"
              y="139"
              width="23"
              height="62"
              rx="12"
            />
            <rect
              className="muse-eye muse-eye-right"
              x="211"
              y="139"
              width="23"
              height="62"
              rx="12"
            />
            <path className="muse-mouth" d="M151 231c18 14 40 14 58 0" />
          </g>
          <g className="muse-voice" aria-hidden="true">
            <path d="M140 250v15" />
            <path d="M160 242v31" />
            <path d="M180 234v47" />
            <path d="M200 242v31" />
            <path d="M220 250v15" />
          </g>
        </g>
      </svg>
    </div>
  );
}
