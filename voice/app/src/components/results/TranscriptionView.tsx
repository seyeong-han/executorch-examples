import type { TranscriptionResult } from "../../types";

interface Props {
  result: TranscriptionResult;
}

export function TranscriptionView({ result }: Props) {
  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Result</h2>
        <div className="flex gap-2">
          <span className="text-[11px] px-2 py-0.5 rounded-md bg-gray-100 text-gray-500">
            {result.model}
          </span>
          <span className="text-[11px] px-2 py-0.5 rounded-md bg-gray-100 text-gray-500">
            {result.backend}
          </span>
        </div>
      </div>

      <p className="text-gray-800 leading-relaxed text-[15px] mb-4">
        {result.text}
      </p>

      {result.segments.length > 0 && (
        <div className="space-y-1.5 mb-4">
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider">
            Segments
          </h3>
          {result.segments.map((seg, i) => (
            <div
              key={i}
              className="flex gap-3 text-sm bg-white/40 rounded-xl px-3 py-2"
            >
              <span className="text-gray-500 font-mono text-xs whitespace-nowrap pt-0.5">
                {seg.start.toFixed(1)}s — {seg.end.toFixed(1)}s
              </span>
              <span className="text-gray-700">{seg.text}</span>
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-4 text-xs text-gray-400 pt-3 border-t border-gray-200/50">
        <span>
          {result.performance.inference_time_s.toFixed(2)}s inference
        </span>
        {result.performance.tokens_generated ? (
          <span>{result.performance.tokens_generated} tokens</span>
        ) : null}
        {result.performance.tokens_per_second ? (
          <span>{result.performance.tokens_per_second.toFixed(1)} tok/s</span>
        ) : null}
      </div>
    </div>
  );
}
