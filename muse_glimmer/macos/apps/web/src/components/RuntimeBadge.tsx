export function RuntimeBadge() {
  return (
    <div className="runtime-badge" aria-label="Running on ExecuTorch">
      <span className="runtime-chip" aria-hidden="true">
        <span>ET</span>
      </span>
      <span>
        Running on <strong>ExecuTorch</strong>
      </span>
    </div>
  );
}
