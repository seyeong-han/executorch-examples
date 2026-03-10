interface Props {
  label: string;
  status: string;
  progress: number;
  fileName?: string;
  fileIndex?: number;
  fileCount?: number;
  downloadedBytes?: number;
  totalBytes?: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function LoadingOverlay({
  label,
  status,
  progress,
  fileName,
  fileIndex,
  fileCount,
  downloadedBytes,
  totalBytes,
}: Props) {
  const isConnecting = status === "starting" || (!fileName && progress === 0);
  const isDone = status === "complete" || status === "done";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-5 rounded-3xl bg-white/80 backdrop-blur-xl px-12 py-10 shadow-2xl min-w-[320px]">
        <div className="relative h-10 w-10">
          {[...Array(8)].map((_, i) => (
            <span
              key={i}
              className="absolute left-1/2 top-0 h-[10px] w-[2.5px] origin-[50%_20px] rounded-full bg-gray-500"
              style={{
                transform: `rotate(${i * 45}deg)`,
                opacity: 0.15 + (i / 8) * 0.85,
                animation: "apple-spin 0.8s linear infinite",
                animationDelay: `${-((8 - i) / 8) * 0.8}s`,
              }}
            />
          ))}
        </div>

        <div className="text-center">
          <p className="text-sm font-medium text-gray-800">
            Downloading {label}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {isConnecting
              ? "Connecting to Hugging Face..."
              : isDone
              ? "Finishing up..."
              : fileName
              ? `${fileName}`
              : "Downloading..."}
          </p>
          {fileCount && fileCount > 0 && !isConnecting && !isDone && (
            <p className="text-[11px] text-gray-400 mt-0.5">
              File {fileIndex || 0} of {fileCount}
            </p>
          )}
        </div>

        <div className="w-full">
          <div className="h-1.5 w-full rounded-full bg-gray-200 overflow-hidden">
            <div
              className="h-full rounded-full bg-gray-600 transition-all duration-300 ease-out"
              style={{
                width: progress > 0 ? `${Math.min(progress, 100)}%` : undefined,
                animation: progress === 0 ? "indeterminate 1.5s ease-in-out infinite" : undefined,
              }}
            />
          </div>
          <div className="flex justify-between mt-1.5">
            <p className="text-[11px] text-gray-400">
              {progress > 0 ? `${progress}%` : ""}
            </p>
            {totalBytes && totalBytes > 0 && (
              <p className="text-[11px] text-gray-400">
                {formatBytes(downloadedBytes || 0)} / {formatBytes(totalBytes)}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
