import { useState } from "react";
import { ModelSelector } from "./components/ModelSelector";
import { AudioInput } from "./components/AudioInput";
import { TranscriptionView } from "./components/results/TranscriptionView";
import { LoadingOverlay } from "./components/LoadingOverlay";
import { useModels } from "./hooks/useModels";
import { useTranscribe } from "./hooks/useTranscribe";
import { downloadModel, deleteModel } from "./services/api";

export default function App() {
  const { models, modelsDir, refresh } = useModels();
  const { result, isProcessing, error, transcribe } = useTranscribe();
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedBackend, setSelectedBackend] = useState<string>("xnnpack");
  const [dlOverlay, setDlOverlay] = useState<{
    label: string;
    status: string;
    progress: number;
    fileName?: string;
    fileIndex?: number;
    fileCount?: number;
    downloadedBytes?: number;
    totalBytes?: number;
  } | null>(null);

  const handleAudio = async (file: File) => {
    if (!selectedModel) return;
    await transcribe(file, selectedModel, selectedBackend);
  };

  const handleDownload = async (
    modelId: string,
    backend: string,
    displayName: string
  ) => {
    setDlOverlay({ label: displayName, status: "starting", progress: 0 });
    try {
      await downloadModel(modelId, backend, (event) => {
        setDlOverlay((prev) => ({
          label: prev?.label || displayName,
          status: event.status,
          progress: event.progress ?? prev?.progress ?? 0,
          fileName: event.file ?? prev?.fileName,
          fileIndex: event.file_index ?? prev?.fileIndex,
          fileCount: event.file_count ?? prev?.fileCount,
          downloadedBytes: event.downloaded_bytes ?? prev?.downloadedBytes,
          totalBytes: event.total_bytes ?? prev?.totalBytes,
        }));
      });
      await refresh();
    } catch (e) {
      console.error("Download failed", e);
    } finally {
      setDlOverlay(null);
    }
  };

  const handleDelete = async (modelId: string, backend: string) => {
    try {
      await deleteModel(modelId, backend);
      await refresh();
      if (selectedModel === modelId && selectedBackend === backend) {
        setSelectedModel(null);
      }
    } catch (e) {
      console.error("Delete failed", e);
    }
  };

  return (
    <div className="h-screen overflow-y-auto px-6 py-5 md:px-10">
      {dlOverlay && (
        <LoadingOverlay
          label={dlOverlay.label}
          status={dlOverlay.status}
          progress={dlOverlay.progress}
          fileName={dlOverlay.fileName}
          fileIndex={dlOverlay.fileIndex}
          fileCount={dlOverlay.fileCount}
          downloadedBytes={dlOverlay.downloadedBytes}
          totalBytes={dlOverlay.totalBytes}
        />
      )}

      <div className="min-h-full flex flex-col items-center justify-center">
        <div className="w-full max-w-4xl flex flex-col gap-4">
          <header className="text-center">
            <h1 className="text-3xl font-semibold tracking-tight text-gray-900">
              ExecuTorch Voice Studio
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              On-device speech AI — powered by ExecuTorch
            </p>
          </header>

          <div className="glass-panel p-5">
            <ModelSelector
              models={models}
              selected={selectedModel}
              selectedBackend={selectedBackend}
              onSelect={(id, backend) => {
                setSelectedModel(id);
                setSelectedBackend(backend);
              }}
              onRefresh={refresh}
              onDownload={handleDownload}
              onDelete={handleDelete}
            />
          </div>

          <div className="glass-panel p-5">
            <AudioInput
              onAudioReady={handleAudio}
              disabled={!selectedModel || isProcessing}
              isProcessing={isProcessing}
            />
          </div>

          {error && (
            <div className="glass-panel p-4 border-red-200 bg-red-50/40">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {result && (
            <div className="glass-panel p-5">
              <TranscriptionView result={result} />
            </div>
          )}
        </div>

        {modelsDir && (
          <p className="text-[10px] text-gray-500 text-center mt-6 select-all font-mono">
            {modelsDir}
          </p>
        )}
      </div>
    </div>
  );
}
