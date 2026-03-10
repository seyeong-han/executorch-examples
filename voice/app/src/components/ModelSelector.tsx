import { useState } from "react";
import { ConfirmDialog } from "./ConfirmDialog";
import type { ModelInfo } from "../types";

interface Props {
  models: ModelInfo[];
  selected: string | null;
  selectedBackend: string;
  onSelect: (modelId: string, backend: string) => void;
  onRefresh: () => void;
  onDownload: (modelId: string, backend: string, displayName: string) => void;
  onDelete: (modelId: string, backend: string) => void;
}

const taskLabels: Record<string, string> = {
  transcription: "Speech-to-Text",
  diarization: "Speaker Diarization",
  vad: "Voice Activity Detection",
};

export function ModelSelector({
  models,
  selected,
  selectedBackend,
  onSelect,
  onRefresh,
  onDownload,
  onDelete,
}: Props) {
  const [deleteTarget, setDeleteTarget] = useState<{
    modelId: string;
    backend: string;
    displayName: string;
  } | null>(null);

  return (
    <div>
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Model"
        message={
          deleteTarget
            ? `Remove ${deleteTarget.displayName} (${deleteTarget.backend}) from this device? You can re-download it later.`
            : ""
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        destructive
        onConfirm={() => {
          if (deleteTarget) {
            onDelete(deleteTarget.modelId, deleteTarget.backend);
          }
          setDeleteTarget(null);
        }}
        onCancel={() => setDeleteTarget(null)}
      />

      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Models</h2>
        <button
          onClick={onRefresh}
          className="text-sm text-gray-500 hover:text-gray-800 transition-colors px-3 py-1 rounded-full hover:bg-black/5"
        >
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {models.map((m) => {
          const readyBackends = m.backends.filter(
            (b) => b.runner_available && b.model_downloaded
          );
          const bestBackend =
            readyBackends[0]?.name || m.backends[0]?.name || "xnnpack";
          const isSelected = selected === m.id;

          return (
            <button
              key={m.id}
              onClick={() =>
                onSelect(m.id, isSelected ? selectedBackend : bestBackend)
              }
              className={`
                text-left rounded-2xl p-4 transition-all
                ${
                  isSelected
                    ? "bg-white/60 ring-2 ring-gray-400/50 shadow-lg"
                    : "bg-white/30 hover:bg-white/50 shadow-sm"
                }
              `}
            >
              <div className="flex items-start justify-between mb-1">
                <h3 className="font-medium text-gray-900 text-sm">
                  {m.display_name}
                </h3>
                <span className="text-[11px] text-gray-400 font-mono">
                  {m.params}
                </span>
              </div>

              <p className="text-[11px] text-gray-400 mb-3">
                {taskLabels[m.task] || m.task}
              </p>

              <div className="flex gap-1.5 flex-wrap">
                {m.backends.map((b) => {
                  const ready = b.runner_available && b.model_downloaded;
                  const canDownload =
                    b.hf_repo && b.runner_available && !b.model_downloaded;
                  const isActive = isSelected && selectedBackend === b.name;

                  return (
                    <span
                      key={b.name}
                      className="group/badge relative"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (ready) {
                          onSelect(m.id, b.name);
                        } else if (canDownload) {
                          onDownload(m.id, b.name, m.display_name);
                        }
                      }}
                    >
                      <span
                        className={`
                          text-[11px] px-2 py-0.5 rounded-md cursor-pointer transition-all select-none
                          ${
                            ready
                              ? isActive
                                ? "bg-gray-800 text-white"
                                : "bg-gray-100 text-gray-600 group-hover/badge:bg-red-50 group-hover/badge:text-red-500"
                              : canDownload
                              ? "bg-gray-50 text-gray-400 hover:bg-gray-100"
                              : "bg-gray-50 text-gray-300"
                          }
                        `}
                      >
                        <span className={ready && !isActive ? "group-hover/badge:hidden" : ""}>
                          {ready
                            ? `${b.name} ✓`
                            : canDownload
                            ? `${b.name} ↓`
                            : b.name}
                        </span>
                        {ready && !isActive && (
                          <span
                            className="hidden group-hover/badge:inline"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteTarget({
                                modelId: m.id,
                                backend: b.name,
                                displayName: m.display_name,
                              });
                            }}
                          >
                            delete
                          </span>
                        )}
                      </span>
                    </span>
                  );
                })}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
