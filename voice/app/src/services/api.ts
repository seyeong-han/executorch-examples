import axios from "axios";
import type {
  ModelInfo,
  TranscriptionResult,
  DiarizationResult,
  VadResult,
} from "../types";

const api = axios.create({
  baseURL: "/v1",
  timeout: 300_000,
});

export async function fetchModels(): Promise<{ models: ModelInfo[]; modelsDir: string }> {
  const { data } = await api.get("/models");
  return { models: data.data, modelsDir: data.models_dir || "" };
}

export async function downloadModel(
  modelId: string,
  backend: string,
  onProgress?: (event: {
    status: string;
    progress: number;
    file?: string;
    file_index?: number;
    file_count?: number;
    downloaded_bytes?: number;
    total_bytes?: number;
  }) => void
): Promise<void> {
  const response = await fetch(`/v1/models/${modelId}/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backend }),
  });

  if (!response.ok) {
    throw new Error(`Download failed: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6));
          onProgress?.(event);
        } catch {
          // skip malformed events
        }
      }
    }
  }
}

export async function deleteModel(
  modelId: string,
  backend: string
): Promise<void> {
  await api.delete(`/models/${modelId}`, { params: { backend } });
}

export async function transcribe(
  file: File,
  model: string,
  backend?: string
): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  if (backend) form.append("backend", backend);
  const { data } = await api.post("/audio/transcriptions", form);
  return data;
}

export async function diarize(
  file: File,
  model: string = "sortformer",
  backend?: string
): Promise<DiarizationResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  if (backend) form.append("backend", backend);
  const { data } = await api.post("/audio/diarizations", form);
  return data;
}

export async function detectVad(
  file: File,
  model: string = "silero-vad",
  backend?: string
): Promise<VadResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  if (backend) form.append("backend", backend);
  const { data } = await api.post("/audio/vad", form);
  return data;
}
