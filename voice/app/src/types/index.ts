export interface ModelInfo {
  id: string;
  display_name: string;
  task: "transcription" | "diarization" | "vad";
  params: string;
  backends: BackendInfo[];
}

export interface BackendInfo {
  name: string;
  runner_available: boolean;
  model_downloaded: boolean;
  hf_repo: string | null;
}

export interface TranscriptionResult {
  text: string;
  model: string;
  backend: string;
  duration: number;
  segments: Segment[];
  performance: PerformanceStats;
}

export interface Segment {
  start: number;
  end: number;
  text?: string;
  speaker?: string;
}

export interface DiarizationResult {
  segments: Segment[];
  model: string;
  backend: string;
  duration: number;
  num_speakers: number;
  performance: PerformanceStats;
}

export interface VadResult {
  segments: Segment[];
  model: string;
  backend: string;
  duration: number;
  speech_ratio: number;
  performance: PerformanceStats;
}

export interface PerformanceStats {
  inference_time_s: number;
  tokens_generated?: number;
  tokens_per_second?: number;
}

export interface DownloadProgress {
  model_id: string;
  backend: string;
  status: "pending" | "downloading" | "complete" | "error";
  progress: number;
  error?: string;
}
