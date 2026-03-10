import { useCallback, useState } from "react";
import { transcribe as apiTranscribe } from "../services/api";
import type { TranscriptionResult } from "../types";

export function useTranscribe() {
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const transcribe = useCallback(
    async (file: File, model: string, backend?: string) => {
      setIsProcessing(true);
      setError(null);
      try {
        const data = await apiTranscribe(file, model, backend);
        setResult(data);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Transcription failed";
        setError(msg);
      } finally {
        setIsProcessing(false);
      }
    },
    []
  );

  return { result, isProcessing, error, transcribe };
}
