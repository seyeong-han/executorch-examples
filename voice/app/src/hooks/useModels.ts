import { useCallback, useEffect, useState } from "react";
import { fetchModels, downloadModel as apiDownload } from "../services/api";
import type { ModelInfo } from "../types";

export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsDir, setModelsDir] = useState<string>("");
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const { models: data, modelsDir: dir } = await fetchModels();
      setModels(data);
      if (dir) setModelsDir(dir);
    } catch (e) {
      console.error("Failed to fetch models", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const download = useCallback(
    async (modelId: string, backend: string) => {
      await apiDownload(modelId, backend);
      await refresh();
    },
    [refresh]
  );

  return { models, modelsDir, loading, refresh, download };
}
