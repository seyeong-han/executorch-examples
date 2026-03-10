import { useCallback, useRef, useState } from "react";

interface Props {
  onAudioReady: (file: File) => void;
  disabled: boolean;
  isProcessing: boolean;
}

export function AudioInput({ onAudioReady, disabled, isProcessing }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!disabled) {
        setFileName(file.name);
        onAudioReady(file);
      }
    },
    [disabled, onAudioReady]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        handleFile(file);
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      console.error("Microphone access denied");
    }
  }, [handleFile]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
  }, []);

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-800 mb-3">Audio</h2>

      <div
        className={`
          rounded-2xl border-2 border-dashed transition-all py-8 px-6
          ${dragOver ? "border-blue-400 bg-blue-50/30" : "border-gray-300/50"}
          ${disabled ? "opacity-40 pointer-events-none" : ""}
        `}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        {isProcessing ? (
          <div className="text-center space-y-2">
            <div className="flex justify-center gap-1.5">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
            <p className="text-gray-500 text-sm">
              {fileName ? `Processing ${fileName}...` : "Processing audio..."}
            </p>
          </div>
        ) : (
          <div className="text-center space-y-3">
            <p className="text-gray-400 text-sm">
              {isRecording
                ? "Recording... click Stop to finish"
                : "Drop an audio file here, or use the controls below"}
            </p>
            <div className="flex justify-center gap-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-sm font-medium text-gray-600 px-5 py-2 rounded-full bg-white/50 hover:bg-white/80 border border-gray-200/50 transition-all shadow-sm"
              >
                Upload File
              </button>
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`text-sm font-medium px-5 py-2 rounded-full border transition-all shadow-sm ${
                  isRecording
                    ? "text-red-600 bg-red-50/80 border-red-200 hover:bg-red-100"
                    : "text-gray-600 bg-white/50 hover:bg-white/80 border-gray-200/50"
                }`}
              >
                {isRecording ? "Stop" : "Record"}
              </button>
            </div>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}
