interface Props {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  onConfirm,
  onCancel,
}: Props) {
  if (!open) return null;

  return (
    <div
      className="rounded-[20px] fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      onClick={onCancel}
    >
      <div
        className="rounded-[20px] bg-white/90 backdrop-blur-xl shadow-2xl w-[280px] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-6 pt-6 pb-4 text-center">
          <h3 className="text-[15px] font-semibold text-gray-900">{title}</h3>
          <p className="text-[13px] text-gray-500 mt-1.5">{message}</p>
        </div>

        <div className="border-t border-gray-200/80">
          <button
            onClick={onCancel}
            className="w-full py-3 text-[15px] text-blue-500 font-normal border-b border-gray-200/80 hover:bg-gray-50 transition-colors"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className={`w-full py-3 text-[15px] font-semibold hover:bg-gray-50 transition-colors ${
              destructive ? "text-red-500" : "text-blue-500"
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
