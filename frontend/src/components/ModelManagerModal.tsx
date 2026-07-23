import { useEffect, useState, useCallback } from "react";
import Modal from "./Modal";
import { apiBase } from "../hooks/useApi";

const PRESET_LABELS: Record<string, string> = {
  "yolo26n-pose.pt": "yolo26n-pose (nano · 가장 빠름)",
  "yolo26s-pose.pt": "yolo26s-pose (small · 균형)",
  "yolo26m-pose.pt": "yolo26m-pose (medium)",
  "yolo26l-pose.pt": "yolo26l-pose (large)",
  "yolo26x-pose.pt": "yolo26x-pose (xlarge · 가장 정확)",
};

interface ModelInfo {
  name: string;
  type: "preset";
  size_mb: number | null;
}

interface Props {
  open: boolean;
  onClose: () => void;
  cameraName: string;
  // 현재 선택된 모델 목록 (즉시 갱신 받기 위해 부모에서 관리)
  selected: string[];
  onSelectedChange: (models: string[]) => void;
}

/**
 * 카메라별 사용 모델 선택 모달 (preset-only).
 *
 * custom .pt 업로드/삭제는 제거됐다 — .pt=pickle=코드실행이라 인증 없는 업로드가 RCE 였다(codex #1).
 * preset(YOLO26 5종)만 선택한다. 체크박스 토글 → 즉시 부모에 새 list 전달(부모가 PUT 처리).
 */
function ModelManagerModal({
  open,
  onClose,
  cameraName,
  selected,
  onSelectedChange,
}: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState("");

  const fetchModels = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase()}/api/inference/models`);
      const data: ModelInfo[] = await r.json();
      setModels(data);
    } catch (e) {
      setError(`목록 로드 실패: ${e}`);
    }
  }, []);

  useEffect(() => {
    if (open) {
      fetchModels();
      setError("");
    }
  }, [open, fetchModels]);

  const toggle = (name: string) => {
    const next = selected.includes(name)
      ? selected.filter((n) => n !== name)
      : [...selected, name];
    onSelectedChange(next);
  };

  return (
    <Modal open={open} onClose={onClose} title={`${cameraName} — 모델 관리`}>
      {/* 사용 모델 선택 (preset only) */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          사용 모델 선택 — {selected.length} 개 활성
        </div>
        <ul style={styles.modelList}>
          {models.map((m) => (
            <ModelRow
              key={m.name}
              checked={selected.includes(m.name)}
              onToggle={() => toggle(m.name)}
              primaryText={m.name}
              secondaryText={PRESET_LABELS[m.name] ?? "(preset)"}
            />
          ))}
        </ul>
      </div>

      {error && <div style={styles.error}>{error}</div>}
    </Modal>
  );
}

interface RowProps {
  checked: boolean;
  onToggle: () => void;
  primaryText: string;
  secondaryText?: string;
}

function ModelRow({ checked, onToggle, primaryText, secondaryText }: RowProps) {
  return (
    <li style={styles.row}>
      <label style={styles.rowLabel}>
        <input
          type="checkbox"
          checked={checked}
          onChange={onToggle}
          style={styles.checkbox}
        />
        <span style={styles.rowName}>{primaryText}</span>
        {secondaryText && (
          <span style={styles.rowSecondary}>— {secondaryText}</span>
        )}
      </label>
    </li>
  );
}

const styles: Record<string, React.CSSProperties> = {
  section: {
    marginBottom: "1rem",
  },
  sectionTitle: {
    fontSize: "0.85rem",
    color: "#aaaaaa",
    marginBottom: "0.5rem",
  },
  modelList: {
    listStyle: "none",
    margin: 0,
    padding: 0,
    backgroundColor: "#16213e",
    borderRadius: "6px",
    overflow: "hidden",
  },
  row: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0.6rem 0.8rem",
    borderBottom: "1px solid #1a1a2e",
    minHeight: "44px", // 모바일 탭 타겟 확보
  },
  rowLabel: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    flex: 1,
    cursor: "pointer",
    fontSize: "0.9rem",
  },
  checkbox: {
    width: "18px",
    height: "18px",
    accentColor: "#4caf50",
    cursor: "pointer",
  },
  rowName: {
    fontFamily: "monospace",
    color: "#ffffff",
  },
  rowSecondary: {
    color: "#aaaaaa",
    fontSize: "0.8rem",
  },
  error: {
    marginTop: "0.5rem",
    color: "#f87171",
    fontSize: "0.85rem",
  },
};

export default ModelManagerModal;
