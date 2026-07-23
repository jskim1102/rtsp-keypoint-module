import { useEffect, useState } from "react";
import Modal from "./Modal";

/**
 * 카메라별 — 선택된 각 모델의 사람 단위 conf 설정 모달.
 *
 * pose 는 단일 class(person)라 class filter/색상 UI 가 없다 — conf 슬라이더만 유지한다.
 * Accordion: 한 번에 한 모델만 펼침. 모델 헤더 클릭 시 토글.
 *
 * conf 는 워커 추론 임계(사람 detection) → 변경 시 상위가 per-source conf_threshold PUT.
 */

export interface ModelSettings {
  conf?: number;
}

interface Props {
  open: boolean;
  onClose: () => void;
  cameraName: string;
  fallbackConf: number;
  selectedModels: string[];
  settings: Record<string, ModelSettings>;
  onSettingsChange: (next: Record<string, ModelSettings>) => void;
}

function ModelSettingsModal({
  open,
  onClose,
  cameraName,
  fallbackConf,
  selectedModels,
  settings,
  onSettingsChange,
}: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  // 모달 열릴 때 모델이 1개뿐이면 자동 펼침
  useEffect(() => {
    if (open && selectedModels.length === 1) setExpanded(selectedModels[0]);
  }, [open, selectedModels]);

  const updateModel = (model: string, patch: Partial<ModelSettings>) => {
    const prev = settings[model] ?? {};
    onSettingsChange({ ...settings, [model]: { ...prev, ...patch } });
  };

  return (
    <Modal open={open} onClose={onClose} title={`${cameraName} — 모델 설정`}>
      {selectedModels.length === 0 ? (
        <div style={styles.empty}>
          선택된 모델이 없습니다. 먼저 [모델] 에서 모델을 선택하세요.
        </div>
      ) : (
        <ul style={styles.list}>
          {selectedModels.map((model) => {
            const isExpanded = expanded === model;
            const ms = settings[model] ?? {};
            const conf = ms.conf ?? fallbackConf;
            const overridden = ms.conf !== undefined;
            return (
              <li key={model} style={styles.modelItem}>
                <button
                  style={styles.modelHeader}
                  onClick={() => setExpanded(isExpanded ? null : model)}
                >
                  <span style={styles.caret}>{isExpanded ? "▼" : "▶"}</span>
                  <span style={styles.modelName}>{model}</span>
                  <span style={styles.confBadge}>
                    {conf.toFixed(2)}
                    {!overridden && <span style={styles.fallback}> (기본)</span>}
                  </span>
                </button>
                {isExpanded && (
                  <div style={styles.modelBody}>
                    <ConfSection
                      conf={conf}
                      overridden={overridden}
                      onChange={(v) => updateModel(model, { conf: v })}
                      onReset={() => updateModel(model, { conf: undefined })}
                    />
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Modal>
  );
}

// ──────────────────────── Conf 슬라이더 ────────────────────────
function ConfSection({
  conf,
  overridden,
  onChange,
  onReset,
}: {
  conf: number;
  overridden: boolean;
  onChange: (v: number) => void;
  onReset: () => void;
}) {
  return (
    <div style={styles.confRow}>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={conf}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={styles.slider}
      />
      {overridden && (
        <button onClick={onReset} style={styles.resetBtn} title="기본값으로">
          기본값
        </button>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  empty: {
    textAlign: "center", color: "#aaaaaa", fontSize: "0.9rem", padding: "1.5rem 0.5rem",
  },
  list: {
    listStyle: "none", margin: 0, padding: 0,
    display: "flex", flexDirection: "column", gap: "0.5rem",
  },
  modelItem: { backgroundColor: "#16213e", borderRadius: "6px", overflow: "hidden" },
  modelHeader: {
    width: "100%", display: "flex", alignItems: "center", gap: "0.6rem",
    padding: "0.7rem 0.9rem", background: "transparent", border: "none",
    cursor: "pointer", color: "#ffffff", fontSize: "0.9rem", textAlign: "left",
  },
  caret: { color: "#aaaaaa", width: "14px" },
  modelName: { flex: 1 },
  confBadge: { color: "#4ade80", fontSize: "0.85rem", fontWeight: 600 },
  fallback: { color: "#888", fontWeight: 400, fontSize: "0.75rem" },
  modelBody: {
    padding: "0 0.9rem 0.9rem",
    display: "flex", flexDirection: "column", gap: "0.7rem",
  },
  confRow: { display: "flex", alignItems: "center", gap: "0.5rem" },
  slider: { flex: 1, accentColor: "#4caf50", cursor: "pointer" },
  resetBtn: {
    padding: "0.25rem 0.6rem", borderRadius: "4px",
    border: "1px solid #555", backgroundColor: "transparent",
    color: "#aaaaaa", fontSize: "0.7rem", cursor: "pointer",
  },
};

export default ModelSettingsModal;
