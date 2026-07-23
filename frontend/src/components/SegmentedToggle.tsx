/**
 * 좌/우 두 개 옵션 사이 슬라이딩 토글.
 * 활성화된 쪽으로 초록 thumb 가 부드럽게 이동, thumb 위 라벨은 흰색.
 *
 * 사용:
 *   <SegmentedToggle
 *     enabled={state}
 *     onChange={setState}
 *     leftLabel="ON"
 *     rightLabel="OFF"
 *   />
 *
 *  enabled=true → 왼쪽(ON), false → 오른쪽(OFF).
 */

interface Props {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  leftLabel?: string;
  rightLabel?: string;
  width?: number;
  height?: number;
  disabled?: boolean;
}

function SegmentedToggle({
  enabled,
  onChange,
  leftLabel = "ON",
  rightLabel = "OFF",
  width = 92,
  height = 32,
  disabled = false,
}: Props) {
  const thumbWidth = width / 2 - 2;

  return (
    <div
      style={{
        position: "relative",
        width,
        height,
        backgroundColor: "#1a2a3e",
        borderRadius: height / 2,
        display: "flex",
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
        userSelect: "none",
      }}
    >
      {/* 슬라이딩 thumb */}
      <div
        style={{
          position: "absolute",
          top: 2,
          left: enabled ? 2 : width / 2,
          width: thumbWidth,
          height: height - 4,
          borderRadius: (height - 4) / 2,
          backgroundColor: "#4caf50",
          transition: "left 0.18s ease",
          pointerEvents: "none",
        }}
      />
      {/* ON 클릭 영역 */}
      <span
        onClick={(e) => {
          e.stopPropagation();
          if (!disabled) onChange(true);
        }}
        style={{
          flex: 1,
          textAlign: "center",
          lineHeight: `${height}px`,
          zIndex: 1,
          fontSize: "0.82rem",
          fontWeight: 600,
          color: enabled ? "#ffffff" : "#7a7a8a",
          transition: "color 0.18s",
        }}
      >
        {leftLabel}
      </span>
      {/* OFF 클릭 영역 */}
      <span
        onClick={(e) => {
          e.stopPropagation();
          if (!disabled) onChange(false);
        }}
        style={{
          flex: 1,
          textAlign: "center",
          lineHeight: `${height}px`,
          zIndex: 1,
          fontSize: "0.82rem",
          fontWeight: 600,
          color: !enabled ? "#ffffff" : "#7a7a8a",
          transition: "color 0.18s",
        }}
      >
        {rightLabel}
      </span>
    </div>
  );
}

export default SegmentedToggle;
