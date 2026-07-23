import { useEffect } from "react";

/**
 * 일반 모달 컴포넌트.
 *
 * - 모바일: 좁은 화면에선 90vw 차지 (모바일 친화)
 * - 데스크톱: 가운데 정렬, 기본 max-width 480px
 * - ESC 키 / 외부 클릭 / ✕ 버튼으로 닫힘
 */

interface Props {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  maxWidth?: number;
}

function Modal({ open, onClose, title, children, maxWidth = 480 }: Props) {
  // ESC 닫기 + body 스크롤 잠금
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div style={styles.backdrop} onClick={onClose}>
      <div
        style={{ ...styles.container, maxWidth }}
        onClick={(e) => e.stopPropagation()}
      >
        <header style={styles.header}>
          <h2 style={styles.title}>{title}</h2>
          <button onClick={onClose} style={styles.closeBtn} aria-label="close">
            ✕
          </button>
        </header>
        <div style={styles.body}>{children}</div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  backdrop: {
    position: "fixed",
    inset: 0,
    backgroundColor: "rgba(0, 0, 0, 0.65)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
    padding: "1rem",
  },
  container: {
    width: "100%",
    maxHeight: "90vh",
    backgroundColor: "#1a1a2e",
    color: "#ffffff",
    borderRadius: "12px",
    boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0.9rem 1.1rem",
    borderBottom: "1px solid #2a2a3e",
  },
  title: {
    margin: 0,
    fontSize: "1.05rem",
    fontWeight: 600,
  },
  closeBtn: {
    background: "transparent",
    border: "none",
    color: "#aaaaaa",
    fontSize: "1.2rem",
    cursor: "pointer",
    padding: "0.3rem 0.5rem",
    lineHeight: 1,
  },
  body: {
    padding: "1rem 1.1rem",
    overflowY: "auto",
  },
};

export default Modal;
