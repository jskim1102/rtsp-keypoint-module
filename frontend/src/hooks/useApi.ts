const API_PORT = import.meta.env.VITE_API_PORT || "8000";
const WEBRTC_PORT = import.meta.env.VITE_MEDIAMTX_WEBRTC_PORT || "8889";
// mediamtx 인증(#100) — 외부 브라우저 WHEP 시청용 viewer 자격증명(빌드타임 인라인).
const VIEWER_USER = import.meta.env.VITE_MEDIAMTX_VIEWER_USER || "";
const VIEWER_PASS = import.meta.env.VITE_MEDIAMTX_VIEWER_PASS || "";

export function apiBase(): string {
  return `http://${window.location.hostname}:${API_PORT}`;
}

// mediamtx WHEP endpoint base — 브라우저가 mediamtx 에 직접 접속(백엔드 경유 X).
export function whepBase(): string {
  return `http://${window.location.hostname}:${WEBRTC_PORT}`;
}

// WHEP 요청 인증 헤더(#100) — viewer Basic. 미설정이면 빈 객체(무인증, 하위호환).
export function whepAuthHeaders(): Record<string, string> {
  if (!VIEWER_PASS) return {};
  return { Authorization: "Basic " + btoa(`${VIEWER_USER}:${VIEWER_PASS}`) };
}
