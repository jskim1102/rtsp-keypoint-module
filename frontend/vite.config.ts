import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 0.0.0.0 — 전 인터페이스 (§5.3)
    allowedHosts: true, // 외부 도메인/IP/터널 Host 헤더 허용
    strictPort: true, // 포트 점유 시 자동 +1 점프 금지 — 할당 포트 침범 방지 (RULES §4)
  },
});
