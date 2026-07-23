FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_PORT
RUN test -n "$VITE_API_PORT" || (echo "VITE_API_PORT is required" && exit 1)
# Vite 가 import.meta.env.VITE_MEDIAMTX_WEBRTC_PORT 를 빌드 시 정적 인라인 → ARG 선언 필수.
# 미선언이면 compose 가 넘긴 build arg 가 무시되고 whepBase() 기본값(8889)으로 굳음(P1-2).
ARG VITE_MEDIAMTX_WEBRTC_PORT
RUN test -n "$VITE_MEDIAMTX_WEBRTC_PORT" || (echo "VITE_MEDIAMTX_WEBRTC_PORT is required" && exit 1)
# WHEP viewer 인증(#100) — Vite 가 빌드타임 인라인하므로 ARG 선언 필수
# (미선언이면 compose build arg 가 무시돼 viewer 비번이 "" → WHEP 무인증 → auth ON mediamtx 가 401).
ARG VITE_MEDIAMTX_VIEWER_USER
ARG VITE_MEDIAMTX_VIEWER_PASS
RUN test -n "$VITE_MEDIAMTX_VIEWER_PASS" || (echo "VITE_MEDIAMTX_VIEWER_PASS is required" && exit 1)
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
