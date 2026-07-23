import { useEffect, useRef, useState } from "react";
import { apiBase } from "./useApi";
import type { KeypointPerson } from "../components/KeypointOverlay";

/**
 * 카메라별 detection 좌표 WS 구독 (좌표 전용 — 하이브리드).
 *
 * deepeye 원본 WS 는 raw JPEG(binary) + detections(text) 둘 다 받았지만, 여기선 영상이
 * WHEP `<video>` 로 따로 오므로 이 WS 는 **text(detections JSON) 만** 파싱한다(binary 무시).
 *
 *   ws://<host>:<VITE_API_PORT>/api/ipcams/{streamKey}/ws
 *   ← { type:"detections", items:[...], frame:{w,h} }
 *
 * - `active=false` 면 연결하지 않고(백엔드 캡처 안 띄움) items 를 즉시 비운다.
 * - `active=true` 면 연결, 끊기면 2s 후 자동 reconnect.
 * - frame{w,h} = YOLO 가 본 추론 캡처 프레임 치수(SEAM 좌표 스케일용). 없으면 0(=identity).
 */

const RECONNECT_DELAY = 2000;
const DET_FPS_WINDOW_MS = 2000; // det fps 측정 윈도우(~2초).

export interface DetectionStream {
  items: KeypointPerson[];
  frameW: number;
  frameH: number;
  detFps: number;
}

export function useDetectionWs(streamKey: string, active: boolean): DetectionStream {
  const [items, setItems] = useState<KeypointPerson[]>([]);
  const [frame, setFrame] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  // detection 메시지 도착률(det fps) — 추론 ON 일 때만 백엔드가 메시지를 보냄(OFF=0).
  const [detFps, setDetFps] = useState(0);
  // 최근 도착 타임스탬프(performance.now()) 윈도우 — 렌더 무관하게 ref 로 관리.
  const detArrivalsRef = useRef<number[]>([]);

  useEffect(() => {
    // 추론 OFF / 미연결 — WS 안 열고(백엔드 capture 미기동) 잔존 detection·측정치 즉시 정리.
    if (!active) {
      setItems([]);
      detArrivalsRef.current = [];
      setDetFps(0);
      return;
    }

    let unmounted = false;
    let reconnectTimer: number | null = null;
    let ws: WebSocket | null = null;

    // det fps 갱신: 윈도우 밖 타임스탬프를 버리고 (윈도우 내 개수)/(윈도우초) 계산.
    const detFpsTimer = window.setInterval(() => {
      if (unmounted) return;
      const cutoff = performance.now() - DET_FPS_WINDOW_MS;
      const arr = detArrivalsRef.current.filter((t) => t >= cutoff);
      detArrivalsRef.current = arr;
      setDetFps(arr.length / (DET_FPS_WINDOW_MS / 1000));
    }, 1000);
    // ws URL = apiBase() 의 http→ws 치환 (동일 host:VITE_API_PORT).
    const wsUrl = `${apiBase().replace(/^http/, "ws")}/api/ipcams/${streamKey}/ws`;

    const connect = () => {
      if (unmounted) return;
      ws = new WebSocket(wsUrl);

      ws.onmessage = (event: MessageEvent) => {
        if (unmounted) return;
        // 슬림 — text(detections JSON) 만. binary 프레임은 오지 않음(WHEP 가 영상 담당).
        if (typeof event.data !== "string") return;
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "detections") {
            // 도착 시각 기록 → det fps 측정(추론 ON 일 때만 메시지가 옴).
            detArrivalsRef.current.push(performance.now());
            setItems((msg.items ?? []) as KeypointPerson[]);
            // frame{w,h} 동봉 시 갱신(SEAM). 없으면 0 유지 → KeypointOverlay identity.
            if (msg.frame) {
              setFrame({ w: msg.frame.w ?? 0, h: msg.frame.h ?? 0 });
            }
          }
        } catch {
          /* malformed — 무시 */
        }
      };

      ws.onclose = () => {
        if (unmounted) return;
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY);
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      unmounted = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      clearInterval(detFpsTimer);
      detArrivalsRef.current = [];
      setDetFps(0);
      if (ws) ws.close();
    };
  }, [streamKey, active]);

  return { items, frameW: frame.w, frameH: frame.h, detFps };
}
