import { useEffect, useRef, useState } from "react";
import {
  KPT_COLOR_IDX,
  SKELETON_EDGES,
  kptColor,
  limbColor,
} from "../utils/colors";

/**
 * WHEP `<video>` 위에 절대 위치 `<canvas>` 로 **스켈레톤 keypoint** 오버레이. ← 하이브리드 SEAM
 *
 * 부모 BboxOverlay(rect draw)를 pose 로 스왑한 것. 영상(WHEP)과 추론 프레임(백엔드 별도 캡처)이
 * **서로 다른 한 장**이라 두 좌표계를 정합하는 SEAM 스케일 로직은 그대로 계승한다:
 *
 * - canvas internal width/height = 부모 `<video>` 의 natural 크기(video.videoWidth/Height)
 *   — `loadedmetadata` · `resize` 이벤트로 갱신(WHEP 는 srcObject 라 `load` 가 없음).
 * - keypoints 좌표는 추론 캡처 프레임(frameW×frameH) 좌표계 → `videoNatural / detFrame`
 *   스케일로 그린다. **두 해상도가 같으면 sx=sy=1 (identity), 다르면 자동 보정.**
 *   (frameW/H 미상=0 이면 identity 가정.)
 * - canvas 는 video 와 동일 박스에 절대배치(inset:0, 100%)하되 `object-fit:contain` 으로
 *   video 와 똑같이 레터/필러박싱 → 비 16:9 카메라(예 4:3)에서도 화면상 정렬.
 *
 * pose 는 단일 class(person)라 class filter/색상 override 가 없다 → settings prop 없음.
 * 사람 단위 conf 는 서버(워커)에서 이미 적용됨. per-keypoint viz 임계는 아래 상수.
 *
 * 새 pose 결과가 도착할 때 최신 스켈레톤을 그대로 그리고 다음 결과까지 유지한다.
 * 빈 결과가 오면 canvas를 즉시 비워 사라진 사람을 유령 스켈레톤으로 남기지 않는다.
 */

// per-keypoint 표시 임계 — conf 미만 관절 점/엣지 미표시 (spec Unknowns: 0.5 시작, gate2 튜닝 knob).
const KPT_CONF_THRESHOLD = 0.5;

// 사람 1명 = 17 COCO keypoint, 각 점 = [x, y, conf] (추론 캡처 프레임 좌표계).
export interface KeypointPerson {
  keypoints: [number, number, number][];
  model: string;
}

interface Props {
  // 부모 WhepPlayer 의 <video> ref — 이 위에 절대배치 canvas 를 겹친다.
  videoRef: React.RefObject<HTMLVideoElement | null>;
  detections: KeypointPerson[];
  // YOLO 가 본 추론 캡처 프레임 치수 (WS frame:{w,h}). 0 이면 스케일 = identity.
  frameW: number;
  frameH: number;
}

function KeypointOverlay({
  videoRef,
  detections,
  frameW,
  frameH,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  // video natural 크기 — loadedmetadata + resize 시 갱신.
  // WHEP 는 srcObject(MediaStream) 라 <img> 의 onload 가 없다. 트랙 해상도 확정/변경은
  // video 의 'resize'(intrinsic 크기 변동) + 'loadedmetadata' 로 통지된다.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const update = () => {
      if (video.videoWidth && video.videoHeight) {
        setSize((prev) => {
          if (prev?.w === video.videoWidth && prev?.h === video.videoHeight) return prev;
          return { w: video.videoWidth, h: video.videoHeight };
        });
      }
    };
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("resize", update);
    update(); // 이미 메타데이터 로드됐을 수 있음
    return () => {
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("resize", update);
    };
  }, [videoRef]);

  // 최신 pose 결과를 한 번 그린다. 다음 결과가 올 때까지 canvas 픽셀을 유지한다.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (detections.length === 0) return;

    // SEAM 좌표 스케일 — keypoint(추론 프레임 좌표계) → canvas internal(video natural).
    // 두 해상도 같으면 identity. (frameW/H=0 이면 identity 가정.)
    const sx = frameW > 0 ? size.w / frameW : 1;
    const sy = frameH > 0 ? size.h / frameH : 1;

    // 원본 좌표계 기준 적정 사이즈 (해상도 클수록 선 두께/점 반경 비율 보정)
    const scale = Math.max(1, Math.min(size.w, size.h) / 600);
    const lineWidth = Math.max(1.5, 2 * scale);
    const pointRadius = Math.max(2, 3 * scale);

    ctx.lineWidth = lineWidth;
    for (const person of detections) {
      const keypoints = person.keypoints;
      if (!keypoints || keypoints.length < KPT_COLOR_IDX.length) continue;

      // 두 끝점 모두 신뢰도 임계 이상일 때만 스켈레톤 엣지를 그린다.
      for (
        let edgeIndex = 0;
        edgeIndex < SKELETON_EDGES.length;
        edgeIndex++
      ) {
        const [a, b] = SKELETON_EDGES[edgeIndex];
        const pointA = keypoints[a];
        const pointB = keypoints[b];
        if (
          pointA[2] < KPT_CONF_THRESHOLD ||
          pointB[2] < KPT_CONF_THRESHOLD
        ) {
          continue;
        }
        ctx.strokeStyle = limbColor(edgeIndex);
        ctx.beginPath();
        ctx.moveTo(pointA[0] * sx, pointA[1] * sy);
        ctx.lineTo(pointB[0] * sx, pointB[1] * sy);
        ctx.stroke();
      }

      for (
        let pointIndex = 0;
        pointIndex < KPT_COLOR_IDX.length;
        pointIndex++
      ) {
        const point = keypoints[pointIndex];
        if (point[2] < KPT_CONF_THRESHOLD) continue;
        ctx.fillStyle = kptColor(pointIndex);
        ctx.beginPath();
        ctx.arc(
          point[0] * sx,
          point[1] * sy,
          pointRadius,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
    }
  }, [detections, frameH, frameW, size]);

  // video 메타데이터가 아직이면(size 미상) 그릴 대상 없음 — 오버레이 생략.
  if (!size) return null;

  return (
    <canvas
      ref={canvasRef}
      width={size.w}
      height={size.h}
      style={canvasStyle}
    />
  );
}

const canvasStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "contain",
  pointerEvents: "none",
};

export default KeypointOverlay;
