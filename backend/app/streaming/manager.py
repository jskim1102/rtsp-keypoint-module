"""모든 영상 소스 + 모델별 InferenceWorker pool의 통합 매니저.

- 캡처 스레드는 source별 최신 frame을 모델 lane에 제출
- 같은 모델은 micro-batch, 다른 모델은 별도 process로 병렬 실행
- pool 결과는 dispatch 스레드가 source_id 별로 캐싱
- 캡처 스레드는 캐시에서 자기 source_id 의 최신 결과만 조회

rtsp-keypoint: JPEG/get_frame 경로 제거(영상은 mediamtx WHEP). detections_to_json
은 `frame:{w,h}`(SEAM) 동봉 — frontend KeypointOverlay 좌표 스케일용.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import cv2

from app.config import (
    ADAPTIVE_DOWNSHIFT_TICKS,
    ADAPTIVE_OVERLOAD_RATIO,
    ADAPTIVE_UNDERLOAD_RATIO,
    ADAPTIVE_UPSHIFT_TICKS,
    AUTOTUNE_EWMA_ALPHA,
    AUTOTUNE_HEADROOM,
    AUTOTUNE_MIN_SAMPLES,
    AUTOTUNE_TARGET_FPS_MAX,
    INFERENCE_IMGSZ_STAGES,
    INFERENCE_INTERVAL,
    MAX_INFER_PER_SEC,
    MAX_INFERENCE_INTERVAL,
    MIN_INFERENCE_INTERVAL,
)
from app.inference import FrameRequest, InferenceResult, InferenceWorker
from app.streaming.capture import SourceType, VideoCaptureThread

logger = logging.getLogger("rtsp-keypoint.streaming.manager")

INFERENCE_IMGSZ = INFERENCE_IMGSZ_STAGES[-1]
GPU_UTIL_TARGET_DEFAULT = 0.85
GPU_UTIL_TARGET_MIN = 0.1
GPU_UTIL_SCALE_MIN = 0.1
GPU_UTIL_DEADBAND = 0.03
GPU_UTIL_RECOVERY_STEP = 0.05


@dataclass
class InferenceTelemetry:
    """worker service/idle 시간의 bounded EWMA 집계."""

    alpha: float = AUTOTUNE_EWMA_ALPHA
    min_samples: int = AUTOTUNE_MIN_SAMPLES
    infer_ms_ewma: float = 0.0
    duty_ewma: float = 0.0
    samples: int = 0

    def observe(self, result: InferenceResult) -> bool:
        infer_ms = float(getattr(result, "infer_ms", 0.0))
        idle_ms = float(getattr(result, "idle_ms", 0.0))
        cycle_ms = infer_ms + idle_ms
        # worker가 0으로 표시한 모델별 첫 추론(warm-up)과 비정상 표본은 버린다.
        if infer_ms <= 0.0 or cycle_ms <= 0.0:
            return False
        duty = infer_ms / cycle_ms
        if self.samples == 0:
            self.infer_ms_ewma = infer_ms
            self.duty_ewma = duty
        else:
            self.infer_ms_ewma += self.alpha * (infer_ms - self.infer_ms_ewma)
            self.duty_ewma += self.alpha * (duty - self.duty_ewma)
        self.samples += 1
        return True

    @property
    def ready(self) -> bool:
        return self.samples >= self.min_samples and self.infer_ms_ewma > 0.0

    @property
    def capacity_fps(self) -> float:
        return 1000.0 / self.infer_ms_ewma if self.infer_ms_ewma > 0.0 else 0.0


@dataclass
class SourceTelemetry:
    """source별 bounded scalar 계측. source 제거 때 dict에서도 함께 제거한다."""

    submitted: int = 0
    received: int = 0
    last_result_ts: float = 0.0
    infer_ms_ewma: float = 0.0
    infer_samples: int = 0

    def observe(self, result: InferenceResult, *, alpha: float) -> None:
        self.received += 1
        self.last_result_ts = result.timestamp
        infer_ms = float(result.infer_ms)
        if infer_ms <= 0.0:
            return
        if self.infer_samples == 0:
            self.infer_ms_ewma = infer_ms
        else:
            self.infer_ms_ewma += alpha * (infer_ms - self.infer_ms_ewma)
        self.infer_samples += 1


def _allocate_fps(
    wants: dict[str, float],
    *,
    budget_fps: float,
    floor_fps: float,
    cost_weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """가중 총량을 budget 아래로 유지하며 want 비율과 가능한 floor를 보존한다.

    n*floor > budget이면 두 계약은 동시에 성립하지 않는다. GPU service 총량 제한을
    우선하고 budget/n을 effective floor로 쓴다. cost_weights는 global infer_ms 대비
    source service cost 비율이다.
    """
    if not wants:
        return {}
    budget = max(0.0, float(budget_fps))
    weights = {
        sid: max(1e-9, float((cost_weights or {}).get(sid, 1.0)))
        for sid in wants
    }
    weight_sum = sum(weights.values())
    effective_floor = min(
        max(0.0, float(floor_fps)),
        budget / weight_sum if weight_sum > 0.0 else 0.0,
    )
    targets = {sid: max(effective_floor, float(want)) for sid, want in wants.items()}
    if sum(targets[sid] * weights[sid] for sid in targets) <= budget:
        return targets

    remaining = max(0.0, budget - effective_floor * weight_sum)
    extra_demand = sum(
        max(0.0, targets[sid] - effective_floor) * weights[sid]
        for sid in targets
    )
    if extra_demand <= 0.0:
        return {sid: effective_floor for sid in targets}
    return {
        sid: effective_floor
        + remaining * max(0.0, target - effective_floor) / extra_demand
        for sid, target in targets.items()
    }


def _gpu_util_scale_step(
    current: float,
    *,
    target: float,
    duty: float,
) -> float:
    """GPU duty ceiling controller 한 tick.

    초과는 즉시 비율 감속하고, 여유는 5%p씩 복구하며, ±3% deadband에서
    cadence 진동을 막는다. 100% 목표는 기존 최대성능 경로와 정확히 동일하다.
    """
    target = max(GPU_UTIL_TARGET_MIN, min(1.0, float(target)))
    if target >= 1.0:
        return 1.0
    current = max(GPU_UTIL_SCALE_MIN, min(1.0, float(current)))
    duty = max(0.0, min(1.0, float(duty)))
    if duty > target + GPU_UTIL_DEADBAND:
        proportional = current * target / max(duty, 1e-9)
        return max(GPU_UTIL_SCALE_MIN, min(current, proportional))
    if duty < target - GPU_UTIL_DEADBAND:
        return min(1.0, current + GPU_UTIL_RECOVERY_STEP)
    return current


def _resize_for_inference(frame: np.ndarray, imgsz: int = INFERENCE_IMGSZ) -> np.ndarray:
    """추론 제출 전 프레임을 모델 입력크기로 다운스케일 — mp.Queue IPC pickle 비용 절감
    (1080p 6.2MB/27ms → 640 691KB/1.6ms). 종횡비 보존, 업스케일 금지. YOLO 가 어차피
    내부 letterbox 로 imgsz 축소하므로 정확도 동등. 좌표 SEAM(frame_w/h=req.frame.shape)이
    자동으로 축소 치수를 따라가 프론트 KeypointOverlay가 video.videoWidth/frame.w로 스케일업."""
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    if scale >= 1.0:
        return frame
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _adaptive_imgsz_step(
    current: int,
    *,
    pressure_ticks: int,
    relief_ticks: int,
    allocation_ratio: float,
    queue_dropped: bool,
    underloaded: bool = False,
) -> tuple[int, int, int, bool]:
    """한 source의 imgsz hysteresis를 한 tick 진행한다.

    처리량이 목표의 85% 아래이거나 latest queue drop이 생기면 두 tick 후 한 단계
    낮춘다. 충분한 처리량과 전역 여유가 다섯 tick 지속될 때만 한 단계 복구해
    경계에서 해상도가 흔들리지 않게 한다.
    """
    stages = INFERENCE_IMGSZ_STAGES
    try:
        index = stages.index(int(current))
    except ValueError:
        index = len(stages) - 1
        current = stages[index]

    overloaded = queue_dropped or allocation_ratio < ADAPTIVE_OVERLOAD_RATIO
    relieved = (
        not queue_dropped
        and underloaded
        and allocation_ratio >= 0.98
    )
    if overloaded:
        pressure_ticks += 1
        relief_ticks = 0
        if pressure_ticks >= ADAPTIVE_DOWNSHIFT_TICKS and index > 0:
            return stages[index - 1], 0, 0, True
        return current, min(pressure_ticks, ADAPTIVE_DOWNSHIFT_TICKS), relief_ticks, False
    if relieved:
        pressure_ticks = 0
        relief_ticks += 1
        if relief_ticks >= ADAPTIVE_UPSHIFT_TICKS and index < len(stages) - 1:
            return stages[index + 1], 0, 0, True
        return current, pressure_ticks, min(relief_ticks, ADAPTIVE_UPSHIFT_TICKS), False
    return current, 0, 0, False


class StreamManager:
    """비디오 소스(웹캠/IP CAM) + 추론 워커 통합 관리.

    싱글톤으로 사용 (`from app.streaming.manager import manager`).
    서버 lifespan 시작/종료에서 `startup()` / `shutdown()` 호출.
    """

    # dispatch 루프의 idle sleep
    _DISPATCH_IDLE_SEC = 0.01

    # H1 tombstone TTL(초): 최근 삭제된 source_id 를 이 시간 동안 기억해 resurrect 를 막는다.
    # delete_ipcam(commit+mediamtx round-trip)~WS start_capture 경합 창(~1 RTT)보다 넉넉히 크면 된다.
    # stream_key 는 재사용되지 않으므로 만료돼 잊혀도 안전(같은 sid 로 새 카메라가 생기지 않음).
    _TOMBSTONE_TTL = 30.0

    def __init__(
        self,
        inference_interval: float = INFERENCE_INTERVAL,
        *,
        headroom: float = AUTOTUNE_HEADROOM,
        target_fps_max: float = AUTOTUNE_TARGET_FPS_MAX,
    ) -> None:
        self._captures: dict[str, VideoCaptureThread] = {}
        self._lock = threading.Lock()
        # telemetry 수렴 전 안전 초기 cadence. 수렴 후에는 worker 실측 용량과 source fps로 갱신.
        self._inference_interval = inference_interval
        self._min_interval = MIN_INFERENCE_INTERVAL
        self._max_interval = MAX_INFERENCE_INTERVAL
        self._headroom = max(0.0, min(1.0, headroom))
        self._target_fps_max = max(0.0, target_fps_max)

        self._worker = InferenceWorker()
        self._latest_results: dict[str, InferenceResult] = {}
        self._results_lock = threading.Lock()
        self._telemetry = InferenceTelemetry()
        self._source_telemetry: dict[str, SourceTelemetry] = {}
        self._telemetry_lock = threading.Lock()
        self._last_drop_count = 0
        self._last_adaptive_drop_count = 0
        self._gpu_util_target = GPU_UTIL_TARGET_DEFAULT
        self._util_scale = 1.0
        self._source_imgsz: dict[str, int] = {}
        self._imgsz_pressure_ticks: dict[str, int] = {}
        self._imgsz_relief_ticks: dict[str, int] = {}
        # MAX interval을 넘긴 source의 transition latch. 매 tick 경고 도배 없이 진입 시 1회만 기록.
        self._slow_interval_warned: set[str] = set()

        # source_id 별 추론 enabled — key 없으면 True 가 기본
        self._per_source_enabled: dict[str, bool] = {}
        # source_id 별 confidence threshold — key 없으면 worker 의 global 값 사용
        self._per_source_conf: dict[str, float] = {}
        # source_id 별 사용 모델 목록.
        #   key 없음 → global 기본 모델 1개 사용
        #   [] (빈 리스트) → 이 카메라 추론 안 함 (스켈레톤 없음)
        #   [m1, m2, ...] → 해당 모델 lane들을 병렬 사용하고 결과를 합침
        self._per_source_models: dict[str, list[str]] = {}
        self._per_source_lock = threading.Lock()

        # H1: 최근 삭제 source_id → 삭제 monotonic 시각. start_capture 의 create 분기가 참조해
        # 삭제된 카메라 캡처 resurrection(delete↔WS TOCTOU)을 막는다. self._lock 로 보호.
        self._tombstones: dict[str, float] = {}

        # fix19: worker 프로세스 사망 감지 watchdog — 마지막 respawn monotonic 시각(백오프 기준).
        self._last_worker_respawn: float = 0.0

        # lifecycle edge는 interval을 직접 쓰지 않고 dispatch autotuner의 즉시 tick만 요청한다.
        self._tune_requested = True

        self._dispatch_thread: Optional[threading.Thread] = None
        self._dispatch_running = False

    # ── lifecycle ────────────────────────────────────────────────

    def startup(self) -> None:
        """워커 + dispatch 스레드 기동."""
        self._sync_worker_models()
        self._worker.start()
        self._dispatch_running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="inference-dispatch"
        )
        self._dispatch_thread.start()
        logger.info("StreamManager started (inference worker + dispatch)")

    def shutdown(self) -> None:
        """모든 캡처 + 워커 + dispatch 정리."""
        # 1) 캡처 강제 종료
        with self._lock:
            captures = list(self._captures.values())
            self._captures.clear()
        for cap in captures:
            cap.force_stop()
            logger.info("Capture %s 강제 종료", cap.source_id)

        # 2) dispatch 종료
        self._dispatch_running = False
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=2)

        # 3) worker 종료
        self._worker.stop()
        logger.info("StreamManager shutdown 완료")

    # ── 캡처 시작/종료 (라우터에서 호출) ─────────────────────────

    def _is_tombstoned_locked(self, source_id: str) -> bool:
        """self._lock 보유 상태에서 호출 — source_id 가 TTL 내 삭제됐으면 True. 조회된 sid 가 만료됐으면
        청소한다(단, 삭제 후 재조회 안 되는 sid 는 이 lazy 경로로 안 지워짐 → 전역 청소는
        _prune_tombstones_locked 가 remove_capture 에서 담당)."""
        ts = self._tombstones.get(source_id)
        if ts is None:
            return False
        if time.monotonic() - ts >= self._TOMBSTONE_TTL:
            del self._tombstones[source_id]
            return False
        return True

    def _prune_tombstones_locked(self) -> None:
        """self._lock 하 — 만료된 tombstone 전부 청소. stream_key 는 재사용 안 되고 삭제된 sid 는
        start_capture 로 재조회되지 않아 lazy 청소가 안 먹으므로, 삭제 때마다 여기서 전역 청소해
        무한증식을 막는다(dict 는 최근 TTL 창의 삭제분만 유지)."""
        now = time.monotonic()
        for sid in [s for s, ts in self._tombstones.items() if now - ts >= self._TOMBSTONE_TTL]:
            del self._tombstones[sid]

    def start_capture(self, source_id: str, source: SourceType) -> bool:
        """source_id 캡처 시작(**없을 때만** 생성) + 뷰어 1명 증가. open 성공 시 True.

        create-only-if-absent: 엔트리가 이미 있으면 넘어온 `source` 를 **무시하고 기존 엔트리를 재사용**한다.
        URL 권위는 오직 update_ipcam→replace_source 로 단일화되어 있고, WS 핸들러가 넘기는 `source` 는
        `await accept()` 前에 읽혀 stale 일 수 있다(URL 편집 in-flight). 옛 "source 상이 시 교체" 방어망은
        stale WS 가 정확한 new_url 캡처를 옛 url 로 되돌려 모든 뷰어에게 딴 카메라를 지속 노출하는
        F1-무력화 SHOULD 였다(독립 게이트 확인) → 제거. 신규 url 반영은 replace_source 가 담당한다.

        H1: create 분기에서 tombstone(최근 삭제 sid)을 확인해 삭제된 카메라 resurrection 을 거부한다 —
        remove_capture 의 tombstone 등록과 self._lock 로 직렬화되어 delete↔WS 경합 창을 닫는다.

        증가(acquire_viewer)는 get/create 와 함께 **manager 락 안에서** 원자 수행 → replace_source 의
        adopt-read/swap 과 직렬화(증가side BLOCKER 제거, 감소side release_viewer 대칭). open(블로킹)은 락 밖.
        open 후 엔트리가 바뀌었으면(orphan: 기동 중 remove/replace) 또는 open 실패면, 증가분을 '현재
        엔트리'에서 release_viewer 로 되돌리고 방금 만든 캡처를 정리한 뒤 False.

        H2 invariant(중요): 이 메서드는 async ipcam_ws 의 **이벤트루프 스레드에서 동기 실행**된다(내부에
        await 없음). 그래서 두 WS 가 같은 sid 의 first-open 을 동시 경합할 수 없다(2nd viewer 가 open
        미확인 _running=True 를 보는 SHOULD-1/2 는 today unreachable). 이 메서드를 async/threadpool 로
        옮기면 그 경합이 재개되므로 옮기지 말 것 — 옮긴다면 acquire/open 을 재설계해야 한다.
        """
        with self._lock:
            cap = self._captures.get(source_id)
            if cap is None:
                if self._is_tombstoned_locked(source_id):
                    # H1: 최근 삭제된 카메라 — resurrect 거부(삭제된 RTSP 를 디코드하지 않는다). WS 는 닫힌다.
                    return False
                cap = VideoCaptureThread(
                    source_id=source_id,
                    source=source,
                    frame_callback=self._submit_frame,
                    # dispatch가 첫 bootstrap tick을 쓰기 전에도 다캠 30fps 폭주가 없도록
                    # 보수적으로 시작한다. 첫 frame은 deadline=0이라 즉시 제출되고, tuner가
                    # 곧 speed-up하면 setter가 다음 deadline을 앞으로 당긴다.
                    inference_interval=self._max_interval,
                )
                self._captures[source_id] = cap
            cap.acquire_viewer()   # 증가를 get/create 와 원자적으로 (락 하)
            target = cap
        ok = target._ensure_running()
        # orphan(기동 중 remove/replace) 또는 open 실패 → 증가분을 '현재 엔트리'에서 되돌린다.
        rollback_thread = None
        with self._lock:
            current = self._captures.get(source_id)
            orphaned = current is not target
            if orphaned:
                # 우리 증가는 (adopt 로) 현재 엔트리에 접혀 있으니 거기서 감소 (감소side 대칭).
                # NOTE: current 가 adopt-후손이 아닌 경우(remove_capture 로 dict 가 비워진 뒤 다른 WS 가
                # fresh 엔트리 생성)엔 남의 뷰어를 감소시킬 수 있으나, remove_capture 는 delete 전용이고
                # delete sweep(ipcam.py)이 그 fresh 엔트리도 force_stop 하므로 오늘은 무해(생존 피해자 없음).
                # 향후 non-delete remove_capture 호출자(idle-eviction 등)를 추가하면 이 가정 재검토 필요.
                if current is not None:
                    rollback_thread = current.release_viewer()
            elif not ok:
                # 현재 엔트리인데 open 실패 → 우리 증가 되돌림 (WS 는 False 에 stop_capture 를 안 부름).
                rollback_thread = target.release_viewer()
        if orphaned or not ok:
            if orphaned:
                target.force_stop()   # 밀려난 캡처 정리 (untracked 스레드 방지)
            if rollback_thread is not None:
                rollback_thread.join(timeout=5)
            self._recompute_cadence()
            return False
        self._recompute_cadence()
        return True

    def stop_capture(self, source_id: str) -> None:
        # 감소는 manager 락 안에서 '현재 엔트리'에 수행 → replace_source 의 swap 과 직렬화되어
        # 감소가 옛 캡처로 오배정되지 않는다(misdirected-decrement/ref 팽창 방지). join 은 락 밖.
        thread = None
        with self._lock:
            cap = self._captures.get(source_id)
            if cap is not None:
                thread = cap.release_viewer()
        if cap is None:
            return
        if thread is not None:
            thread.join(timeout=5)
        self._recompute_cadence()

    def replace_source(self, source_id: str, new_source: SourceType) -> bool:
        """URL 편집 시 캡처를 새 source 로 원자 교체하거나, 캡처가 없으면(idle 카메라) 새 source 를
        **URL 권위로 기록**한다(idle ref-0 엔트리 신설). 변경했으면 True.

        옛 캡처가 있으면 새 캡처로 락 안에서 원자 교체 + 뷰어 수(ref_count) 이관(adopt_viewers) 후 옛
        캡처 force-stop — 교체 도중 WS 가 연결/해제돼도 decrement 유실·ref 팽창이 없다(부재 창 없음).
        옛 캡처가 없으면(old=None: 뷰어 없는 idle 카메라) no-op 하지 않고 ref-0 idle 엔트리를 신설한다 —
        이렇게 URL 권위를 기록해야, 편집 중 stale 값을 읽은 첫 연결이 create-only-if-absent 로 이
        엔트리(=새 url)를 재사용해 옛 url 로 persistent stale 엔트리를 만들지 않는다(SHOULD-4,
        safety-critical: rescue-pose 로 전파되는 정본. round-2 의 old=None no-op 은 이 self-heal 을
        없앴었다). idle 엔트리는 ref-0·미기동이라 cadence·리소스 영향 없고, 첫 연결 시 open 되거나
        재편집/삭제로 교체·제거된다.
        NOTE(accept): 옛 스레드가 force_stop 직전에 워커에 제출한 프레임이 dispatch 루프에서 pop 이후
        다시 캐시될 수 있어 새 카메라 WS가 ~1 추론간격 동안 옛 스켈레톤을 볼 수 있다 — transient 수용.
        """
        with self._lock:
            old = self._captures.get(source_id)
            if old is not None and old.source == new_source:
                return False  # 변화 없음
            if old is None and self._is_tombstoned_locked(source_id):
                # H1 대칭: 삭제된(tombstoned) 카메라엔 idle 엔트리도 신설하지 않는다 (update↔delete
                # 경합 → 삭제 카메라의 dormant resurrection 방지). 살아있는 카메라의 실 교체(old not
                # None)는 삭제가 아니므로 tombstone 무관.
                return False
            new = VideoCaptureThread(
                source_id=source_id,
                source=new_source,
                frame_callback=self._submit_frame,
                inference_interval=self._max_interval,
            )
            if old is not None:
                new.adopt_viewers(old.ref_count)   # 뷰어 수 원자 이관 (idle 신설이면 ref-0 유지)
            self._captures[source_id] = new    # 엔트리 원자 교체/신설 (부재 창 없음)
        if old is not None:
            old.force_stop()
            with self._results_lock:
                self._latest_results.pop(source_id, None)  # 옛 카메라 검출 잔상 제거
        new._ensure_running()                  # ref>0 이면 1회 open, ref-0(idle 신설)이면 미기동
        # 기동 중 remove/재replace 로 엔트리가 바뀌었으면 방금 만든 new 는 orphan → 정리.
        orphan = None
        with self._lock:
            if self._captures.get(source_id) is not new:
                orphan = new
        if orphan is not None:
            orphan.force_stop()
        self._recompute_cadence()
        return True

    def remove_capture(self, source_id: str) -> None:
        """카메라 삭제 시 캡처를 ref_count 무시하고 완전 제거한다.

        stop_capture 는 ref-count 기반이라 viewer 2+ 면 삭제해도 스레드가 살아 삭제된 카메라를
        계속 디코드한다(F2). 삭제는 force-stop + dict 제거 + 검출/per-source 캐시 정리로 즉시 끝낸다.
        H1: pop 과 함께 tombstone 을 등록해, 아직 start_capture 를 안 부른 in-flight WS 가 삭제된
        카메라를 resurrect 하지 못하게 한다(start_capture create 분기가 self._lock 하에서 확인).
        """
        with self._lock:
            cap = self._captures.pop(source_id, None)
            self._tombstones[source_id] = time.monotonic()  # H1: resurrect 방지 (pop 과 원자)
            self._prune_tombstones_locked()  # 만료분 전역 청소 (무한증식 방지)
        if cap is not None:
            cap.force_stop()
        with self._results_lock:
            self._latest_results.pop(source_id, None)
        with self._per_source_lock:
            self._per_source_enabled.pop(source_id, None)
            self._per_source_conf.pop(source_id, None)
            self._per_source_models.pop(source_id, None)
        with self._telemetry_lock:
            self._source_telemetry.pop(source_id, None)
            getattr(self, "_source_imgsz", {}).pop(source_id, None)
            getattr(self, "_imgsz_pressure_ticks", {}).pop(source_id, None)
            getattr(self, "_imgsz_relief_ticks", {}).pop(source_id, None)
            warned = getattr(self, "_slow_interval_warned", None)
            if warned is not None:
                warned.discard(source_id)
        self._recompute_cadence()

    def _sync_worker_models(self) -> None:
        """현재 실행 중이고 ON인 source가 요구하는 model lane만 유지한다."""
        worker = getattr(self, "_worker", None)
        status_getter = getattr(worker, "get_status", None)
        configure = getattr(worker, "configure_models", None)
        if not callable(status_getter) or not callable(configure):
            return
        status = status_getter()
        desired: set[str] = set()
        if status.get("enabled", True):
            with self._lock:
                active_sources = {
                    sid for sid, cap in self._captures.items() if cap.is_running
                }
            with self._per_source_lock:
                for sid in active_sources:
                    if not self._per_source_enabled.get(sid, True):
                        continue
                    models = self._per_source_models.get(sid)
                    if models == []:
                        continue
                    desired.update(models or [str(status["model"])])
        configure(desired)

    def _recompute_cadence(self) -> None:
        """lifecycle edge에서 autotuner의 다음 tick을 즉시 요청한다.

        interval 쓰기는 _run_autotune 한 곳만 담당해 viewer 접속/해제가 직전 튜너 값을
        고정 N/MAX_INFER_PER_SEC 값으로 덮어쓰지 않는다.
        """
        self._tune_requested = True
        self._sync_worker_models()

    def _apply_bootstrap_cadence(
        self,
        active: list[tuple[str, VideoCaptureThread]],
    ) -> bool:
        """worker 표본 전에는 과거 보수적 상한으로 폐쇄형 총 제출률을 제한한다."""
        if not active:
            return False
        budget = (
            MAX_INFER_PER_SEC
            * self._headroom
            * max(GPU_UTIL_SCALE_MIN, min(1.0, getattr(self, "_util_scale", 1.0)))
        )
        if budget <= 0.0:
            interval = self._max_interval
        else:
            interval = max(self._inference_interval, len(active) / budget)
        interval = max(self._min_interval, interval)
        for _, cap in active:
            cap.set_inference_interval(interval)
        logger.info(
            "cadence bootstrap: active=%d budget=%.1f/s assigned=%.1f/s",
            len(active),
            budget,
            len(active) / interval,
        )
        return True

    def _update_adaptive_imgsz(
        self,
        source_id: str,
        *,
        allocation_ratio: float,
        queue_dropped: bool,
        underloaded: bool,
    ) -> bool:
        """source별 해상도 단계를 갱신하고 stage 변경 시 해당 cost 표본만 폐기한다."""
        with self._telemetry_lock:
            # object.__new__ 기반 단위 fixture와 이전 runtime state도 안전하게 승격한다.
            if not hasattr(self, "_source_imgsz"):
                self._source_imgsz = {}
                self._imgsz_pressure_ticks = {}
                self._imgsz_relief_ticks = {}
            current = self._source_imgsz.get(source_id, INFERENCE_IMGSZ)
            stage, pressure, relief, changed = _adaptive_imgsz_step(
                current,
                pressure_ticks=self._imgsz_pressure_ticks.get(source_id, 0),
                relief_ticks=self._imgsz_relief_ticks.get(source_id, 0),
                allocation_ratio=allocation_ratio,
                queue_dropped=queue_dropped,
                underloaded=underloaded,
            )
            self._source_imgsz[source_id] = stage
            self._imgsz_pressure_ticks[source_id] = pressure
            self._imgsz_relief_ticks[source_id] = relief
            if changed:
                metrics = self._source_telemetry.get(source_id)
                if metrics is not None:
                    metrics.infer_ms_ewma = 0.0
                    metrics.infer_samples = 0
        if changed:
            logger.info(
                "adaptive imgsz: source=%s %d→%d allocation=%.2f drop=%s",
                source_id,
                current,
                stage,
                allocation_ratio,
                queue_dropped,
            )
        return changed

    def _run_autotune(self) -> bool:
        """worker 실측 용량과 source fps snapshot으로 capture cadence를 갱신."""
        with self._lock:
            active = [(sid, cap) for sid, cap in self._captures.items() if cap.is_running]
        if not active:
            return False

        with self._per_source_lock:
            enabled = dict(self._per_source_enabled)
            models = {sid: list(names) for sid, names in self._per_source_models.items()}
        active = [
            (sid, cap)
            for sid, cap in active
            if enabled.get(sid, True) and models.get(sid, ["default"])
        ]
        if not active:
            return False

        pool_duty_getter = getattr(self._worker, "get_pool_duty", None)
        pool_duty = (
            max(0.0, min(1.0, float(pool_duty_getter())))
            if callable(pool_duty_getter)
            else None
        )
        with self._telemetry_lock:
            if pool_duty is not None:
                # result 단위 duty가 아니라 모델 lane들의 pool-wide busy union을 제어 신호로 쓴다.
                self._telemetry.duty_ewma = pool_duty
            telemetry_ready = self._telemetry.ready
            capacity = self._telemetry.capacity_fps
            duty = self._telemetry.duty_ewma
            global_infer_ms = self._telemetry.infer_ms_ewma
            target = max(
                GPU_UTIL_TARGET_MIN,
                min(
                    1.0,
                    getattr(
                        self,
                        "_gpu_util_target",
                        GPU_UTIL_TARGET_DEFAULT,
                    ),
                ),
            )
            util_scale = _gpu_util_scale_step(
                getattr(self, "_util_scale", 1.0),
                target=target,
                duty=duty,
            )
            self._gpu_util_target = target
            self._util_scale = util_scale
            source_costs = {
                sid: (metrics.infer_ms_ewma, metrics.infer_samples)
                for sid, metrics in self._source_telemetry.items()
            }
        if not telemetry_ready:
            return self._apply_bootstrap_cadence(active)

        wants: dict[str, float] = {}
        cost_weights: dict[str, float] = {}
        unknown_sources = 0
        for sid, cap in active:
            if cap.source_fps <= 0.0:
                # isOpened=True지만 frame advance가 없는 stall도 함대 전체를 막지 않는다.
                unknown_sources += 1
                wants[sid] = 0.0
            else:
                wants[sid] = min(cap.source_fps, self._target_fps_max)

            measured_cost, cost_samples = source_costs.get(sid, (0.0, 0))
            if cost_samples > 0 and measured_cost > 0.0:
                source_infer_ms = measured_cost
            else:
                # 다중 모델은 한 request 안에서 직렬 실행된다. source 표본 전에는 독립
                # 실측한 선형 비용을 보수적으로 model count로 근사하고 이후 EWMA로 교체한다.
                model_count = max(1, len(models.get(sid, [])))
                source_infer_ms = global_infer_ms * model_count
            cost_weights[sid] = source_infer_ms / global_infer_ms

        budget = capacity * self._headroom
        allocated = _allocate_fps(
            wants,
            budget_fps=budget,
            floor_fps=0.0,
            cost_weights=cost_weights,
        )
        stats_getter = getattr(self._worker, "get_queue_stats", None)
        stats = stats_getter() if callable(stats_getter) else {"dropped": 0}
        drops = int(stats.get("dropped", 0))
        previous_adaptive_drops = getattr(self, "_last_adaptive_drop_count", 0)
        queue_dropped = drops > previous_adaptive_drops
        self._last_adaptive_drop_count = drops
        weighted_demand = sum(wants[sid] * cost_weights[sid] for sid in wants)
        underloaded = weighted_demand <= budget * ADAPTIVE_UNDERLOAD_RATIO

        effective_floor = 0.0
        overruns: dict[str, float] = {}
        for sid, cap in active:
            base_fps = allocated.get(sid, 0.0)
            want = wants.get(sid, 0.0)
            allocation_ratio = (
                min(1.0, base_fps / want) if want > 0.0 else 1.0
            )
            self._update_adaptive_imgsz(
                sid,
                allocation_ratio=allocation_ratio,
                queue_dropped=queue_dropped,
                underloaded=underloaded,
            )
            # adaptive imgsz 판단은 원래 capacity allocation으로 끝낸 뒤, 사용자가 정한
            # GPU ceiling을 최종 cadence에만 적용한다. 의도적 감속을 overload로 오인하지 않는다.
            fps = base_fps * util_scale
            if fps <= 0.0:
                interval = self._max_interval
            else:
                interval = 1.0 / fps
            # n*FLOOR가 실측 budget보다 큰 경우 두 계약은 물리적으로 양립 불가다.
            # GPU service 총량 계약을 우선해 MAX보다 느린 cadence를 허용한다.
            assigned_interval = max(self._min_interval, interval)
            cap.set_inference_interval(assigned_interval)
            if assigned_interval > self._max_interval:
                overruns[sid] = assigned_interval

        # active snapshot 뒤 삭제된 source를 경고 latch에 되살리지 않는다. remove_capture가
        # telemetry lock에서 discard하므로 두 경합 순서 모두 최종 상태가 bounded다.
        with self._lock:
            live_sources = set(self._captures)
        current_overruns = set(overruns).intersection(live_sources)
        with self._telemetry_lock:
            previous_overruns = set(getattr(self, "_slow_interval_warned", set()))
            self._slow_interval_warned = current_overruns
        for sid in sorted(current_overruns - previous_overruns):
            logger.warning(
                "autotune MAX interval 초과: source=%s interval=%.3fs budget=%.3f/s "
                "effective_floor=%.3ffps",
                sid,
                overruns[sid],
                budget,
                effective_floor,
            )

        if drops > self._last_drop_count:
            logger.warning(
                "autotune queue drop 증가: +%d (capacity=%.1f/s, budget=%.1f/s)",
                drops - self._last_drop_count,
                capacity,
                budget,
            )
        self._last_drop_count = drops
        logger.info(
            "cadence autotune: active=%d unknown=%d capacity=%.1f/s duty=%.2f "
            "target=%.2f scale=%.2f budget=%.1f/s demand=%.1f/s",
            len(active),
            unknown_sources,
            capacity,
            duty,
            target,
            util_scale,
            budget,
            weighted_demand,
        )
        return True

    # (get_frame 제거 — JPEG 버퍼 없음. 영상은 mediamtx WHEP, detection 은 좌표 WS)
    # (get_capture_stats 제거 — stats endpoint 가 mediamtx readers 기반으로 바뀌어 死코드였고,
    #  그게 읽던 _inference_ts deque 가 dispatch 루프에서 영구 append 돼 source 당 메모리 누수였다.)

    # ── 캡처 ↔ 워커 bridge ───────────────────────────────────────

    def _submit_frame(self, source_id: str, frame: np.ndarray, captured_at: Optional[float] = None) -> None:
        """캡처 스레드 callback — global AND per-source 둘 다 ON 인 경우만 워커에 제출.

        per-source conf 가 설정돼 있으면 FrameRequest 에 포함 → 워커가 그 값으로 추론.
        없으면 None 이라 워커가 global 값 사용.
        """
        status = self._worker.get_status()
        if not status.get("enabled", True):
            return
        if not self.is_source_inference_enabled(source_id):
            return
        with self._per_source_lock:
            conf = self._per_source_conf.get(source_id)
            models_list = self._per_source_models.get(source_id)  # None or list

        # 빈 리스트 = 명시적 "추론 안 함"
        if models_list is not None and len(models_list) == 0:
            return

        # 가드 통과(추론 ON)한 프레임만 현재 adaptive stage로 축소한다. WS frame.w/h는
        # 이 실제 request frame 치수를 worker가 그대로 돌려줘 좌표 SEAM을 유지한다.
        with self._telemetry_lock:
            imgsz = getattr(self, "_source_imgsz", {}).get(source_id, INFERENCE_IMGSZ)
        frame = _resize_for_inference(frame, imgsz=imgsz)

        # 모델 list 전체를 pool에 전달 → 모델별 lane 병렬 실행 후 결과 합침.
        accepted = self._worker.submit(
            FrameRequest(
                source_id=source_id,
                frame=frame,
                timestamp=captured_at if captured_at is not None else time.time(),
                conf_threshold=conf,
                model_names=models_list,  # None or non-empty list
                imgsz=imgsz,
            )
        )
        if accepted:
            with self._telemetry_lock:
                # remove_capture와 원자적인 membership fence. 삭제 직후 in-flight callback이
                # source telemetry를 되살리지 못한다(락 순서 telemetry→manager, 역순 중첩 없음).
                with self._lock:
                    if source_id in self._captures:
                        metrics = self._source_telemetry.setdefault(
                            source_id, SourceTelemetry()
                        )
                        metrics.submitted += 1

    def _get_latest_result(self, source_id: str) -> Optional[InferenceResult]:
        """source_id 의 최신 추론 결과 (없으면 None)."""
        with self._results_lock:
            return self._latest_results.get(source_id)

    # WS 핸들러용 public alias — frontend overlay 가 detections JSON 으로 받음 (§4.19)
    def get_source_latest_detections(self, source_id: str) -> Optional[InferenceResult]:
        return self._get_latest_result(source_id)

    def _maybe_respawn_worker(self) -> None:
        """worker 프로세스가 죽었으면 respawn 한다 (fix19). shared state(_state)는 재사용되고 큐(in_q/out_q)는 재생성되며(fix21)
        worker_main 이 새 프로세스에서 모델을 lazy 재로드한다. respawn 폭주를 막으려 직전
        respawn 후 최소 10s 백오프를 둔다. _dispatch_loop(데몬 스레드, 이벤트루프 아님 →
        H2 무관)이 ~5s 주기로 호출한다."""
        if self._worker.is_alive():
            return
        nowm = time.monotonic()
        if nowm - self._last_worker_respawn < 10.0:
            return  # 백오프 — 직전 respawn 후 10s 미경과
        logger.error("Inference worker 프로세스 사망 감지 — respawn 시도")
        # fix22: start() 前에 갱신 — start() 가 raise(persistent OOM spawn 실패, fix21 armor 가 캐치)해도
        # 백오프가 유지돼 5s 마다 재시도(spawn-storm)하지 않는다. attempt-based(성공기반 아님).
        self._last_worker_respawn = nowm
        self._worker.start()

    def _record_results(self, results: list[InferenceResult]) -> None:
        """drain 결과를 전역 계측과 아직 추적 중인 source 캐시에 기록한다.

        worker 전역 service-time은 삭제 직전 결과도 유효하지만, source별 상태와 스켈레톤은
        remove_capture 뒤 되살아나면 안 된다. 각 cache lock 안에서 manager membership을
        재확인해 remove와 원자적으로 직렬화한다.
        """
        if not results:
            return
        with self._telemetry_lock:
            for result in results:
                self._telemetry.observe(result)
                with self._lock:
                    if result.source_id not in self._captures:
                        continue
                    metrics = self._source_telemetry.setdefault(
                        result.source_id, SourceTelemetry()
                    )
                    metrics.observe(result, alpha=self._telemetry.alpha)

        with self._results_lock:
            for result in results:
                with self._lock:
                    if result.source_id in self._captures:
                        current = self._latest_results.get(result.source_id)
                        if current is None or (
                            result.timestamp,
                            getattr(result, "request_id", 0),
                        ) >= (
                            current.timestamp,
                            getattr(current, "request_id", 0),
                        ):
                            self._latest_results[result.source_id] = result

    def _dispatch_loop(self) -> None:
        """worker.out_q → _latest_results 캐시 + worker 사망 watchdog(fix19). 별도 스레드.

        fix21(G1): while-body 전체를 try/except 로 감싼다 — worker 가 out_q.put 도중 killed(OOM,
        watchdog 의 표적 시나리오)되면 drain_results 가 queue.Empty 가 아니라 EOFError/OSError/
        UnpicklingError 를 던질 수 있고, 그러면 이 스레드가 죽어 결과캐싱+watchdog 이 함께 죽는다
        (watchdog 이 감시하던 바로 그 죽음에 자신이 피살). 어떤 예외든 살아남아야 한다(watchdog 계약).
        """
        next_worker_check = 0.0
        next_tune_ts = 0.0
        while self._dispatch_running:
            try:
                # worker 사망 감지 — drain 정상경로와 독립, ~5s 주기 게이트로 저부하.
                nowm = time.monotonic()
                if nowm >= next_worker_check:
                    next_worker_check = nowm + 5.0
                    self._maybe_respawn_worker()

                results = self._worker.drain_results()
                if results:
                    self._record_results(results)

                # 새 스레드 없이 기존 dispatch loop에서 1s 주기 + lifecycle 즉시 trigger.
                nowm = time.monotonic()
                if self._tune_requested or nowm >= next_tune_ts:
                    self._tune_requested = False  # run 중 새 요청은 다시 True로 남도록 먼저 clear.
                    next_tune_ts = nowm + 1.0
                    self._run_autotune()
                if not results:
                    time.sleep(self._DISPATCH_IDLE_SEC)
            except Exception:
                logger.exception("Dispatch loop 예외 — 복구 후 계속")
                time.sleep(1.0)
                continue
        logger.info("Dispatch loop 종료")

    # ── 추론 제어 (FastAPI 라우터에서 호출) ──────────────────────

    def get_inference_config(self) -> dict:
        status = dict(self._worker.get_status())
        duty_getter = getattr(self._worker, "get_pool_duty", None)
        if callable(duty_getter):
            duty = max(0.0, min(1.0, float(duty_getter())))
        else:
            with self._telemetry_lock:
                duty = max(0.0, min(1.0, self._telemetry.duty_ewma))
        with self._telemetry_lock:
            target = max(
                GPU_UTIL_TARGET_MIN,
                min(
                    1.0,
                    getattr(
                        self,
                        "_gpu_util_target",
                        GPU_UTIL_TARGET_DEFAULT,
                    ),
                ),
            )
        status.update(
            gpu_util_target=target,
            gpu_util_duty=duty,
        )
        return status

    def set_gpu_util_target(self, target: float) -> None:
        """전역 GPU duty ceiling 설정(10~100%)."""
        clamped = max(GPU_UTIL_TARGET_MIN, min(1.0, float(target)))
        with self._telemetry_lock:
            self._gpu_util_target = clamped
            if clamped >= 1.0:
                self._util_scale = 1.0
        self._tune_requested = True
        logger.info("GPU utilization target=%.0f%%", clamped * 100.0)

    def set_inference_enabled(self, enabled: bool) -> None:
        self._worker.set_enabled(enabled)
        if not enabled:
            # OFF 시 캐시 비워서 raw 프레임으로 회귀
            with self._results_lock:
                self._latest_results.clear()
        self._recompute_cadence()

    def set_inference_model(self, model_name: str) -> None:
        if self._worker.get_status().get("model") == model_name:
            return
        self._worker.set_model(model_name)
        # global 모델 비용이 달라지므로 이전 capacity로 새 모델을 과구독하지 않는다.
        # 새 모델의 warm-up 제외 표본이 모일 때까지 bootstrap cadence로 돌아간다.
        with self._telemetry_lock:
            self._telemetry = InferenceTelemetry()
            for metrics in self._source_telemetry.values():
                metrics.infer_ms_ewma = 0.0
                metrics.infer_samples = 0
            for source_id in list(getattr(self, "_source_imgsz", {})):
                self._source_imgsz[source_id] = INFERENCE_IMGSZ
                self._imgsz_pressure_ticks[source_id] = 0
                self._imgsz_relief_ticks[source_id] = 0
        self._recompute_cadence()

    def set_inference_conf_threshold(self, threshold: float) -> None:
        self._worker.set_conf_threshold(threshold)

    # ── per-source 추론 ON/OFF (각 카메라마다 독립적으로 제어) ──

    def is_source_inference_enabled(self, source_id: str) -> bool:
        """key 없으면 True (기본 ON)."""
        with self._per_source_lock:
            return self._per_source_enabled.get(source_id, True)

    def set_source_inference_enabled(self, source_id: str, enabled: bool) -> None:
        """source_id의 추론 ON/OFF. OFF 시 기존 결과 캐시를 비워 스켈레톤을 즉시 지운다."""
        with self._per_source_lock:
            self._per_source_enabled[source_id] = enabled
        if not enabled:
            with self._results_lock:
                self._latest_results.pop(source_id, None)
        self._recompute_cadence()
        logger.info("Per-source inference: %s = %s", source_id, enabled)

    def get_source_conf_threshold(self, source_id: str) -> Optional[float]:
        """source_id 의 per-source conf. 없으면 None (= global 사용)."""
        with self._per_source_lock:
            return self._per_source_conf.get(source_id)

    def set_source_conf_threshold(self, source_id: str, conf: float) -> None:
        """source_id 의 per-source conf 설정 (0~1)."""
        conf = max(0.0, min(1.0, float(conf)))
        with self._per_source_lock:
            self._per_source_conf[source_id] = conf
        logger.info("Per-source conf: %s = %.2f", source_id, conf)

    def get_source_models(self, source_id: str) -> Optional[list[str]]:
        """source_id 의 per-source 모델 목록.

        - None: 미설정 (global 기본 1개 사용)
        - []  : 명시적 추론 안 함
        - [m1, m2, ...]: 해당 모델들 (Phase 1 에선 [0] 만 적용)
        """
        with self._per_source_lock:
            models = self._per_source_models.get(source_id)
            return list(models) if models is not None else None  # 복사 반환

    def set_source_models(self, source_id: str, models: list[str]) -> None:
        """source_id 의 per-source 모델 목록 설정. 빈 리스트면 추론 안 함."""
        with self._per_source_lock:
            self._per_source_models[source_id] = list(models)
        with self._telemetry_lock:
            metrics = self._source_telemetry.get(source_id)
            if metrics is not None:
                metrics.infer_ms_ewma = 0.0
                metrics.infer_samples = 0
            if not hasattr(self, "_source_imgsz"):
                self._source_imgsz = {}
                self._imgsz_pressure_ticks = {}
                self._imgsz_relief_ticks = {}
            self._source_imgsz[source_id] = INFERENCE_IMGSZ
            self._imgsz_pressure_ticks[source_id] = 0
            self._imgsz_relief_ticks[source_id] = 0
        self._recompute_cadence()
        logger.info("Per-source models: %s = %s", source_id, models)


def detections_to_json(result: InferenceResult) -> str:
    """InferenceResult → WS 로 보낼 JSON 문자열. frontend overlay 가 파싱.

    box와 keypoint 좌표는 추론 캡처 frame 픽셀 기준. `frame:{w,h}` 동봉(SEAM) —
    frontend KeypointOverlay가 video.videoWidth/frame.w 비율로 keypoint를 변환한다.
    """
    return json.dumps({
        "type": "detections",
        "timestamp": result.timestamp,
        "frame": {"w": result.frame_w, "h": result.frame_h},
        "items": [
            {
                "class_id": d.class_id,
                "name": d.class_name,
                "conf": d.confidence,
                "xyxy": list(d.xyxy),
                "keypoints": [
                    [x, y, confidence] for x, y, confidence in d.keypoints
                ],
                "model": d.model,
            }
            for d in result.detections
        ],
    }, ensure_ascii=False)


# 싱글톤 — main.py 에서 startup/shutdown 호출.
# telemetry 수렴 전 안전 초기 cadence. 이후 interval 쓰기는 _run_autotune 단일 경로.
manager = StreamManager(inference_interval=INFERENCE_INTERVAL)
