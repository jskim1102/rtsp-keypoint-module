"""단일 영상 소스의 캡처 스레드 — 추론 제출 전용(좌표만).

`source` 가 `int` 면 V4L2 웹캠, `str` 이면 RTSP 등 OpenCV 가 인식하는 URL.
rtsp-keypoint: JPEG 인코딩 경로 제거 — 영상은 mediamtx WHEP 가 담당하고
이 스레드는 추론 워커에 raw 프레임만 제출한다(keypoint 좌표 전용).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import zlib
from typing import Callable, Optional, Union

import cv2
import numpy as np

from app.masking import mask_rtsp_credentials

logger = logging.getLogger("rtsp-keypoint.streaming.capture")

# Type alias
SourceType = Union[int, str]
FrameCallback = Callable[[str, np.ndarray, float], None]

_OPENCV_FFMPEG_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_LOW_LATENCY_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "analyzeduration;0|"
    "probesize;32"
)
_MAX_PTS_FUTURE_SEC = 0.25
STALL_SEC = 5.0
RECONNECT_BACKOFF_SEC = 2.0


class VideoCaptureThread:
    """단일 영상 소스의 캡처 루프 (추론 제출 전용).

    - 백그라운드 스레드에서 OpenCV `VideoCapture` 로 프레임 읽기
    - 쓰로틀링된 간격으로 추론 워커에 raw frame 제출 (callback)
    - ref_count 기반 lifecycle (여러 클라이언트 동시 시청 가능)

    `source` 가 `int` (예: 0) 면 V4L2 웹캠, `str` (예: "rtsp://...") 면 IP CAM.
    """

    def __init__(
        self,
        source_id: str,
        source: SourceType,
        *,
        frame_callback: Optional[FrameCallback] = None,
        inference_interval: float = 0.0,
    ) -> None:
        self.source_id = source_id
        self.source = source
        self._frame_cb = frame_callback
        # 추론 워커에 매 프레임마다 보내면 GPU 과부하 — drift-free 쓰로틀링.
        # `_next_submit_ts` 는 "이상적 다음 제출 시각" — 한 프레임 늦어져도 다음에 보충되어
        # 누적 drift 가 없음. (예전 `last_submit + interval >= now` 방식은 캡처 fps 와
        # 목표 fps 의 비율이 정수가 아닐 때 1 frame 씩 밀려 7.5fps 등으로 떨어짐.)
        self._inference_interval = inference_interval
        # 프로세스 재시작 뒤에도 같은 source는 같은 위상을 쓴다. Python hash는 실행마다
        # salt가 달라지므로 crc32로 0..1 사이의 안정 위상을 만든다.
        self._submit_phase = (zlib.crc32(source_id.encode("utf-8")) + 0.5) / (2**32)
        self._next_submit_ts = self._phase_deadline(time.time(), inference_interval)

        # 실제 capture frame advance(grab/read 성공) fps의 scalar EWMA. 과거의 무한
        # timestamp deque를 되살리지 않고 카메라 fps 상한(B)을 추정한다.
        self._frame_interval_ewma = 0.0
        self._last_frame_monotonic: Optional[float] = None
        self._fps_samples = 0

        self._cap: Optional[cv2.VideoCapture] = None
        self._pts_origin_wall: Optional[float] = None
        self._last_pts_msec: Optional[float] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # fix20: 기동 세대 카운터 — _ensure_running 이 매 기동마다 ++. 느리게 열린 구세대
        # 스레드(좀비)가 현세대 공유 상태를 덮어쓰지 못하게 _capture_loop 이 gen 을 fence 한다.
        self._generation = 0
        self._ref_count = 0
        self._open_event = threading.Event()
        # force_stop 되면 True (영구). _ensure_running 이 죽은 캡처를 다시 켜지 못하게 한다 —
        # delete↔reuse 경합에서 삭제된 카메라 RTSP 재디코드(resurrection)를 막는다.
        self._dead = False

    # ── lifecycle ────────────────────────────────────────────────

    def acquire_viewer(self) -> None:
        """뷰어 1명 증가(ref++)만 수행 — 기동(open)은 안 한다.

        manager.start_capture 가 이 증가를 **자기 락 안에서** 엔트리 get/create/adopt 와 원자적으로
        수행하고 기동(_ensure_running)은 락 밖에서 한다 — 증가가 항상 '현재 엔트리'에 원자 적용돼
        replace_source 의 adopt-read/swap 과 직렬화된다(증가side: 옛 설계는 cap.start() 가 락 밖에서
        ref++ → adopt 가 in-flight 증가를 놓치거나 orphan 경로에서 유령 ref 를 남겼다). release_viewer 대칭."""
        with self._lock:
            self._ref_count += 1

    def start(self) -> bool:
        """뷰어 1명 추가(ref++) 후 안 돌고 있으면 기동. 기동 실패 시 ref 되돌리고 False.
        (acquire_viewer + _ensure_running 편의 래퍼 — manager 는 두 단계를 락 분리해 직접 호출한다.)"""
        self.acquire_viewer()
        if not self._ensure_running():
            with self._lock:
                self._ref_count = max(0, self._ref_count - 1)
            return False
        return True

    def _ensure_running(self) -> bool:
        """ref_count 를 건드리지 않고 캡처 스레드만 기동한다(뷰어가 있을 때만).

        replace_source 가 뷰어 수를 이관(adopt_viewers)한 뒤 재기동에 쓰는 경로 — start() 와 달리
        ref 를 올리지 않는다. 뷰어가 없거나(_ref_count<=0) 이미 돌고 있으면 기동하지 않는다.
        `_running` 판정·설정을 락 안에서 하므로 stop() 의 0→종료와 경합해도 유령 스레드가 남지 않는다.
        """
        with self._lock:
            if self._dead:
                return False  # force_stop 된 캡처 — 재기동 금지(delete↔reuse resurrection 방지)
            if self._running or self._ref_count <= 0:
                return self._running
            self._running = True
            self._open_event.clear()
            self._generation += 1          # fix20: 이 기동의 세대 (좀비 스레드 fencing)
            my_gen = self._generation
            self._thread = threading.Thread(
                target=self._capture_loop,
                args=(my_gen,),
                daemon=True,
                name=f"capture-{self.source_id}-gen{my_gen}",
            )
            self._thread.start()

        # RTSP 는 연결까지 시간 걸릴 수 있어 timeout 넉넉히 (락 밖 — 다른 캡처 op 을 막지 않음).
        timeout = 15.0 if isinstance(self.source, str) else 5.0
        self._open_event.wait(timeout=timeout)

        if not self._cap or not self._cap.isOpened():
            with self._lock:
                self._running = False
            return False
        return True

    def adopt_viewers(self, n: int) -> None:
        """replace_source 전용 — 교체 직전(미공유·미기동) 새 캡처가 옛 캡처의 뷰어 수를 이관받는다.
        이관 후 _captures 에 원자적으로 꽂히므로 이후의 start/stop 이 모두 이 캡처를 올바로 증감한다
        (엔트리 부재 창이 없어 decrement 유실·ref 팽창이 없다 — replace_source BLOCKER 제거)."""
        with self._lock:
            self._ref_count = max(0, n)

    def release_viewer(self) -> Optional[threading.Thread]:
        """뷰어 1명 감소. 0 이 되면 _running=False 로 표시하고 join 할 스레드를 반환(아니면 None).

        manager.stop_capture 가 이 감소를 **자기 락 안에서** 수행하고 join(블로킹)은 락 밖에서 한다 —
        감소가 '현재 엔트리'에 원자 적용돼, URL 교체(replace_source)의 swap 과 겹쳐도 감소가 옛
        캡처로 오배정되지 않는다(new.ref 팽창 → 유령 디코드 스레드 방지). `_running`/thread 를 락
        안에서 확정하므로 _ensure_running 의 기동과도 일관된다."""
        with self._lock:
            self._ref_count = max(0, self._ref_count - 1)
            if self._ref_count > 0 or not self._running:
                return None
            self._running = False
            return self._thread

    def stop(self) -> None:
        """ref_count 감소, 0 이면 종료+join (release_viewer + join 편의 래퍼)."""
        thread = self.release_viewer()
        if thread:
            thread.join(timeout=5)

    def force_stop(self) -> None:
        """ref_count 무시하고 즉시 종료 + **영구 dead** 표시 (서버 셧다운·삭제·교체 시).
        dead 표시로 이후 _ensure_running 이 이 캡처를 다시 못 켠다 — reuse 분기에서 delete 가 lock-밖
        open 과 경합해도 삭제된 카메라를 재디코드하지 않는다."""
        with self._lock:
            self._dead = True
            self._running = False
            thread = self._thread
        if thread:
            thread.join(timeout=3)

    # ── 외부 조회 ────────────────────────────────────────────────
    # (get_frame 제거 — JPEG 버퍼 없음. 좌표 전용; 영상은 mediamtx WHEP)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def ref_count(self) -> int:
        """현재 활성 뷰어 수 — URL 교체 시 뷰어 보존용(manager.replace_source)."""
        with self._lock:
            return self._ref_count

    def set_inference_interval(self, interval: float) -> None:
        """autotuner가 호출 — 제출 케이던스 동적 갱신.
        float 단일 대입은 CPython 에서 원자적이고 캡처 루프가 매 iteration 마다
        self._inference_interval 를 읽으므로 락 불요."""
        old_interval = self._inference_interval
        self._inference_interval = interval
        if interval < old_interval:
            # autotuner가 속도를 올렸는데 옛 느린 deadline을 그대로 기다리지 않되, 모든 source를
            # now+interval 한 점에 모으지 않는다. 안정 위상으로 latest-wins burst를 막는다.
            self._next_submit_ts = self._phase_deadline(time.time(), interval)

    def _phase_deadline(self, now: float, interval: float) -> float:
        """현재 주기 안의 source 고유 위상에서 다음 deadline을 반환한다."""
        if interval <= 0.0:
            return now
        cycle_start = math.floor(now / interval) * interval
        deadline = cycle_start + self._submit_phase * interval
        if deadline <= now:
            deadline += interval
        return deadline

    @property
    def source_fps(self) -> float:
        """고정 메모리 frame-interval EWMA로 추정한 실제 카메라 fps.

        순간 fps(1/dt)를 평균하면 프레임 지터 때문에 상향 편향되므로 dt를 먼저
        평활한 뒤 한 번만 역수로 바꾼다.
        """
        return 1.0 / self._frame_interval_ewma if self._frame_interval_ewma > 0.0 else 0.0

    def _record_frame_advance(self, now: float) -> None:
        previous = self._last_frame_monotonic
        self._last_frame_monotonic = now
        if previous is None or now <= previous:
            return
        sample_interval = now - previous
        self._fps_samples += 1
        if self._fps_samples == 1:
            self._frame_interval_ewma = sample_interval
        else:
            self._frame_interval_ewma += 0.2 * (
                sample_interval - self._frame_interval_ewma
            )

    # ── 내부 ────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        """source 종류에 맞춰 OpenCV backend 선택."""
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            self._configure_capture(cap)
            return cap
        # RTSP 등 URL — FFMPEG backend 가 timeout 처리 강함
        os.environ.setdefault(_OPENCV_FFMPEG_OPTIONS_ENV, _LOW_LATENCY_FFMPEG_OPTIONS)
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self._configure_capture(cap)
        return cap

    def _configure_capture(self, cap: cv2.VideoCapture) -> None:
        """OpenCV backend 가 지원하면 내부 버퍼를 최소화한다."""
        prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if prop is None:
            return
        try:
            cap.set(prop, 1)
        except Exception:
            logger.debug("Capture %s — CAP_PROP_BUFFERSIZE 설정 실패", self.source_id, exc_info=True)

    def _frame_timestamp(self, cap: cv2.VideoCapture, read_ts: float) -> float:
        """가능하면 OpenCV PTS 를 wall-clock 에 매핑해 실제 프레임 간격을 보존한다."""
        get = getattr(cap, "get", None)
        if get is None:
            return read_ts
        try:
            pts_msec = float(get(cv2.CAP_PROP_POS_MSEC))
        except Exception:
            return read_ts
        if not math.isfinite(pts_msec) or pts_msec <= 0:
            return read_ts

        if (
            self._pts_origin_wall is None
            or self._last_pts_msec is None
            or pts_msec < self._last_pts_msec
        ):
            self._pts_origin_wall = read_ts - (pts_msec / 1000.0)

        ts = self._pts_origin_wall + (pts_msec / 1000.0)
        if ts > read_ts + _MAX_PTS_FUTURE_SEC:
            self._pts_origin_wall = read_ts - (pts_msec / 1000.0)
            ts = read_ts
        self._last_pts_msec = pts_msec
        return ts

    def _reconnect_capture(
        self,
        old_cap: cv2.VideoCapture,
        gen: int,
    ) -> Optional[cv2.VideoCapture]:
        """진전 없는 decoder를 닫고 같은 세대일 때만 새 capture를 공유 상태에 연결한다."""
        logger.warning("Capture %s decoder stall — reconnecting", self.source_id)
        old_cap.release()

        with self._lock:
            if not self._running or gen != self._generation:
                return None

        # 새 연결은 PTS·프레임 간격 계보가 모두 불연속이다. 재연결 중에는
        # source_fps=0으로 보여 autotuner가 옛 30fps를 계속 예약하지 않게 한다.
        self._pts_origin_wall = None
        self._last_pts_msec = None
        self._frame_interval_ewma = 0.0
        self._last_frame_monotonic = None
        self._fps_samples = 0

        while True:
            with self._lock:
                if not self._running or gen != self._generation:
                    return None
            time.sleep(RECONNECT_BACKOFF_SEC)
            # force_stop/supersede가 backoff 중 발생했으면 새 socket을 열지 않는다.
            with self._lock:
                if not self._running or gen != self._generation:
                    return None

            new_cap = self._open_capture()
            if not new_cap.isOpened():
                new_cap.release()
                continue

            with self._lock:
                if not self._running or gen != self._generation:
                    accepted = False
                else:
                    self._cap = new_cap
                    accepted = True
            if not accepted:
                new_cap.release()
                return None

            # source 고유 위상은 유지하고 deadline만 현재 주기에서 다시 계산한다.
            self._next_submit_ts = self._phase_deadline(
                time.time(), self._inference_interval
            )
            return new_cap

    def _capture_loop(self, gen: int) -> None:
        """캡처 + 추론 제출 (단일 스레드, 좌표 전용 — JPEG 인코딩 없음).

        fix20: `gen` 은 이 스레드의 세대(_ensure_running 이 락 안에서 발급). slow-open 좀비
        (구세대 스레드가 늦게 열려 현세대 공유 상태를 덮어쓰는 것)를 막기 위해 공유 상태
        (_cap/_open_event/_running)는 gen == self._generation 일 때만 건드린다. force_stop 은
        gen 을 바꾸지 않으므로(=_running=False+_dead) 열던 스레드가 gen 일치로 _cap 을 세팅해도
        루프조건 _running 이 False 라 즉시 미진입 → finally 가 정리(2중 디코드/orphan 없음).
        """
        cap = self._open_capture()

        # 로그용 source — rtsp_url 은 자격증명 마스킹, 정수 webcam source 는 그대로.
        safe_source = (
            mask_rtsp_credentials(self.source)
            if isinstance(self.source, str)
            else self.source
        )

        # open(블로킹) 후 — 이 스레드가 최신 세대인지 확인. 구세대(superseded)면 stale:
        # 방금 연 cap 만 정리하고 공유 상태(_cap/_open_event/_running)는 절대 건드리지 않는다.
        with self._lock:
            superseded = gen != self._generation
            if not superseded:
                self._cap = cap
                self._open_event.set()
        if superseded:
            cap.release()
            return

        if not cap.isOpened():
            logger.error("Capture %s 열기 실패: source=%s", self.source_id, safe_source)
            with self._lock:
                if gen == self._generation:
                    self._cap = None
                    self._running = False
            cap.release()  # 실패한 VideoCapture 도 FD(FFMPEG/socket) 해제 — 반복 재시도 시 누수 방지
            return

        logger.info("Capture %s 시작 (source=%s, gen=%d)", self.source_id, safe_source, gen)
        last_ok = time.time()
        try:
            while self._running and cap.isOpened() and gen == self._generation:
                now = time.time()

                # DECODE cadence 스로틀: 제출할 프레임만 read()(grab+decode)하고,
                # 그 사이 프레임은 grab()(패킷 drain, 디코드 X)으로만 흘려보낸다 — 16 카메라에서
                # 쓰지도 않을 프레임을 full-fps 로 디코드하던 CPU/네트워크 낭비 제거.
                submit_due = self._frame_cb is not None and now >= self._next_submit_ts

                if not submit_due:
                    # 디코드 없이 버퍼만 비운다(최신 프레임 유지). inference_interval==0 이면
                    # submit_due 가 항상 True → 이 분기 미진입 → full-rate 디코드(하위호환).
                    if not cap.grab():
                        if time.time() - last_ok >= STALL_SEC:
                            reconnected = self._reconnect_capture(cap, gen)
                            if reconnected is None:
                                break
                            cap = reconnected
                            last_ok = time.time()
                            continue
                        time.sleep(0.01)
                    elif gen != self._generation:
                        break
                    else:
                        last_ok = time.time()
                        self._record_frame_advance(time.monotonic())
                    continue

                # 제출 시점에만 한 번 decode한다.
                ret, frame = cap.read()
                read_ts = time.time()
                if not ret:
                    if read_ts - last_ok >= STALL_SEC:
                        reconnected = self._reconnect_capture(cap, gen)
                        if reconnected is None:
                            break
                        cap = reconnected
                        last_ok = time.time()
                        continue
                    time.sleep(0.01)
                    continue

                # fix20: 디코드 직후 세대 재확인 — superseded면 shared metrics/제출 모두 차단.
                if gen != self._generation:
                    break
                last_ok = read_ts
                frame_monotonic = time.monotonic()
                self._record_frame_advance(frame_monotonic)

                captured_at = self._frame_timestamp(cap, read_ts)

                # 추론 워커에 raw 프레임 제출 (drift-free 쓰로틀링) — 좌표 전용.
                # 이상적 다음 시각으로 진행 — 늦어졌으면 now 로 점프하여 누적 drift 없음.
                try:
                    self._frame_cb(self.source_id, frame, captured_at)
                    self._next_submit_ts = max(
                        self._next_submit_ts + self._inference_interval,
                        now,
                    )
                except Exception:
                    logger.exception("Capture %s — frame_cb 예외", self.source_id)
                # (JPEG 인코딩/내부 버퍼 제거 — 영상은 mediamtx WHEP 가 담당, §4.19)
        except Exception:
            logger.exception("Capture %s 루프 예외", self.source_id)
        finally:
            cap.release()
            with self._lock:
                if gen == self._generation:
                    self._cap = None
                    self._running = False
            logger.info("Capture %s 종료 (gen=%d)", self.source_id, gen)
