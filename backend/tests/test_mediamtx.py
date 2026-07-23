"""mediamtx 모듈 테스트 — fail-fast(MEDIAMTX_API 미설정) + runOnDemand transcode
payload + rtsp_url injection 가드.
"""

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

import app.mediamtx as mediamtx

# autouse 패치 전 원본 _probe_codec — 실제 함수 단위테스트에서 mock subprocess 로 직접 호출.
_REAL_PROBE = mediamtx._probe_codec


@pytest.fixture(autouse=True)
def _stub_probe_codec():
    """_transcode_command 의 코덱 probe 가 실제 ffprobe(네트워크/바이너리)를 호출하지 않게 —
    기본 None(→ libx264 폴백). h264 분기는 개별 테스트가 명시 override 한다."""
    with patch("app.mediamtx._probe_codec", return_value=None) as p:
        yield p


def test_register_stream_raises_when_api_unset():
    with patch.object(mediamtx, "MEDIAMTX_API", ""):
        with pytest.raises(RuntimeError, match="MEDIAMTX_API"):
            mediamtx.register_stream("ipcam-x", "rtsp://x/1")


def test_remove_stream_raises_when_api_unset():
    with patch.object(mediamtx, "MEDIAMTX_API", ""):
        with pytest.raises(RuntimeError, match="MEDIAMTX_API"):
            mediamtx.remove_stream("ipcam-x")


def test_get_path_raises_when_api_unset():
    with patch.object(mediamtx, "MEDIAMTX_API", ""):
        with pytest.raises(RuntimeError, match="MEDIAMTX_API"):
            mediamtx.get_path("ipcam-x")


# ─── remove_stream 끝상태 반환 (codex #4 — orphan 라이브 스트림 차단) ───


@pytest.mark.parametrize("status", [200, 204])
def test_remove_stream_returns_true_on_success(status):
    """삭제 성공(200/204) → True (끝상태=path 없음)."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.delete") as delete:
        delete.return_value = MagicMock(status_code=status)
        assert mediamtx.remove_stream("ipcam-x") is True


def test_remove_stream_returns_true_on_404_idempotent():
    """path 가 애초에 없음(404) → 끝상태(path 없음)는 동일하므로 멱등 성공(True)."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.delete") as delete:
        delete.return_value = MagicMock(status_code=404)
        assert mediamtx.remove_stream("ipcam-x") is True


def test_remove_stream_returns_false_on_error_status():
    """그 외 상태코드(예: 500) → False (호출부가 DB commit 중단 → orphan 차단)."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.delete") as delete:
        delete.return_value = MagicMock(status_code=500, text="boom")
        assert mediamtx.remove_stream("ipcam-x") is False


def test_remove_stream_returns_false_on_connection_error():
    """mediamtx 연결 실패(HTTPError) → False (끝상태 미확인 → 삭제 보류)."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.delete", side_effect=mediamtx.httpx.ConnectError("down")):
        assert mediamtx.remove_stream("ipcam-x") is False


# ─── runOnDemand 조건부 transcode payload (패턴 C) ───


def test_register_posts_runondemand_payload():
    """register_stream 이 source 대신 runOnDemand transcode 커맨드를 보낸다."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        mediamtx.register_stream("ipcam-abc", "rtsp://cam/stream1")

        _, kwargs = post.call_args
        body = kwargs["json"]
        # source 패스스루 아님 — runOnDemand 사용
        assert "source" not in body
        cmd = body["runOnDemand"]
        # 셸 없이 직접 ffmpeg (mediamtx 가 셸변수 $C/$V 를 먹어서 조건분기 불가)
        assert cmd.startswith("ffmpeg")
        # 카메라 rtsp 는 등록 시점 치환
        assert "rtsp://cam/stream1" in cmd
        # 항상 H.264 transcode (모든 코덱 호환 — passthrough 최적화 포기)
        assert "libx264" in cmd
        # 시간기반 keyframe 강제 — 저fps 카메라 WHEP join 지연 방지 (~2초 내 IDR)
        assert "force_key_frames" in cmd
        # mediamtx 런타임 변수는 literal($ 그대로)
        assert "$RTSP_PORT" in cmd and "$MTX_PATH" in cmd
        # on-demand 종료
        assert body["runOnDemandCloseAfter"] == "10s"


# ─── rtsp_url injection 가드 (셸리스 exec + 스킴/셸메타 denylist) ───


def test_register_rejects_non_rtsp_scheme():
    """rtsp/rtsps 아닌 스킴은 ValueError (POST 도달 전)."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        with pytest.raises(ValueError, match="스킴"):
            mediamtx.register_stream("ipcam-x", "http://cam/stream")
        post.assert_not_called()


def test_register_accepts_rtsps_scheme():
    """rtsps:// (TLS) 스킴도 허용 — 일부 카메라가 암호화 RTSP 를 쓴다."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        assert mediamtx.register_stream("ipcam-x", "rtsps://cam:322/stream") is True


@pytest.mark.parametrize("bad", [
    "rtsp://cam/stream\nmalicious",   # 개행 (헤더/명령 분리 표면)
    "rtsp://cam/stream\rmalicious",   # 캐리지리턴
    "rtsp://cam/stream\x00trunc",     # 널바이트 (C 문자열 절단)
])
def test_register_rejects_control_chars(bad):
    """제어문자(\\n \\r \\x00)는 ValueError (denylist) — POST 도달 전."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        with pytest.raises(ValueError, match="금지문자"):
            mediamtx.register_stream("ipcam-x", bad)
        post.assert_not_called()


@pytest.mark.parametrize("evil", [
    "rtsp://cam/stream'; touch /tmp/pwned; echo '",  # quote breakout + 명령주입
    'rtsp://cam/stream"$(whoami)"',                   # command substitution
    "rtsp://cam/`id`",                                 # backtick
    "rtsp://cam/stream; echo hi",                      # 세미콜론 명령 분리
])
def test_register_rejects_shell_metachars(evil):
    """셸 메타문자(; $ ` | & ( ) ' " 등)는 denylist 로 ValueError — POST 도달 전.

    runOnDemand 는 셸 없이 직접 exec 되지만(mediamtx), mediamtx 의 $ 변수치환 +
    미래 셸-wrapping 대비 defense-in-depth 로 메타문자를 등록 시점에 거부한다.
    """
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        with pytest.raises(ValueError, match="금지문자"):
            mediamtx.register_stream("ipcam-x", evil)
        post.assert_not_called()


def test_register_accepts_normal_rtsp_url():
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        # 인증정보 포함 정상 URL 도 통과
        assert mediamtx.register_stream("ipcam-x", "rtsp://user:pass@192.168.0.10:554/h264") is True


def test_register_accepts_query_string_ampersand():
    """`&` 는 RTSP query string(Dahua/Hikvision channel=X&subtype=Y)의 정상 문자 — 허용.

    셸 없이 exec 되는 runOnDemand 에선 `&` 가 argv 토큰 내부 리터럴이라 무해하므로
    denylist 에서 제외했다. 실제 카메라 URL 형태가 400 으로 막히지 않아야 한다.
    """
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        url = "rtsp://admin:pw@192.168.0.113:554/cam/realmonitor?channel=3&subtype=0"
        assert mediamtx.register_stream("ipcam-x", url) is True
        # POST 가 실제로 도달하고 rtsp_url 이 runOnDemand 커맨드에 그대로 들어갔는지
        cmd = post.call_args.kwargs["json"]["runOnDemand"]
        assert "channel=3&subtype=0" in cmd


# ─── P1 로그 평문 비번 누수 차단 (CEO #62) ───


def test_register_masks_password_in_log(caplog):
    """register_stream 의 등록 로그가 비번을 평문으로 남기지 않는다.

    버그: `→ %s` 로 rtsp_url 전체를 찍어 stdout/docker 로그에 비번 평문 노출.
    / 와 # 를 포함한 비번으로도 로그에 *** 만 남아야 한다.
    """
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        with caplog.at_level(logging.INFO, logger="rtsp-keypoint.mediamtx"):
            mediamtx.register_stream("ipcam-x", "rtsp://admin:pa/ss#word@10.0.0.5:554/cam")

    text = caplog.text
    assert "ipcam-x" in text  # 등록 로그가 실제로 찍혔는지
    assert "rtsp://admin:***@10.0.0.5:554/cam" in text  # 마스킹된 형태로
    for leak in ("pa/ss", "#word", "pa/ss#word"):
        assert leak not in text, f"로그 비번 평문 누수: {leak!r}"


# ─── update_stream (PATCH 원자 부분갱신 — remove→add 재등록 갭 제거) ───


def test_update_stream_uses_patch_verb_and_path():
    """update_stream 은 PATCH /v3/config/paths/patch/{key} 로 runOnDemand 만 갱신한다."""
    with patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.patch") as patch_req:
        patch_req.return_value = MagicMock(status_code=200)
        assert mediamtx.update_stream("ipcam-abc", "rtsp://cam/stream2") is True

        args, kwargs = patch_req.call_args
        assert args[0].endswith("/v3/config/paths/patch/ipcam-abc")  # PATCH verb + patch path
        body = kwargs["json"]
        assert "rtsp://cam/stream2" in body["runOnDemand"]  # 새 url 로 runOnDemand 갱신
        assert "source" not in body  # runOnDemand 만 부분갱신


def test_update_stream_raises_when_api_unset():
    with patch.object(mediamtx, "MEDIAMTX_API", ""):
        with pytest.raises(RuntimeError, match="MEDIAMTX_API"):
            mediamtx.update_stream("ipcam-x", "rtsp://x/1")


# ─── ensure_stream (self-heal — mediamtx recreate 로 path 소멸 시 DB 로 재등록) ───


def test_ensure_stream_returns_path_when_exists():
    """path 가 이미 있으면 재등록 안 하고(idempotent) 그 path 를 반환해 호출부가 재사용한다."""
    db = MagicMock()
    existing = {"ready": True, "readers": []}
    with patch("app.mediamtx.get_path", return_value=existing) as gp, \
         patch("app.mediamtx.register_stream") as reg:
        assert mediamtx.ensure_stream(db, "ipcam-x") == existing
        reg.assert_not_called()
        gp.assert_called_once_with("ipcam-x")  # path 존재 시 get_path 1회만(중복호출 없음)


def test_ensure_stream_reregisters_when_path_missing():
    """path 가 사라졌고(None) DB 에 카메라가 있으면 DB 의 rtsp_url 로 재등록 후 갱신된 path 반환."""
    cam = MagicMock(stream_key="ipcam-x", rtsp_url="rtsp://cam/1")
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = cam
    fresh = {"ready": False, "readers": []}
    # 재등록 전 get_path=None(부재) → 재등록 후 get_path=fresh. side_effect 로 2회 분기.
    with patch("app.mediamtx.get_path", side_effect=[None, fresh]), \
         patch("app.mediamtx.register_stream", return_value=True) as reg:
        assert mediamtx.ensure_stream(db, "ipcam-x") == fresh
        reg.assert_called_once_with("ipcam-x", "rtsp://cam/1")


def test_ensure_stream_returns_none_when_cam_missing():
    """path 없고 DB 에도 카메라 없으면 재등록 안 하고 None 반환."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    with patch("app.mediamtx.get_path", return_value=None), \
         patch("app.mediamtx.register_stream") as reg:
        assert mediamtx.ensure_stream(db, "ipcam-x") is None
        reg.assert_not_called()


# ─── 조건부 transcode: h264=passthrough(copy) / 그 외=libx264 (CEO #97 화질 정본) ───


def test_transcode_h264_source_uses_passthrough_copy():
    """h264 소스 → -c:v copy 무손실 패스스루(재인코딩 0, 화질 보존). WebRTC 네이티브."""
    with patch("app.mediamtx._probe_codec", return_value="h264"):
        cmd = mediamtx._transcode_command("rtsp://cam/h264")
    assert "-c:v copy" in cmd
    assert "libx264" not in cmd
    assert "force_key_frames" not in cmd  # copy 라 인코더 옵션 없음
    assert "$RTSP_PORT" in cmd and "$MTX_PATH" in cmd  # mediamtx 변수는 literal 유지


def test_transcode_non_h264_source_uses_libx264():
    """H.265 등 비-h264 → libx264 재인코딩(WebRTC 호환) + 2초 IDR 강제."""
    with patch("app.mediamtx._probe_codec", return_value="hevc"):
        cmd = mediamtx._transcode_command("rtsp://cam/h265")
    assert "-c:v libx264" in cmd
    assert "force_key_frames" in cmd
    assert "-c:v copy" not in cmd


def test_transcode_probe_failure_falls_back_to_libx264():
    """probe 실패(None) → 안전기본 libx264(현행 유지 = 롤백 안전)."""
    with patch("app.mediamtx._probe_codec", return_value=None):
        cmd = mediamtx._transcode_command("rtsp://cam/unknown")
    assert "-c:v libx264" in cmd


def test_probe_codec_parses_ffprobe_codec_name():
    """ffprobe stdout 의 codec_name 을 소문자로 반환."""
    with patch("app.mediamtx.subprocess.run") as run:
        run.return_value = MagicMock(stdout="H264\n")
        assert _REAL_PROBE("rtsp://cam/1") == "h264"
        assert run.call_args.kwargs.get("timeout") == 4  # CEO #124 — 도달불가 카메라 등록 latency 단축(10→4s)


def test_probe_codec_returns_none_on_failure():
    """ffprobe 타임아웃/미설치 등 실패 → None(→ libx264 폴백)."""
    with patch("app.mediamtx.subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 10)):
        assert _REAL_PROBE("rtsp://cam/1") is None


# ─── mediamtx API 인증 (CEO #100 — backend Basic auth) + 로그 마스킹 ───


def test_register_sends_backend_basic_auth_when_pass_set():
    """MEDIAMTX_BACKEND_PASS 설정 시 httpx 호출에 (user,pass) Basic auth 전달."""
    with patch.object(mediamtx, "MEDIAMTX_BACKEND_USER", "backend"), \
         patch.object(mediamtx, "MEDIAMTX_BACKEND_PASS", "s3cret"), \
         patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        mediamtx.register_stream("ipcam-x", "rtsp://cam/1")
        assert post.call_args.kwargs.get("auth") == ("backend", "s3cret")


def test_register_no_auth_when_pass_unset():
    """비번 미설정(빈 문자열)이면 auth=None — 무인증(로컬/테스트 하위호환)."""
    with patch.object(mediamtx, "MEDIAMTX_BACKEND_PASS", ""), \
         patch.object(mediamtx, "MEDIAMTX_API", "http://mtx:9997"), \
         patch("app.mediamtx.httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        mediamtx.register_stream("ipcam-x", "rtsp://cam/1")
        assert post.call_args.kwargs.get("auth") is None


def test_transcode_suppresses_ffmpeg_plaintext_url_log():
    """ffmpeg -hide_banner -loglevel error: stdout 의 rtsp 평문 URL 덤프 차단(#100.4)."""
    cmd = mediamtx._transcode_command("rtsp://admin:pw@cam/1")
    assert "-hide_banner" in cmd and "-loglevel error" in cmd
