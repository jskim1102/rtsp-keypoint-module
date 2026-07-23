"""RTSP 자격증명 마스킹 — API 응답·로그에서 카메라 비밀번호 평문 노출 차단(P1).

중립 모듈로 둔다(ipcam.py·mediamtx.py 양쪽이 import → 순환 의존 회피). 표준 라이브러리만
의존하며 app 내부 모듈을 import 하지 않는다.

urlsplit 을 쓰지 않는 이유: 비밀번호에 `/ # ?` 같은 URL 예약문자가 있으면 urlsplit 이
거기서 끊어 비번 일부를 path/fragment 로 흘려 마스킹을 우회시킨다(보안버그). 여기선
**마지막 @**(userinfo|host 경계 — host 엔 @ 없음)와 **첫 :**(user|password 경계)로만
문자열을 가른다.
"""

_MASK = "***"


def _split_credentials(url: str) -> tuple[str, str, str | None, str] | None:
    """url 의 자격증명 구간을 reserved-char-safe 하게 분해 — mask/restore 공통 헬퍼.

    `scheme://[user[:password]@]rest` 에서 **마지막 @** 로 userinfo|rest 를 가르고,
    userinfo 안 **첫 :** 로 user|password 를 가른다. 비밀번호에 `/ # ? @` 등 URL
    예약문자가 있어도 안전하다.

    반환 `(prefix, user, password, rest)`:
      - prefix   = `scheme://`
      - user     = 사용자명 (없으면 "")
      - password = 비밀번호 (없으면 None — `user@host` 또는 자격증명 없음)
      - rest     = `host[:port]/path...` (마지막 @ 이후 전부, 그대로 보존)
    자격증명(@)이 없으면 None.
    """
    sep = url.find("://")
    if sep < 0:
        return None
    prefix = url[: sep + 3]
    authority_plus = url[sep + 3 :]
    at = authority_plus.rfind("@")  # host 엔 @ 없음 → 마지막 @ = userinfo 경계
    if at < 0:
        return None
    userinfo = authority_plus[:at]
    rest = authority_plus[at + 1 :]
    colon = userinfo.find(":")  # 첫 : = user|password 경계 (비번 안의 : 는 보존)
    if colon < 0:
        return (prefix, userinfo, None, rest)
    return (prefix, userinfo[:colon], userinfo[colon + 1 :], rest)


def mask_rtsp_credentials(url: str) -> str:
    """rtsp_url 의 비밀번호를 *** 로 마스킹 — API/UI·로그 노출 시 카메라 자격증명 보호.

    `rtsp://user:pass@host/path` → `rtsp://user:***@host/path`. 비밀번호 없으면 원본.
    비밀번호에 `/ # ? @` 가 있어도 전체를 마스킹한다(_split_credentials 사용).
    """
    parts = _split_credentials(url)
    if parts is None:
        return url
    prefix, user, password, rest = parts
    if not password:
        return url  # `user@host` / 빈 비번 — 마스킹할 게 없음
    userpart = f"{user}:{_MASK}" if user else f":{_MASK}"
    return f"{prefix}{userpart}@{rest}"


def _restore_masked_password(new_url: str, old_url: str) -> str:
    """new_url 의 비밀번호가 ***(마스킹)이면 old_url 의 실제 비밀번호로 치환해 돌려준다.

    host/port/path/user/scheme 등 나머지 컴포넌트는 new_url(사용자 수정값)을 보존한다.
    목록은 비밀번호를 *** 로 마스킹해 내려가므로, 사용자가 비밀번호를 다시 입력하지 않고
    주소만 바꿔 저장하면 new_url 의 비번이 *** 인 채로 들어온다. 이때 *** 만 기존 실제
    비번으로 되돌리고 바뀐 주소는 그대로 적용한다.
    마스킹이 아니면(= 사용자가 전체 URL 을 새로 입력) new_url 을 그대로 쓴다.
    비밀번호에 `/ # ? @` 가 있어도 깨지지 않는다(_split_credentials 공통 헬퍼).

    리터럴 *** 비번 구분: 실제 비번이 정확히 `***` 일 수도 있다(드물지만 유효). old 에
    복원할 실제 비번이 없으면 들어온 `***` 는 마스킹 토큰이 아니라 **사용자가 입력한
    리터럴 비번** 이므로, 기존값으로 되돌리지 않고(복원할 게 없음) new_url 을 그대로 둔다.
    (복원은 old 에 실제 비번이 있을 때만 — 그 경우만 *** 가 마스킹 토큰으로 해석된다.)
    """
    new = _split_credentials(new_url)
    if new is None or new[2] != _MASK:
        return new_url  # 자격증명 없음 or 비번이 *** 아님(= 사용자가 전체 새로 입력)
    old = _split_credentials(old_url)
    if old is None or old[2] is None:
        return new_url  # old 에 복원할 실제 비번 없음 → 들어온 *** 는 리터럴 비번, 그대로 둠
    prefix, user, _new_pw, rest = new
    real_pw = old[2]
    userpart = f"{user}:{real_pw}" if user else f":{real_pw}"
    return f"{prefix}{userpart}@{rest}"
