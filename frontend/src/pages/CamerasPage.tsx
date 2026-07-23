import { useState, useEffect, useCallback, useRef } from "react";
import { apiBase } from "../hooks/useApi";
import CameraFormModal from "../components/CameraFormModal";
import CameraGrid from "../components/CameraGrid";
import SegmentedToggle from "../components/SegmentedToggle";
import ModelManagerModal from "../components/ModelManagerModal";
import ModelSettingsModal, { type ModelSettings } from "../components/ModelSettingsModal";
import {
  GpuUtilTargetUpdater,
  normalizeGpuUtilConfig,
  type GpuUtilConfigStatus,
} from "../utils/gpuUtilControl";

export interface Cam {
  id: number;
  name: string;
  rtsp_url: string;
  stream_key: string;
  created_at: string;
}

interface Stat {
  active: boolean;
  readers: number;
}

const MAX_IPCAMS_FALLBACK = 16; // spec F4 — /api/config 로딩 전 기본값. 실제 cap 은 백엔드 env.
const DEFAULT_CONF = 0.5; // deepeye 정본 (YOLO_CONF_THRESHOLD).
const DEFAULT_GPU_UTIL_TARGET_PCT = 85;

// 실패 응답에서 backend detail(비어있지 않은 문자열)만 노출, 없거나 비문자열·공백이면 fallback.
// detail 은 <p>{error}</p> 로 렌더되므로 문자열 보장 필수 — 객체/배열 detail 렌더 크래시 차단.
async function errorDetail(resp: Response, fallback: string): Promise<string> {
  // .catch(null) + ?. — 본문이 JSON `null` 이거나 파싱 실패여도 안전(널 역참조 크래시 차단).
  const body = await resp.json().catch(() => null);
  const detail = (body as { detail?: unknown } | null)?.detail;
  return typeof detail === "string" && detail.trim() ? detail : fallback;
}

export default function CamerasPage() {
  const [cams, setCams] = useState<Cam[]>([]);
  const [stats, setStats] = useState<Record<string, Stat>>({});
  // 실측 FPS — 그리드의 WhepPlayer 가 WebRTC getStats 로 올려주는 카메라별 디코딩 프레임레이트.
  const [fps, setFps] = useState<Record<string, number>>({});
  // 실측 det fps — 그리드의 detection WS(useDetectionWs) 가 올려주는 카메라별 메시지 도착률.
  const [detFps, setDetFps] = useState<Record<string, number>>({});
  // 등록 cap — 백엔드 /api/config(MAX_IPCAMS env)에서 받음. 프론트 하드코딩 제거(P2-1).
  const [maxIpcams, setMaxIpcams] = useState(MAX_IPCAMS_FALLBACK);
  const [formOpen, setFormOpen] = useState(false);
  const [editCam, setEditCam] = useState<Cam | null>(null);
  const [error, setError] = useState("");
  // 카메라별 remount epoch — RTSP 변경 편집 성공 시 bump(→ CameraGrid key 변경 → 셀 remount).
  // 응답 rtsp_url 은 마스킹(:***@)이라 비번-only 변경을 서버 응답으론 못 잡으므로 이 신호를 쓴다.
  const [playerEpoch, setPlayerEpoch] = useState<Record<number, number>>({});
  const [gpuUtilTargetPct, setGpuUtilTargetPct] = useState(
    DEFAULT_GPU_UTIL_TARGET_PCT
  );
  const [gpuUtilDutyPct, setGpuUtilDutyPct] = useState(0);
  const gpuUtilInitializedRef = useRef(false);
  const gpuUtilUpdaterRef = useRef<GpuUtilTargetUpdater | null>(null);

  // ── detection 추론 컨트롤 (deepeye IpcamPage 차용, react-router 제외) ──
  // 카메라별 추론 ON/OFF.
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  // 카메라별 conf fallback (모델 설정 모달의 기본값).
  const [confs, setConfs] = useState<Record<string, number>>({});
  // 카메라별 선택 모델 목록. null=미설정 / []=추론안함 / [m..]=활성.
  const [modelsByCam, setModelsByCam] = useState<Record<string, string[] | null>>({});
  // 카메라별 — 모델별 conf override(사람 단위 임계). 워커 conf PUT 용 — 오버레이엔 전달 안 함.
  const [modelSettingsByCam, setModelSettingsByCam] = useState<
    Record<string, Record<string, ModelSettings>>
  >({});
  const [modalCamKey, setModalCamKey] = useState<string | null>(null); // 모델 관리 모달
  const [confModalCamKey, setConfModalCamKey] = useState<string | null>(null); // 모델 설정 모달

  const fetchCams = useCallback(async () => {
    const resp = await fetch(`${apiBase()}/api/ipcams`);
    if (!resp.ok) return;
    setCams(await resp.json());
  }, []);

  useEffect(() => {
    fetchCams();
  }, [fetchCams]);

  // 등록 cap 을 백엔드에서 1회 로딩 (없으면 fallback 유지).
  useEffect(() => {
    fetch(`${apiBase()}/api/config`)
      .then((r) => (r.ok ? r.json() : null))
      .then((cfg) => {
        if (cfg?.max_ipcams) setMaxIpcams(cfg.max_ipcams);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const updater = new GpuUtilTargetUpdater({
      endpoint: `${apiBase()}/api/inference/config`,
      initialTarget: DEFAULT_GPU_UTIL_TARGET_PCT / 100,
      onAccepted: (config) => {
        setGpuUtilTargetPct(Math.round(config.gpu_util_target * 100));
        setGpuUtilDutyPct(config.gpu_util_duty * 100);
      },
      onRejected: (lastTarget) => {
        setError("GPU 사용률 설정을 저장하지 못했습니다.");
        setGpuUtilTargetPct(Math.round(lastTarget * 100));
      },
    });
    gpuUtilUpdaterRef.current = updater;
    return () => {
      updater.dispose();
      gpuUtilUpdaterRef.current = null;
    };
  }, []);

  // 모델 lane 전체의 실제 busy duty를 1초마다 표시한다. target은 최초 응답으로만
  // 초기화해 polling이 사용자가 드래그 중인 slider를 덮어쓰지 않게 한다.
  useEffect(() => {
    let cancelled = false;
    async function pollGpuUtil() {
      try {
        const response = await fetch(`${apiBase()}/api/inference/config`);
        if (!response.ok) return;
        const rawConfig = (await response.json()) as Partial<GpuUtilConfigStatus>;
        if (cancelled) return;
        const config = normalizeGpuUtilConfig(
          rawConfig,
          DEFAULT_GPU_UTIL_TARGET_PCT / 100
        );
        setGpuUtilDutyPct(config.gpu_util_duty * 100);
        if (!gpuUtilInitializedRef.current) {
          gpuUtilInitializedRef.current = true;
          gpuUtilUpdaterRef.current?.acceptServerTarget(config.gpu_util_target);
          setGpuUtilTargetPct(Math.round(config.gpu_util_target * 100));
        }
      } catch {
        // 카메라/pose 화면은 GPU 상태 endpoint 일시 실패와 독립적으로 유지한다.
      }
    }
    pollGpuUtil();
    const timer = window.setInterval(pollGpuUtil, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    gpuUtilUpdaterRef.current?.schedule(gpuUtilTargetPct / 100);
  }, [gpuUtilTargetPct]);

  // stats 1초 polling — 등록 카메라별 {active, readers} (mediamtx path 상태).
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const entries = await Promise.all(
        cams.map(async (c) => {
          try {
            const resp = await fetch(`${apiBase()}/api/ipcams/${c.stream_key}/stats`);
            if (!resp.ok) return [c.stream_key, { active: false, readers: 0 }] as const;
            return [c.stream_key, (await resp.json()) as Stat] as const;
          } catch {
            return [c.stream_key, { active: false, readers: 0 }] as const;
          }
        })
      );
      if (!cancelled) setStats(Object.fromEntries(entries));
    }
    poll();
    const timer = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [cams]);

  // 카메라별 추론(enabled + conf + models) 상태 초기 fetch + normalize.
  //   UX 규칙: ON + 빈 모델 조합 금지 → models 가 비면 enabled=false 로 normalize, backend 와도 동기화.
  useEffect(() => {
    if (cams.length === 0) return;
    let cancelled = false;
    async function fetchAll() {
      const results = await Promise.all(
        cams.map(async (cam) => {
          try {
            const r = await fetch(`${apiBase()}/api/ipcams/${cam.stream_key}/inference`);
            const data = await r.json();
            return [cam.stream_key, data] as const;
          } catch {
            return [
              cam.stream_key,
              { enabled: true, conf_threshold: null, models: null },
            ] as const;
          }
        })
      );
      if (cancelled) return;
      const normalized = results.map(([k, v]) => {
        const modelsArr = (v.models ?? []) as string[];
        const en = !!v.enabled && modelsArr.length > 0;
        return [k, { enabled: en, conf: v.conf_threshold ?? DEFAULT_CONF, models: modelsArr }] as const;
      });
      setEnabled(Object.fromEntries(normalized.map(([k, v]) => [k, v.enabled])));
      setConfs(Object.fromEntries(normalized.map(([k, v]) => [k, v.conf])));
      setModelsByCam(Object.fromEntries(normalized.map(([k, v]) => [k, v.models])));

      // backend state 와 normalize 결과가 다르면 즉시 PUT 으로 동기화.
      await Promise.all(
        results.map(([k, v]) => {
          const modelsArr = (v.models ?? []) as string[];
          const desiredEn = !!v.enabled && modelsArr.length > 0;
          const needsModelPut = v.models === null || v.models === undefined;
          const needsEnabledPut = !!v.enabled !== desiredEn;
          if (!needsModelPut && !needsEnabledPut) return Promise.resolve();
          const body: Record<string, unknown> = {};
          if (needsModelPut) body.models = [];
          if (needsEnabledPut) body.enabled = desiredEn;
          return fetch(`${apiBase()}/api/ipcams/${k}/inference`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }).catch(() => {});
        })
      );
    }
    fetchAll();
    return () => {
      cancelled = true;
    };
  }, [cams]);

  const online = cams.filter((c) => stats[c.stream_key]?.active).length;
  const atCap = cams.length >= maxIpcams;

  const handleFps = useCallback((key: string, f: number) => {
    setFps((prev) => ({ ...prev, [key]: f }));
  }, []);

  const handleDetFps = useCallback((key: string, f: number) => {
    setDetFps((prev) => ({ ...prev, [key]: f }));
  }, []);

  const toggleInference = async (streamKey: string, on: boolean) => {
    setEnabled((prev) => ({ ...prev, [streamKey]: on })); // 낙관적 업데이트
    try {
      const r = await fetch(`${apiBase()}/api/ipcams/${streamKey}/inference`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: on }),
      });
      const data = await r.json();
      setEnabled((prev) => ({ ...prev, [streamKey]: !!data.enabled }));
    } catch {
      setEnabled((prev) => ({ ...prev, [streamKey]: !on }));
    }
  };

  const handleModelsChange = async (streamKey: string, list: string[]) => {
    setModelsByCam((prev) => ({ ...prev, [streamKey]: list })); // 낙관적
    // UX 규칙: 모델이 모두 해제되면 추론도 자동 OFF (ON+빈 모델 조합 방지).
    const body: Record<string, unknown> = { models: list };
    if (list.length === 0 && (enabled[streamKey] ?? false)) {
      setEnabled((prev) => ({ ...prev, [streamKey]: false }));
      body.enabled = false;
    }
    try {
      await fetch(`${apiBase()}/api/ipcams/${streamKey}/inference`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch {
      /* ignore */
    }
  };

  // 모델 설정 변경 — 클라 settings 갱신 + per-source conf_threshold(워커 conf)를 동기화 PUT.
  // 모달 conf 슬라이더는 워커 추론 conf(사람 detection 임계)를 낮춘다 — 서버측 필터.
  // 워커 conf = 선택 모델 conf 의 최솟값(모든 모델이 원하는 최저 임계값까지 emit).
  const handleSettingsChange = (streamKey: string, next: Record<string, ModelSettings>) => {
    setModelSettingsByCam((prev) => ({ ...prev, [streamKey]: next }));
    const models = modelsByCam[streamKey] ?? [];
    const confVals = models.map((m) => next[m]?.conf ?? DEFAULT_CONF);
    const workerConf = confVals.length ? Math.min(...confVals) : DEFAULT_CONF;
    setConfs((prev) => ({ ...prev, [streamKey]: workerConf }));
    fetch(`${apiBase()}/api/ipcams/${streamKey}/inference`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conf_threshold: workerConf }),
    }).catch(() => {});
  };

  // 등록/수정 — 성공 시 null, 실패 시 에러메시지 반환(모달이 표시 + 로딩상태 제어, #124).
  async function handleSave(name: string, rtspUrl: string): Promise<string | null> {
    if (editCam) {
      // 사용자가 입력한 url 이 기존값(마스킹 :***@ 포함)과 다르면 스트림 변경(비번·주소 등).
      // 이름만 바꾼 편집은 여기서 false → remount 하지 않는다(불필요한 WHEP 재연결/깜빡임 방지).
      const urlChanged = rtspUrl !== editCam.rtsp_url;
      const resp = await fetch(`${apiBase()}/api/ipcams/${editCam.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, rtsp_url: rtspUrl }),
      });
      if (!resp.ok) {
        // 백엔드 detail(예: 마스킹 *** 인 채 주소 변경 → 실비번 재입력 안내)을 그대로 노출.
        return await errorDetail(resp, "카메라 수정에 실패했습니다.");
      }
      // RTSP 변경 성공 시에만 셀 remount 신호 bump(비번-only 편집도 새 자격증명으로 WHEP 재연결).
      if (urlChanged) {
        const id = editCam.id;
        setPlayerEpoch((prev) => ({ ...prev, [id]: (prev[id] ?? 0) + 1 }));
      }
    } else {
      const resp = await fetch(`${apiBase()}/api/ipcams`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, rtsp_url: rtspUrl }),
      });
      if (resp.status === 409) {
        return await errorDetail(resp, `최대 ${maxIpcams}대까지 등록할 수 있습니다`);
      }
      // 409 외 실패(예: 503 mediamtx 미가용, 400 URL 검증)도 backend detail 노출 — PUT 경로와 일관.
      if (!resp.ok) return await errorDetail(resp, "카메라 등록에 실패했습니다.");
    }
    await fetchCams();
    return null;
  }

  async function deleteCam(cam: Cam) {
    if (!window.confirm(`${cam.name} 삭제?`)) return;
    const resp = await fetch(`${apiBase()}/api/ipcams/${cam.id}`, { method: "DELETE" });
    if (!resp.ok) {
      setError("카메라 삭제에 실패했습니다.");
      return;
    }
    await fetchCams();
  }

  // 카메라별 추론 실활성 여부 (추론 ON && 모델 ≥ 1) — 그리드 오버레이/ WS 가 이걸로 켜고 끔.
  const inferenceActive: Record<string, boolean> = Object.fromEntries(
    cams.map((c) => [
      c.stream_key,
      (enabled[c.stream_key] ?? false) && (modelsByCam[c.stream_key]?.length ?? 0) > 0,
    ])
  );

  return (
    <main className="app">
      <header className="page-head">
        <div>
          <h1>RTSP Keypoint</h1>
          <p className="subtitle">카메라 관리 · YOLO 포즈 추정</p>
        </div>
        <button
          className="primary"
          disabled={atCap}
          onClick={() => {
            setEditCam(null);
            setFormOpen(true);
          }}
        >
          + 카메라 등록
        </button>
      </header>

      {error && <p className="form-error">{error}</p>}

      <section className="summary">
        <div className="summary-cell">
          <div className="summary-label">전체 카메라</div>
          <div className="summary-value">{cams.length}</div>
        </div>
        <div className="summary-cell">
          <div className="summary-label">온라인</div>
          <div className="summary-value">
            {online}
            <span className="summary-sub"> / {cams.length}</span>
          </div>
        </div>
      </section>

      <table>
        <thead>
          <tr>
            <th style={{ width: 180 }}>카메라</th>
            <th>RTSP URL</th>
            <th style={{ width: 110 }}>상태</th>
            <th style={{ width: 95 }}>FPS(det)</th>
            <th style={{ width: 260 }}>추론</th>
            <th style={{ width: 140, textAlign: "right" }}>관리</th>
          </tr>
        </thead>
        <tbody>
          {cams.map((cam) => {
            const st = stats[cam.stream_key];
            const active = st?.active ?? false;
            const sel = modelsByCam[cam.stream_key] ?? [];
            const hasModels = sel.length > 0;
            const camEnabled = enabled[cam.stream_key] ?? false;
            const modelText = !hasModels
              ? "모델 없음"
              : sel.length === 1
              ? sel[0]
              : `${sel[0]} +${sel.length - 1}`;
            return (
              <tr key={cam.id}>
                <td>
                  <div className="cam-id">CAM-{String(cam.id).padStart(2, "0")}</div>
                  <div className="cam-name">{cam.name}</div>
                </td>
                <td className="url-cell">{cam.rtsp_url}</td>
                <td>
                  <span className={active ? "status status-on" : "status status-off"}>
                    {active ? "● 온라인" : "● 오프라인"}
                  </span>
                </td>
                <td>{active && fps[cam.stream_key] != null
                  ? `${fps[cam.stream_key].toFixed(1)}${
                      inferenceActive[cam.stream_key] && detFps[cam.stream_key] != null && detFps[cam.stream_key] > 0
                        ? ` (${detFps[cam.stream_key].toFixed(1)})`
                        : ""}`
                  : "—"}</td>
                <td>
                  {/* 모델 선택 → 모델 설정 → 추론 토글 (deepeye 컨트롤 순서) */}
                  <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                    <button onClick={() => setModalCamKey(cam.stream_key)} title={modelText}>
                      모델{hasModels ? ` (${sel.length})` : ""}
                    </button>
                    <button
                      onClick={() => setConfModalCamKey(cam.stream_key)}
                      disabled={!hasModels}
                    >
                      설정
                    </button>
                    <SegmentedToggle
                      enabled={camEnabled}
                      onChange={(on) => toggleInference(cam.stream_key, on)}
                      disabled={!hasModels}
                    />
                  </div>
                </td>
                <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                  <button
                    onClick={() => {
                      setEditCam(cam);
                      setFormOpen(true);
                    }}
                  >
                    수정
                  </button>{" "}
                  <button className="danger" onClick={() => deleteCam(cam)}>
                    삭제
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <section className="gpu-util-control" aria-labelledby="gpu-util-label">
        <div className="gpu-util-copy">
          <label id="gpu-util-label" htmlFor="gpu-util-target">
            최대 GPU 사용 %
          </label>
          <p>카메라 수와 pose 모델 부하에 맞춰 keypoint FPS를 자동 배분합니다.</p>
        </div>
        <div className="gpu-util-slider">
          <input
            id="gpu-util-target"
            type="range"
            min="10"
            max="100"
            step="1"
            value={gpuUtilTargetPct}
            onChange={(event) => {
              setError("");
              setGpuUtilTargetPct(Number(event.target.value));
            }}
          />
          <output htmlFor="gpu-util-target">{gpuUtilTargetPct}%</output>
        </div>
        <div className="gpu-util-duty" aria-live="polite">
          <span>현재 측정</span>
          <strong>{gpuUtilDutyPct.toFixed(0)}%</strong>
        </div>
      </section>

      <section className="live-section">
        <h2 className="grid-heading">실시간 그리드</h2>
        <CameraGrid
          cams={cams}
          onFps={handleFps}
          onDetFps={handleDetFps}
          inferenceActive={inferenceActive}
          epochs={playerEpoch}
        />
      </section>

      <CameraFormModal
        open={formOpen}
        editCam={editCam}
        onClose={() => setFormOpen(false)}
        onSave={handleSave}
      />

      {/* 모델 관리 모달 (preset 선택 + custom .pt 업로드/삭제) */}
      {modalCamKey !== null && (
        <ModelManagerModal
          open={modalCamKey !== null}
          onClose={() => setModalCamKey(null)}
          cameraName={cams.find((c) => c.stream_key === modalCamKey)?.name ?? "카메라"}
          selected={modelsByCam[modalCamKey] ?? []}
          onSelectedChange={(list) => handleModelsChange(modalCamKey, list)}
        />
      )}

      {/* 모델 설정 모달 (사람 단위 conf 슬라이더) */}
      {confModalCamKey !== null && (
        <ModelSettingsModal
          open={confModalCamKey !== null}
          onClose={() => setConfModalCamKey(null)}
          cameraName={cams.find((c) => c.stream_key === confModalCamKey)?.name ?? "카메라"}
          fallbackConf={confs[confModalCamKey] ?? DEFAULT_CONF}
          selectedModels={modelsByCam[confModalCamKey] ?? []}
          settings={modelSettingsByCam[confModalCamKey] ?? {}}
          onSettingsChange={(next) => handleSettingsChange(confModalCamKey, next)}
        />
      )}
    </main>
  );
}
