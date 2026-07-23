export interface GpuUtilConfigStatus {
  gpu_util_target: number;
  gpu_util_duty: number;
}

type Fetcher = (
  input: RequestInfo | URL,
  init?: RequestInit
) => Promise<Response>;

interface GpuUtilTargetUpdaterOptions {
  endpoint: string;
  initialTarget: number;
  debounceMs?: number;
  fetcher?: Fetcher;
  onAccepted?: (config: GpuUtilConfigStatus) => void;
  onRejected?: (lastTarget: number) => void;
}

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.max(minimum, Math.min(maximum, value));

export function normalizeGpuUtilConfig(
  value: Partial<GpuUtilConfigStatus>,
  fallbackTarget = 1
): GpuUtilConfigStatus {
  const rawTarget = Number(value.gpu_util_target);
  const rawDuty = Number(value.gpu_util_duty);
  return {
    gpu_util_target: clamp(
      Number.isFinite(rawTarget) ? rawTarget : fallbackTarget,
      0.1,
      1
    ),
    gpu_util_duty: clamp(Number.isFinite(rawDuty) ? rawDuty : 0, 0, 1),
  };
}

/**
 * GPU target PUT의 debounce/abort/rollback 수명주기를 DOM과 분리한다.
 * 컴포넌트 unmount 시 dispose()를 호출하면 timer와 in-flight 요청이 함께 정리된다.
 */
export class GpuUtilTargetUpdater {
  private readonly endpoint: string;
  private readonly debounceMs: number;
  private readonly fetcher: Fetcher;
  private readonly onAccepted?: (config: GpuUtilConfigStatus) => void;
  private readonly onRejected?: (lastTarget: number) => void;
  private lastTarget: number;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private controller: AbortController | null = null;

  constructor(options: GpuUtilTargetUpdaterOptions) {
    this.endpoint = options.endpoint;
    this.debounceMs = Math.max(0, options.debounceMs ?? 200);
    this.fetcher = options.fetcher ?? globalThis.fetch.bind(globalThis);
    this.onAccepted = options.onAccepted;
    this.onRejected = options.onRejected;
    this.lastTarget = normalizeGpuUtilConfig({
      gpu_util_target: options.initialTarget,
    }).gpu_util_target;
  }

  acceptServerTarget(target: number): void {
    this.lastTarget = normalizeGpuUtilConfig({
      gpu_util_target: target,
    }).gpu_util_target;
  }

  schedule(target: number): void {
    const requestedTarget = normalizeGpuUtilConfig({
      gpu_util_target: target,
    }).gpu_util_target;
    if (Math.abs(requestedTarget - this.lastTarget) < 0.0001) {
      this.cancelPending();
      return;
    }

    this.cancelPending();
    const controller = new AbortController();
    this.controller = controller;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.commit(requestedTarget, controller);
    }, this.debounceMs);
  }

  dispose(): void {
    this.cancelPending();
  }

  private cancelPending(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.controller !== null) {
      this.controller.abort();
      this.controller = null;
    }
  }

  private async commit(
    requestedTarget: number,
    controller: AbortController
  ): Promise<void> {
    try {
      const response = await this.fetcher(this.endpoint, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gpu_util_target: requestedTarget }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("gpu util update failed");
      const rawConfig =
        (await response.json()) as Partial<GpuUtilConfigStatus>;
      if (controller.signal.aborted) return;
      const config = normalizeGpuUtilConfig(rawConfig, requestedTarget);
      this.lastTarget = config.gpu_util_target;
      if (this.controller === controller) this.controller = null;
      this.onAccepted?.(config);
    } catch {
      if (!controller.signal.aborted) this.onRejected?.(this.lastTarget);
    } finally {
      if (this.controller === controller) this.controller = null;
    }
  }
}
