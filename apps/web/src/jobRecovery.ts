import { REQUEST_DEADLINES, RequestTimeoutError, withRequestDeadline } from "./requestDeadline.ts";

// In-memory recovery only. A retry reads an existing job; it never submits a file.
export type ConversionJob = {
  id: string;
  filename?: string;
  auto_orient_right_angles?: boolean;
  status: "uploading" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
  cancellation_requested?: boolean;
  error?: string | null;
  download_ready?: boolean;
  orientation_review?: unknown;
  page_rotations?: unknown;
};

export type JobSession = Readonly<{
  id: string;
  sourceName: string;
  rotations: readonly string[];
}>;

export type RecoveryCode = "expired" | "invalid-response" | "request-failed" | "request-timeout" | "corrections-unconfirmed";
export class JobRecoveryError extends Error {
  readonly code: RecoveryCode;
  readonly retryable: boolean;

  constructor(code: RecoveryCode, retryable = false) {
    // Do not carry response bodies, engine errors, or network diagnostics into messages.
    super(code);
    this.code = code;
    this.retryable = retryable;
  }
}

const STATUSES = new Set(["uploading", "queued", "running", "succeeded", "failed", "cancelled"]);

/** Reject foreign/malformed job identities before using them in an endpoint or UI. */
export function parseConversionJob(value: unknown, expectedId?: string): ConversionJob {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new JobRecoveryError("invalid-response");
  }
  const data = value as Record<string, unknown>;
  if (typeof data.id !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(data.id) ||
      (expectedId !== undefined && data.id !== expectedId) ||
      typeof data.status !== "string" || !STATUSES.has(data.status)) {
    throw new JobRecoveryError("invalid-response");
  }
  return {
    id: data.id,
    status: data.status as ConversionJob["status"],
    cancellation_requested: data.cancellation_requested === true,
    download_ready: data.download_ready === true,
    error: typeof data.error === "string" ? data.error : null,
    orientation_review: data.orientation_review,
    page_rotations: data.page_rotations,
  };
}

/** Snapshot the acknowledged request, not whatever values the editable form has later. */
export function createJobSession(
  job: ConversionJob,
  sourceName: string,
  rotations: readonly string[],
): JobSession {
  const validated = parseConversionJob(job);
  return Object.freeze({
    id: validated.id,
    sourceName,
    rotations: Object.freeze([...rotations]),
  });
}

async function request(
  url: string,
  signal: AbortSignal,
  fetcher: typeof fetch,
): Promise<Response> {
  signal.throwIfAborted();
  let response: Response;
  try {
    response = await fetcher(url, { method: "GET", signal, cache: "no-store" });
  } catch {
    signal.throwIfAborted();
    throw new JobRecoveryError("request-failed", true);
  }
  signal.throwIfAborted();
  if (response.status === 404 || response.status === 410) {
    throw new JobRecoveryError("expired");
  }
  if (!response.ok) {
    const retryable = [408, 425, 429].includes(response.status) || response.status >= 500;
    throw new JobRecoveryError("request-failed", retryable);
  }
  return response;
}

/** Translate request-local timeout into same-job recovery without aborting its owner. */
async function recoverableRequest<T>(
  signal: AbortSignal,
  milliseconds: number,
  operation: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  try {
    return await withRequestDeadline(signal, milliseconds, operation);
  } catch (error) {
    signal.throwIfAborted();
    if (error instanceof RequestTimeoutError) throw new JobRecoveryError("request-timeout", true);
    throw error;
  }
}

export async function loadJobStatus(
  api: string,
  session: JobSession,
  signal: AbortSignal,
  fetcher: typeof fetch = fetch,
  milliseconds: number = REQUEST_DEADLINES.status,
): Promise<ConversionJob> {
  return recoverableRequest(signal, milliseconds, async requestSignal => {
    const response = await request(
      `${api}/v1/jobs/${encodeURIComponent(session.id)}`, requestSignal, fetcher,
    );
    let data: unknown;
    try {
      data = await response.json();
    } catch (error) {
      requestSignal.throwIfAborted();
      if (error instanceof SyntaxError) throw new JobRecoveryError("invalid-response");
      throw new JobRecoveryError("request-failed", true);
    }
    requestSignal.throwIfAborted();
    return parseConversionJob(data, session.id);
  });
}

export async function loadJobDownload(
  api: string,
  session: JobSession,
  signal: AbortSignal,
  fetcher: typeof fetch = fetch,
  milliseconds: number = REQUEST_DEADLINES.download,
): Promise<{ blob: Blob; filename: string }> {
  return recoverableRequest(signal, milliseconds, async requestSignal => {
    const response = await request(
      `${api}/v1/jobs/${encodeURIComponent(session.id)}/download`, requestSignal, fetcher,
    );
    let blob: Blob;
    try {
      blob = await response.blob();
    } catch {
      requestSignal.throwIfAborted();
      throw new JobRecoveryError("request-failed", true);
    }
    requestSignal.throwIfAborted();
    const disposition = response.headers.get("content-disposition") ?? "";
    const match = disposition.match(/filename="?([^";]+)"?/i);
    const filename = match?.[1] ?? `${session.sourceName.replace(/\.[^.]+$/, "")}-ocr.zip`;
    return { blob, filename };
  });
}

/** A discarded screen/attempt must not schedule one more poll or leave a timer. */
export function waitForNextPoll(signal: AbortSignal, milliseconds = 650): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason);
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}
