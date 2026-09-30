// Per-request limits, including consumption of the response body. Not OCR job limits.
export const REQUEST_DEADLINES = Object.freeze({
  status: 30_000,
  cancellation: 30_000,
  submission: 120_000,
  download: 120_000,
});

export class RequestTimeoutError extends Error {
  constructor() {
    // No request URL, document data, or underlying network error is retained.
    super("request-timeout");
    this.name = "RequestTimeoutError";
  }
}

/**
 * Consume fetch AND its body inside operation; return data, not a live Response.
 * A private signal aborts the request without cancelling its owning UI attempt.
 * Settle independently of fetch so ignored aborts/late streams cannot hold the UI.
 * This does not cancel a server job and never retries any operation itself.
 */
export function withRequestDeadline<T>(
  parent: AbortSignal,
  milliseconds: number,
  operation: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  if (!Number.isSafeInteger(milliseconds) || milliseconds <= 0 || milliseconds > 2 ** 31 - 1) {
    return Promise.reject(new RangeError("Request deadline must be a positive timer-safe integer"));
  }
  if (parent.aborted) return Promise.reject(parent.reason);

  return new Promise<T>((resolve, reject) => {
    const request = new AbortController();
    let settled = false;
    const finish = (complete: () => void, reason?: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      parent.removeEventListener("abort", onAbort);
      // Settle first; a synchronous abort reaction cannot replace the winning result.
      complete();
      request.abort(reason);
    };
    const onAbort = () => finish(() => reject(parent.reason), parent.reason);
    parent.addEventListener("abort", onAbort, { once: true });
    const timer = setTimeout(() => {
      const error = new RequestTimeoutError();
      finish(() => reject(error), error);
    }, milliseconds);

    // Attach both handlers immediately, including to work that finishes after timeout.
    Promise.resolve().then(() => {
      request.signal.throwIfAborted();
      return operation(request.signal);
    }).then(
      value => finish(() => resolve(value)),
      error => finish(() => reject(error), error),
    );
  });
}
