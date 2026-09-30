import assert from "node:assert/strict";
import { test } from "node:test";
import { getEventListeners } from "node:events";
import {
  REQUEST_DEADLINES, RequestTimeoutError, withRequestDeadline,
} from "../src/requestDeadline.ts";

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const timeout = error => {
  assert.ok(error instanceof RequestTimeoutError);
  assert.equal(error.message, "request-timeout");
  assert.ok(!error.message.includes("PRIVATE"));
  return true;
};

test("production deadlines are fixed per request, not an OCR job lifetime", () => {
  assert.deepEqual(REQUEST_DEADLINES, {
    status: 30_000, cancellation: 30_000, submission: 120_000, download: 120_000,
  });
  assert.ok(Object.isFrozen(REQUEST_DEADLINES));
});

for (const milliseconds of [0, -1, 0.5, NaN, Infinity, "30", true, 2 ** 31]) {
  test(`invalid deadline cannot start work: ${String(milliseconds)}`, async () => {
    let called = false;
    await assert.rejects(withRequestDeadline(new AbortController().signal, milliseconds,
      async () => { called = true; }), RangeError);
    assert.equal(called, false);
  });
}

test("already cancelled operation starts no callback or parent listener", async () => {
  const parent = new AbortController();
  const reason = new Error("PRIVATE cancellation reason");
  parent.abort(reason);
  let called = false;
  await assert.rejects(withRequestDeadline(parent.signal, 100,
    async () => { called = true; }), error => error === reason);
  assert.equal(called, false);
  assert.equal(getEventListeners(parent.signal, "abort").length, 0);
});

test("deadline rejects a never-resolving fetch even when it ignores abort", async () => {
  const parent = new AbortController();
  let child;
  await assert.rejects(withRequestDeadline(parent.signal, 10, signal => {
    child = signal;
    return new Promise(() => {});
  }), timeout);
  assert.notEqual(child, parent.signal);
  assert.equal(child.aborted, true);
  assert.ok(child.reason instanceof RequestTimeoutError);
  assert.equal(parent.signal.aborted, false);
  assert.equal(getEventListeners(parent.signal, "abort").length, 0);
});

for (const body of ["json", "blob"]) {
  test(`deadline includes a stalled ${body} body after successful headers`, async () => {
    let headersRead = false;
    await assert.rejects(withRequestDeadline(new AbortController().signal, 10, async () => {
      const response = { [body]: () => new Promise(() => {}) };
      headersRead = true;
      return await response[body]();
    }), timeout);
    assert.ok(headersRead);
  });
}

test("parent cancellation promptly settles work that ignores abort", async () => {
  const parent = new AbortController();
  let child;
  const waiting = withRequestDeadline(parent.signal, 60_000, signal => {
    child = signal;
    return new Promise(() => {});
  });
  await Promise.resolve();
  const reason = new DOMException("cancelled", "AbortError");
  parent.abort(reason);
  await assert.rejects(waiting, error => error === reason);
  assert.equal(child.aborted, true);
  assert.equal(child.reason, reason);
  assert.equal(getEventListeners(parent.signal, "abort").length, 0);
});

test("cancellation before the scheduled callback prevents its execution", async () => {
  const parent = new AbortController();
  let called = false;
  const waiting = withRequestDeadline(parent.signal, 100, async () => { called = true; });
  parent.abort();
  await assert.rejects(waiting, { name: "AbortError" });
  assert.equal(called, false);
});

test("success clears timer/listener and releases remaining fetch resources", async () => {
  const originalSet = globalThis.setTimeout;
  const originalClear = globalThis.clearTimeout;
  const outstanding = new Set();
  globalThis.setTimeout = (...args) => {
    const timer = originalSet(...args); outstanding.add(timer); return timer;
  };
  globalThis.clearTimeout = timer => { outstanding.delete(timer); originalClear(timer); };
  const parent = new AbortController();
  const unrelated = () => {};
  parent.signal.addEventListener("abort", unrelated);
  let child;
  try {
    const value = await withRequestDeadline(parent.signal, 60_000, async signal => {
      child = signal;
      return { complete: true };
    });
    assert.deepEqual(value, { complete: true });
    assert.equal(outstanding.size, 0);
    assert.equal(child.aborted, true);
    assert.equal(parent.signal.aborted, false);
    assert.deepEqual(getEventListeners(parent.signal, "abort"), [unrelated]);
  } finally {
    for (const timer of outstanding) originalClear(timer);
    globalThis.setTimeout = originalSet;
    globalThis.clearTimeout = originalClear;
    parent.signal.removeEventListener("abort", unrelated);
  }
});

for (const synchronous of [true, false]) {
  test(`operation errors preserve identity and clean resources (${synchronous})`, async () => {
    const parent = new AbortController();
    let child;
    const error = new Error("fixture error");
    await assert.rejects(withRequestDeadline(parent.signal, 60_000, signal => {
      child = signal;
      if (synchronous) throw error;
      return Promise.reject(error);
    }), result => result === error);
    assert.equal(child.aborted, true);
    assert.equal(parent.signal.aborted, false);
    assert.equal(getEventListeners(parent.signal, "abort").length, 0);
  });
}

for (const outcome of ["resolve", "reject"]) {
  test(`late ${outcome} after timeout cannot replace the result or go unhandled`, async () => {
    const work = deferred();
    let consumerRan = false;
    const waiting = withRequestDeadline(new AbortController().signal, 10, () => work.promise)
      .then(() => { consumerRan = true; });
    await assert.rejects(waiting, timeout);
    work[outcome](outcome === "resolve" ? "PRIVATE late data" : new Error("PRIVATE late error"));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(consumerRan, false);
  });
}

test("timing out one request does not abort a sibling or poison recovery", async () => {
  const parent = new AbortController();
  const work = deferred();
  const healthy = withRequestDeadline(parent.signal, 60_000, () => work.promise);
  await assert.rejects(withRequestDeadline(parent.signal, 10, () => new Promise(() => {})), timeout);
  work.resolve("sibling succeeded");
  assert.equal(await healthy, "sibling succeeded");
  assert.equal(await withRequestDeadline(parent.signal, 100, async () => "retry"), "retry");
  assert.equal(parent.signal.aborted, false);
});
