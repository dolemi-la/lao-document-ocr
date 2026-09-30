import assert from "node:assert/strict";
import { test } from "node:test";
import { MESSAGES } from "../src/i18n.ts";
import {
  JobRecoveryError,
  createJobSession,
  loadJobStatus,
  loadJobDownload,
  parseConversionJob,
  waitForNextPoll,
} from "../src/jobRecovery.ts";

const job = (overrides = {}) => ({ id: "job-a", status: "queued", ...overrides });
const session = () => createJobSession(job(), "original.pdf", ["1:0", "2:270"]);
const controller = () => new AbortController();
const json = (payload, status = 200) => new Response(JSON.stringify(payload), { status });
const code = (name, retryable = false) => (error) => {
  assert.ok(error instanceof JobRecoveryError);
  assert.equal(error.code, name);
  assert.equal(error.retryable, retryable);
  assert.ok(!error.message.includes("PRIVATE"));
  return true;
};

test("recovery freezes original job identity, filename and explicit-zero corrections", () => {
  const specs = ["1:0", "2:270"];
  const original = job();
  const saved = createJobSession(original, "original.pdf", specs);
  original.id = "different-job";
  specs[0] = "1:90";
  assert.deepEqual(saved, { id: "job-a", sourceName: "original.pdf", rotations: ["1:0", "2:270"] });
  assert.ok(Object.isFrozen(saved));
  assert.ok(Object.isFrozen(saved.rotations));
  assert.throws(() => saved.rotations.push("3:90"), TypeError);
});

for (const invalid of [null, [], {}, job({ id: "" }), job({ id: "../other" }),
  job({ id: "a/b" }), job({ id: "a?b" }), job({ id: "x".repeat(129) }),
  job({ id: 123 }), job({ status: "PRIVATE-STATUS" }), job({ status: null })]) {
  test(`malformed job is rejected rather than followed: ${JSON.stringify(invalid)}`, () => {
    assert.throws(() => parseConversionJob(invalid), code("invalid-response"));
  });
}

test("every known job response must belong to the original job", () => {
  assert.throws(() => parseConversionJob(job({ id: "job-b" }), "job-a"), code("invalid-response"));
  assert.equal(parseConversionJob(job(), "job-a").id, "job-a");
});

test("status recovery performs GET on the same job with the abort signal, never POST", async () => {
  const saved = session();
  const signal = controller().signal;
  let requests = 0;
  const result = await loadJobStatus("http://local", saved, signal, async (url, options) => {
    requests++;
    assert.equal(url, "http://local/v1/jobs/job-a");
    assert.equal(options.method, "GET");
    assert.equal(options.cache, "no-store");
    assert.equal(options.signal, signal);
    assert.equal(options.body, undefined);
    return json(job({ status: "succeeded", page_rotations: [{ page: 1, degrees_clockwise: 0 }] }));
  });
  assert.equal(result.status, "succeeded");
  assert.equal(requests, 1);
});

for (const status of [404, 410]) {
  for (const load of [loadJobStatus, loadJobDownload]) {
    test(`${load.name} stops recovery for missing/expired job (${status})`, async () => {
      await assert.rejects(load("http://local", session(), controller().signal,
        async () => json({ detail: "PRIVATE response body" }, status)), code("expired"));
    });
  }
}

for (const status of [408, 425, 429, 500, 502, 503]) {
  test(`temporary HTTP ${status} allows same-job retry`, async () => {
    await assert.rejects(loadJobStatus("http://local", session(), controller().signal,
      async () => json({ detail: "PRIVATE" }, status)), code("request-failed", true));
  });
}
for (const status of [400, 401, 403, 409, 422]) {
  test(`nontransient HTTP ${status} does not suggest blind retries`, async () => {
    await assert.rejects(loadJobStatus("http://local", session(), controller().signal,
      async () => json({ detail: "PRIVATE" }, status)), code("request-failed"));
  });
}
for (const load of [loadJobStatus, loadJobDownload]) {
  test(`${load.name} network failure is retryable without retaining its message`, async () => {
    await assert.rejects(load("http://local", session(), controller().signal,
      async () => { throw new TypeError("PRIVATE network message"); }), code("request-failed", true));
  });
}

test("invalid JSON is not treated as a trusted completed job", async () => {
  await assert.rejects(loadJobStatus("http://local", session(), controller().signal,
    async () => new Response("PRIVATE invalid json")), code("invalid-response"));
});

test("a foreign job cannot be returned by the recovery endpoint", async () => {
  await assert.rejects(loadJobStatus("http://local", session(), controller().signal,
    async () => json(job({ id: "job-b", status: "succeeded" }))), code("invalid-response"));
});

test("download recovery uses the saved source filename and only the saved job", async () => {
  const result = await loadJobDownload("http://local", session(), controller().signal,
    async (url, options) => {
      assert.equal(url, "http://local/v1/jobs/job-a/download");
      assert.equal(options.method, "GET");
    assert.equal(options.cache, "no-store");
      return new Response("fixture archive", { headers: { "Content-Type": "application/zip" } });
    });
  assert.equal(result.filename, "original-ocr.zip");
  assert.equal(await result.blob.text(), "fixture archive");
});

test("a failed response stream is retryable, not a completed download", async () => {
  await assert.rejects(loadJobDownload("http://local", session(), controller().signal,
    async () => ({ ok: true, headers: new Headers(), blob: async () => { throw Error("PRIVATE"); } })),
  code("request-failed", true));
});

test("response filename still honors content disposition", async () => {
  const result = await loadJobDownload("http://local", session(), controller().signal,
    async () => new Response("zip", { headers: { "content-disposition": 'attachment; filename="result.zip"' } }));
  assert.equal(result.filename, "result.zip");
});

test("an already aborted operation cannot initiate a request or timer", async () => {
  const control = controller(); control.abort();
  let calls = 0;
  for (const load of [loadJobStatus, loadJobDownload]) {
    await assert.rejects(load("http://local", session(), control.signal,
      async () => { calls++; return json(job()); }), { name: "AbortError" });
  }
  await assert.rejects(waitForNextPoll(control.signal, 1), { name: "AbortError" });
  assert.equal(calls, 0);
});

test("aborting during polling delay stops the next request promptly", async () => {
  const control = controller();
  const pending = waitForNextPoll(control.signal, 60_000);
  control.abort();
  await assert.rejects(pending, { name: "AbortError" });
});

test("polling delay resolves normally when not cancelled", async () => {
  await waitForNextPoll(controller().signal, 1);
});

test("late response and late blob after abort are discarded even if fetch ignores abort", async () => {
  for (const load of [loadJobStatus, loadJobDownload]) {
    const control = controller();
    await assert.rejects(load("http://local", session(), control.signal, async () => {
      control.abort(); return json(job({ status: "succeeded" }));
    }), { name: "AbortError" });
  }
  const control = controller();
  await assert.rejects(loadJobDownload("http://local", session(), control.signal, async () => ({
    ok: true, headers: new Headers(), blob: async () => { control.abort(); return new Blob(["zip"]); },
  })), { name: "AbortError" });
});


test("broken JSON response stream offers retry but syntax errors do not", async () => {
  await assert.rejects(loadJobStatus("http://local", session(), controller().signal,
    async () => ({ ok: true, json: async () => { throw new TypeError("PRIVATE stream failure"); } })),
  code("request-failed", true));
});

test("job parser retains safe status fields and ignores arbitrary response additions", () => {
  const parsed = parseConversionJob(job({ error: { private: "PRIVATE" },
    cancellation_requested: "true", private: "PRIVATE", download_ready: 1 }));
  assert.equal(parsed.error, null);
  assert.equal(parsed.cancellation_requested, false);
  assert.equal(parsed.download_ready, false);
  assert.equal(parsed.private, undefined);
});


for (const locale of ["lo", "en"]) {
  test(`same-job recovery messages are localized (${locale})`, () => {
    for (const key of ["resumeConversion", "retryDownload", "recovering", "recoveryHelp",
      "recoveryUnavailable", "recoveryInvalid", "recoveryInterrupted", "submissionUnknown"]) {
      assert.equal(typeof MESSAGES[locale][key], "string");
      assert.ok(MESSAGES[locale][key].length > 0);
      if (locale === "lo") assert.match(MESSAGES[locale][key], /[\u0e80-\u0eff]/);
    }
  });
}
