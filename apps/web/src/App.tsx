import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

import { initialLocale, Locale, MESSAGES, persistLocale } from "./i18n";
import { REQUEST_DEADLINES, RequestTimeoutError, withRequestDeadline } from "./requestDeadline";
import {
  createJobSession, JobRecoveryError, loadJobDownload, loadJobStatus,
  parseConversionJob, waitForNextPoll,
} from "./jobRecovery";
import type { ConversionJob, JobSession } from "./jobRecovery";
import { orientationReviewWarnings } from "./orientationReview";
import { appendPageRotations, confirmsPageRotations, parsePageRotations } from "./pageRotations";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const ACCEPTED = [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"];

type Health = {
  ocr_ready: boolean;
  engine: string;
  required_languages?: string[];
  error: string | null;
};

function App() {
  const [locale, setLocale] = useState<Locale>(initialLocale);
  const [file, setFile] = useState<File | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [autoOrient, setAutoOrient] = useState(false);
  const [manualRotations, setManualRotations] = useState("");
  const [message, setMessage] = useState("");
  const [job, setJob] = useState<ConversionJob | null>(null);
  const [recovery, setRecovery] = useState<{ session: JobSession; download: boolean } | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const sessionRef = useRef<JobSession | null>(null);
  const attemptRef = useRef<AbortController | null>(null);
  const cancelRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false);

  useEffect(() => () => {
    attemptRef.current?.abort();
    cancelRef.current?.abort();
  }, []);
  const m = MESSAGES[locale];
  const rotations = parsePageRotations(manualRotations, file?.name);
  const rotationError = rotations.error ? {
    syntax: m.manualRotationsSyntax,
    duplicate: m.manualRotationsDuplicate,
    imagePage: m.manualRotationsImagePage,
  }[rotations.error] : "";

  useEffect(() => {
    document.documentElement.lang = locale;
    document.title = m.pageTitle;
    persistLocale(locale);
  }, [locale, m.pageTitle]);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const fileDescription = file
    ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`
    : m.fileHelp;

  function choose(next: File | undefined) {
    if (!next || busyRef.current) return;
    const lower = next.name.toLowerCase();
    if (!ACCEPTED.some((extension) => lower.endsWith(extension))) {
      setMessage(m.unsupported);
      return;
    }
    attemptRef.current?.abort();
    attemptRef.current = null;
    cancelRef.current?.abort();
    cancelRef.current = null;
    sessionRef.current = null;
    setRecovery(null);
    setCancelling(false);
    setFile(next);
    setManualRotations("");
    setJob(null);
    setMessage("");
  }

  function onInput(event: ChangeEvent<HTMLInputElement>) {
    choose(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    choose(event.dataTransfer.files?.[0]);
  }

  async function readError(response: Response, fallback: string) {
    const payload = await response.json().catch(() => null);
    return typeof payload?.detail === "string" ? payload.detail : fallback;
  }

  function beginAttempt() {
    attemptRef.current?.abort();
    cancelRef.current?.abort();
    cancelRef.current = null;
    const attempt = new AbortController();
    attemptRef.current = attempt;
    busyRef.current = true;
    setBusy(true);
    setCancelling(false);
    return attempt;
  }

  function isCurrent(attempt: AbortController) {
    return attemptRef.current === attempt && !attempt.signal.aborted;
  }

  function finishAttempt(attempt: AbortController) {
    if (!isCurrent(attempt)) return;
    busyRef.current = false;
    setBusy(false);
  }

  function errorMessage(error: unknown) {
    if (error instanceof RequestTimeoutError) return m.requestTimedOut;
    if (error instanceof JobRecoveryError) {
      if (error.code === "request-timeout") return m.requestTimedOut;
      if (error.code === "expired") return m.recoveryUnavailable;
      if (error.code === "corrections-unconfirmed") return m.manualRotationsNotConfirmed;
      if (error.code === "invalid-response") return m.recoveryInvalid;
      return m.recoveryInterrupted;
    }
    return error instanceof Error ? error.message : m.conversionFailed;
  }

  async function pollJob(
    initial: ConversionJob | null,
    session: JobSession,
    attempt: AbortController,
    downloadRetry = false,
  ) {
    let downloading = downloadRetry;
    try {
      // Recovery always refreshes status before downloading. It never POSTs again.
      let current = initial ?? await loadJobStatus(API_URL, session, attempt.signal);
      while (isCurrent(attempt)) {
        current = parseConversionJob(current, session.id);
        if (!confirmsPageRotations(current.page_rotations, session.rotations)) {
          throw new JobRecoveryError("corrections-unconfirmed");
        }
        if (["succeeded", "failed", "cancelled"].includes(current.status)) {
          // A completed status supersedes a still-pending cancellation request.
          // Its eventual timeout/reply must not overwrite this terminal result.
          cancelRef.current?.abort();
          cancelRef.current = null;
          setCancelling(false);
        }
        setJob(current);
        if (current.status === "succeeded") {
          downloading = true;
          const result = await loadJobDownload(API_URL, session, attempt.signal);
          if (!isCurrent(attempt)) return;
          const url = URL.createObjectURL(result.blob);
          const anchor = document.createElement("a");
          try {
            anchor.href = url;
            anchor.download = result.filename;
            document.body.appendChild(anchor);
            anchor.click();
          } finally {
            anchor.remove();
            URL.revokeObjectURL(url);
          }
          setRecovery(null);
          setMessage(m.done);
          return;
        }
        if (current.status === "cancelled") {
          setRecovery(null);
          setMessage(m.cancelled);
          return;
        }
        if (current.status === "failed") {
          setRecovery(null);
          throw new Error(current.error || m.conversionFailed);
        }
        downloading = false;
        if (current.status === "queued") setMessage(m.queued);
        else setMessage(current.cancellation_requested ? m.cancelling : m.running);
        await waitForNextPoll(attempt.signal);
        current = await loadJobStatus(API_URL, session, attempt.signal);
      }
    } catch (error) {
      if (!isCurrent(attempt)) return;
      setRecovery(error instanceof JobRecoveryError && error.retryable
        ? { session, download: downloading } : null);
      if (error instanceof JobRecoveryError && !error.retryable) {
        // Do not display a stale success/review assessment after expiry or a foreign response.
        setJob(null);
        sessionRef.current = null;
      }
      setMessage(errorMessage(error));
    }
  }

  async function convert() {
    if (!file || busyRef.current) return;
    if (rotationError) {
      setMessage(rotationError);
      return;
    }
    const attempt = beginAttempt();
    sessionRef.current = null;
    setRecovery(null);
    setJob(null);
    setMessage(m.uploading);
    let responseReceived = false;

    try {
      const form = new FormData();
      form.append("file", file);
      appendPageRotations(form, rotations);
      if (autoOrient) {
        form.append("auto_orient_right_angles", "true");
      }
      const created = await withRequestDeadline(
        attempt.signal, REQUEST_DEADLINES.submission, async signal => {
          const response = await fetch(`${API_URL}/v1/jobs`, {
            method: "POST", body: form, signal,
          });
          if (!response.ok) {
            responseReceived = true; // Known rejection, not a confirmed created job.
            throw new Error(await readError(response, m.conversionFailedStatus(response.status)));
          }
          // Successful headers alone do not acknowledge an ID. Bound the JSON body too.
          return parseConversionJob(await response.json());
        },
      );
      if (!isCurrent(attempt)) return;
      responseReceived = true;
      if (!confirmsPageRotations(created.page_rotations, rotations.specs)) {
        // A stale API can ignore unknown fields. Never recover/download that result.
        await withRequestDeadline(
          attempt.signal, REQUEST_DEADLINES.cancellation, async signal => {
            await fetch(`${API_URL}/v1/jobs/${encodeURIComponent(created.id)}`, {
              method: "DELETE", signal,
            });
          },
        ).catch(() => null);
        throw new JobRecoveryError("corrections-unconfirmed");
      }
      const session = createJobSession(created, file.name, rotations.specs);
      sessionRef.current = session;
      await pollJob(created, session, attempt);
    } catch (error) {
      if (isCurrent(attempt)) {
        // No acknowledged ID means no safe same-job retry. Do not automatically resubmit.
        setMessage(responseReceived ? errorMessage(error) : m.submissionUnknown);
      }
    } finally {
      finishAttempt(attempt);
    }
  }

  async function resume() {
    if (!recovery || busyRef.current || sessionRef.current !== recovery.session) return;
    const saved = recovery;
    const attempt = beginAttempt();
    setMessage(m.recovering);
    try {
      await pollJob(null, saved.session, attempt, saved.download);
    } finally {
      finishAttempt(attempt);
    }
  }

  async function cancel() {
    const session = sessionRef.current;
    const owner = attemptRef.current;
    if (!session || !owner || !job || cancelling ||
        !["uploading", "queued", "running"].includes(job.status)) return;
    const request = new AbortController();
    cancelRef.current?.abort();
    cancelRef.current = request;
    setCancelling(true);
    setMessage(m.cancelling);
    const stillCurrent = () => cancelRef.current === request && !request.signal.aborted &&
      attemptRef.current === owner && sessionRef.current === session;
    try {
      const cancelled = await withRequestDeadline(
        request.signal, REQUEST_DEADLINES.cancellation, async signal => {
          const response = await fetch(`${API_URL}/v1/jobs/${encodeURIComponent(session.id)}`, {
            method: "DELETE", signal, cache: "no-store",
          });
          if (!response.ok) {
            throw new Error(await readError(response, m.cancellationFailedStatus(response.status)));
          }
          return parseConversionJob(await response.json(), session.id);
        },
      );
      if (!stillCurrent()) return;
      if (!confirmsPageRotations(cancelled.page_rotations, session.rotations)) {
        throw new JobRecoveryError("corrections-unconfirmed");
      }
      setJob(cancelled);
      if (cancelled.status === "cancelled") {
        owner.abort();
        attemptRef.current = null;
        busyRef.current = false;
        setBusy(false);
        setRecovery(null);
        sessionRef.current = null;
        setMessage(m.cancelled);
      }
    } catch (error) {
      if (stillCurrent()) setMessage(
        error instanceof RequestTimeoutError ? m.cancellationUnconfirmed : errorMessage(error),
      );
    } finally {
      if (cancelRef.current === request && !request.signal.aborted) setCancelling(false);
    }
  }

  const canCancel = Boolean(job && ["uploading", "queued", "running"].includes(job.status));
  const reviewWarnings = orientationReviewWarnings(job?.status, job?.orientation_review, m);

  return (
    <main aria-labelledby="page-heading">
      <section className="shell">
        <header>
          <div className="brandLockup">
            <div className="brand" aria-hidden="true">
              LO
            </div>
            <div>
              <p className="eyebrow">{m.eyebrow}</p>
              <h1>{m.title}</h1>
            </div>
          </div>

          <div className="languageSwitch" role="group" aria-label={m.languageSelector}>
            <button
              type="button"
              className={locale === "lo" ? "active" : ""}
              aria-pressed={locale === "lo"}
              aria-label={m.languageLao}
              lang="lo"
              onClick={() => setLocale("lo")}
            >
              ລາວ
            </button>
            <button
              type="button"
              className={locale === "en" ? "active" : ""}
              aria-pressed={locale === "en"}
              aria-label={m.languageEnglish}
              lang="en"
              onClick={() => setLocale("en")}
            >
              EN
            </button>
          </div>
        </header>

        <div className="hero">
          <div>
            <p className="kicker">{m.kicker}</p>
            <h2 id="page-heading">{m.headline}</h2>
            <p className="lede">{m.lede}</p>
          </div>

          <div
            className="status"
            data-ready={health?.ocr_ready === true}
            role="status"
            aria-live="polite"
          >
            <span className="dot" aria-hidden="true" />
            {health
              ? health.ocr_ready
                ? m.ready(health.engine)
                : m.needsSetup
              : m.checking}
          </div>
        </div>

        <label
          className={`dropzone ${busy ? "disabled" : ""}`}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
          aria-disabled={busy}
        >
          <input
            className="fileInput"
            type="file"
            accept={ACCEPTED.join(",")}
            onChange={onInput}
            disabled={busy}
            aria-describedby="file-help"
          />
          <div className="uploadIcon" aria-hidden="true">
            ↑
          </div>
          <strong>{file ? m.selected : m.drop}</strong>
          <span id="file-help">{fileDescription}</span>
          <span className="browse">{file ? m.chooseAnother : m.browse}</span>
        </label>

        <label className={busy ? "advancedOption disabled" : "advancedOption"}>
          <input
            type="checkbox"
            checked={autoOrient}
            onChange={(event) => setAutoOrient(event.target.checked)}
            disabled={busy}
          />
          <span>
            <strong>{m.autoOrientTitle}</strong>
            <small>{m.autoOrientHelp}</small>
          </span>
        </label>

        <div className="manualRotations">
          <label htmlFor="manual-rotations">{m.manualRotationsTitle}</label>
          <input
            id="manual-rotations"
            type="text"
            dir="ltr"
            value={manualRotations}
            onChange={(event) => setManualRotations(event.target.value)}
            placeholder={file && !file.name.toLowerCase().endsWith(".pdf") ? "1:270" : "1:0, 2:270"}
            disabled={busy || !file}
            maxLength={4096}
            autoComplete="off"
            spellCheck={false}
            aria-invalid={Boolean(rotationError)}
            aria-describedby="manual-rotations-help manual-rotations-error"
          />
          <p id="manual-rotations-help">{m.manualRotationsHelp}</p>
          <p id="manual-rotations-error" className="warning" role="status" aria-live="polite">
            {rotationError}
          </p>
        </div>

        <div className="actions">
          <button
            type="button"
            disabled={!file || busy || Boolean(rotationError) || health?.ocr_ready === false}
            onClick={convert}
            aria-busy={busy}
          >
            {busy ? m.processing : m.convert}
          </button>
          {recovery && (
            <button type="button" className="secondary" onClick={resume} disabled={busy}
              aria-describedby="job-recovery-help" data-testid="resume-job">
              {recovery.download ? m.retryDownload : m.resumeConversion}
            </button>
          )}
          {canCancel && (
            <button type="button" className="secondary" onClick={cancel} disabled={cancelling}>
              {m.cancel}
            </button>
          )}
        </div>

        <div className="announcements" aria-live="polite" aria-atomic="true">
          {recovery && <p id="job-recovery-help" className="message">{m.recoveryHelp}</p>}
          {job && busy && <p className="jobStatus">{m.jobStatus(job.status)}</p>}
          {message && <p className="message">{message}</p>}
          {reviewWarnings.map((warning) => (
            <p className="warning" key={warning}>{warning}</p>
          ))}
        </div>
        {health?.error && (
          <p className="warning" role="alert">
            {health.error}
          </p>
        )}

        <footer>
          <span>{m.outputs}</span>
          <span>{m.license}</span>
        </footer>
      </section>
    </main>
  );
}

export default App;
