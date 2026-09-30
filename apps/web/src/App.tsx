import { ChangeEvent, DragEvent, useEffect, useState } from "react";

import { initialLocale, Locale, MESSAGES, persistLocale } from "./i18n";
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

type ConversionJob = {
  id: string;
  filename: string;
  auto_orient_right_angles: boolean;
  status: "uploading" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
  cancellation_requested: boolean;
  error: string | null;
  download_ready: boolean;
  orientation_review?: unknown;
  page_rotations?: unknown;
};

const sleep = (milliseconds: number) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function App() {
  const [locale, setLocale] = useState<Locale>(initialLocale);
  const [file, setFile] = useState<File | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [autoOrient, setAutoOrient] = useState(false);
  const [manualRotations, setManualRotations] = useState("");
  const [message, setMessage] = useState("");
  const [job, setJob] = useState<ConversionJob | null>(null);
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
    if (!next || busy) return;
    const lower = next.name.toLowerCase();
    if (!ACCEPTED.some((extension) => lower.endsWith(extension))) {
      setMessage(m.unsupported);
      return;
    }
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
    return payload?.detail ?? fallback;
  }

  async function downloadResult(currentJob: ConversionJob, sourceFile: File) {
    const response = await fetch(`${API_URL}/v1/jobs/${currentJob.id}/download`);
    if (!response.ok) {
      throw new Error(await readError(response, m.downloadFailed(response.status)));
    }

    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") ?? "";
    const match = disposition.match(/filename="?([^";]+)"?/i);
    const downloadName = match?.[1] ?? `${sourceFile.name.replace(/\.[^.]+$/, "")}-ocr.zip`;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = downloadName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  async function pollJob(initial: ConversionJob, sourceFile: File, expectedRotations: string[]) {
    let current = initial;
    setJob(current);

    while (["uploading", "queued", "running"].includes(current.status)) {
      if (current.status === "queued") {
        setMessage(m.queued);
      } else if (current.status === "running") {
        setMessage(current.cancellation_requested ? m.cancelling : m.running);
      }

      await sleep(650);
      const response = await fetch(`${API_URL}/v1/jobs/${current.id}`);
      if (!response.ok) {
        throw new Error(await readError(response, m.jobStatusFailed(response.status)));
      }
      current = await response.json();
      setJob(current);
    }

    if (current.status === "succeeded") {
      if (!confirmsPageRotations(current.page_rotations, expectedRotations)) {
        throw new Error(m.manualRotationsNotConfirmed);
      }
      await downloadResult(current, sourceFile);
      setMessage(m.done);
      return;
    }
    if (current.status === "cancelled") {
      setMessage(m.cancelled);
      return;
    }
    throw new Error(current.error || m.conversionFailed);
  }

  async function convert() {
    if (!file || busy) return;
    if (rotationError) {
      setMessage(rotationError);
      return;
    }
    setBusy(true);
    setJob(null);
    setMessage(m.uploading);

    try {
      const form = new FormData();
      form.append("file", file);
      appendPageRotations(form, rotations);
      if (autoOrient) {
        form.append("auto_orient_right_angles", "true");
      }
      const response = await fetch(`${API_URL}/v1/jobs`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        throw new Error(await readError(response, m.conversionFailedStatus(response.status)));
      }

      const created: ConversionJob = await response.json();
      if (!confirmsPageRotations(created.page_rotations, rotations.specs)) {
        // A stale API can silently ignore an unknown form field. Never download
        // that result as corrected; attempt to cancel only this submitted job.
        await fetch(`${API_URL}/v1/jobs/${created.id}`, { method: "DELETE" }).catch(() => null);
        throw new Error(m.manualRotationsNotConfirmed);
      }
      await pollJob(created, file, rotations.specs);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : m.conversionFailed);
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!job || !["uploading", "queued", "running"].includes(job.status)) return;
    setMessage(m.cancelling);
    try {
      const response = await fetch(`${API_URL}/v1/jobs/${job.id}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(await readError(response, m.cancellationFailedStatus(response.status)));
      }
      setJob(await response.json());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : m.cancellationFailed);
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
          {canCancel && (
            <button type="button" className="secondary" onClick={cancel}>
              {m.cancel}
            </button>
          )}
        </div>

        <div className="announcements" aria-live="polite" aria-atomic="true">
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
