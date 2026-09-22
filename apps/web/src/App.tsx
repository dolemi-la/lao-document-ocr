import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";

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
  status: "uploading" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
  cancellation_requested: boolean;
  error: string | null;
  download_ready: boolean;
};

const sleep = (milliseconds: number) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [job, setJob] = useState<ConversionJob | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const fileDescription = useMemo(() => {
    if (!file) return "PDF, PNG, JPG, TIFF or WebP · up to 25 MB · PDFs up to 60 pages";
    return `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  }, [file]);

  function choose(next: File | undefined) {
    if (!next || busy) return;
    const lower = next.name.toLowerCase();
    if (!ACCEPTED.some((extension) => lower.endsWith(extension))) {
      setMessage("Unsupported file type.");
      return;
    }
    setFile(next);
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
      throw new Error(await readError(response, `Download failed (${response.status})`));
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

  async function pollJob(initial: ConversionJob, sourceFile: File) {
    let current = initial;
    setJob(current);

    while (["uploading", "queued", "running"].includes(current.status)) {
      if (current.status === "queued") {
        setMessage("Queued for local OCR processing…");
      } else if (current.status === "running") {
        setMessage(
          current.cancellation_requested
            ? "Cancellation requested…"
            : "Reading document and building editable outputs…",
        );
      }

      await sleep(650);
      const response = await fetch(`${API_URL}/v1/jobs/${current.id}`);
      if (!response.ok) {
        throw new Error(await readError(response, `Job status failed (${response.status})`));
      }
      current = await response.json();
      setJob(current);
    }

    if (current.status === "succeeded") {
      await downloadResult(current, sourceFile);
      setMessage("Done. The ZIP contains DOCX, Markdown, TXT and structured JSON.");
      return;
    }
    if (current.status === "cancelled") {
      setMessage("Conversion cancelled.");
      return;
    }
    throw new Error(current.error || "Conversion failed.");
  }

  async function convert() {
    if (!file || busy) return;
    setBusy(true);
    setJob(null);
    setMessage("Uploading document…");

    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_URL}/v1/jobs`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        throw new Error(await readError(response, `Conversion failed (${response.status})`));
      }

      const created: ConversionJob = await response.json();
      await pollJob(created, file);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Conversion failed.");
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!job || !["uploading", "queued", "running"].includes(job.status)) return;
    setMessage("Cancellation requested…");
    try {
      const response = await fetch(`${API_URL}/v1/jobs/${job.id}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(await readError(response, `Cancellation failed (${response.status})`));
      }
      setJob(await response.json());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Cancellation failed.");
    }
  }

  const canCancel = Boolean(job && ["uploading", "queued", "running"].includes(job.status));

  return (
    <main>
      <section className="shell">
        <header>
          <div className="brand">LO</div>
          <div>
            <p className="eyebrow">OPEN-SOURCE · LOCAL-FIRST</p>
            <h1>Lao Document OCR</h1>
          </div>
        </header>

        <div className="hero">
          <div>
            <p className="kicker">ຈາກເອກະສານສະແກນ → Word ທີ່ແກ້ໄຂໄດ້</p>
            <h2>Turn Lao scans and PDFs into editable documents.</h2>
            <p className="lede">
              No account, no credits, no cloud requirement. Run it yourself and keep your documents
              on infrastructure you control.
            </p>
          </div>

          <div className="status" data-ready={health?.ocr_ready === true}>
            <span className="dot" />
            {health
              ? health.ocr_ready
                ? `OCR engine ready · ${health.engine}`
                : "API online · OCR engine needs setup"
              : "Checking local OCR engine…"}
          </div>
        </div>

        <label
          className={`dropzone ${busy ? "disabled" : ""}`}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <input type="file" accept={ACCEPTED.join(",")} onChange={onInput} disabled={busy} />
          <div className="uploadIcon">↑</div>
          <strong>{file ? "Document selected" : "Drop a document here"}</strong>
          <span>{fileDescription}</span>
          <span className="browse">{file ? "Choose another file" : "Browse files"}</span>
        </label>

        <div className="actions">
          <button
            type="button"
            disabled={!file || busy || health?.ocr_ready === false}
            onClick={convert}
          >
            {busy ? "Processing…" : "Convert to editable files"}
          </button>
          {canCancel && (
            <button type="button" className="secondary" onClick={cancel}>
              Cancel
            </button>
          )}
        </div>

        {job && busy && <p className="jobStatus">Job status: {job.status}</p>}
        {message && <p className="message">{message}</p>}
        {health?.error && <p className="warning">{health.error}</p>}

        <footer>
          <span>Outputs: .docx · .md · .txt · .json</span>
          <span>Apache-2.0</span>
        </footer>
      </section>
    </main>
  );
}

export default App;
