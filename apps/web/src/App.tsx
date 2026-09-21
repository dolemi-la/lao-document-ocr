import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const ACCEPTED = [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"];

type Health = {
  ocr_ready: boolean;
  engine: string;
  required_languages: string[];
  error: string | null;
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

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
    if (!next) return;
    const lower = next.name.toLowerCase();
    if (!ACCEPTED.some((extension) => lower.endsWith(extension))) {
      setMessage("Unsupported file type.");
      return;
    }
    setFile(next);
    setMessage("");
  }

  function onInput(event: ChangeEvent<HTMLInputElement>) {
    choose(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    choose(event.dataTransfer.files?.[0]);
  }

  async function convert() {
    if (!file || busy) return;
    setBusy(true);
    setMessage("Reading document and building editable outputs…");

    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_URL}/v1/convert`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? `Conversion failed (${response.status})`);
      }

      const blob = await response.blob();
      const disposition = response.headers.get("content-disposition") ?? "";
      const match = disposition.match(/filename="?([^";]+)"?/i);
      const downloadName = match?.[1] ?? `${file.name.replace(/\.[^.]+$/, "")}-ocr.zip`;

      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = downloadName;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      setMessage("Done. The ZIP contains DOCX, Markdown, TXT and structured JSON.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Conversion failed.");
    } finally {
      setBusy(false);
    }
  }

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
                ? "OCR engine ready · Lao + English"
                : "API online · OCR language data needs setup"
              : "Checking local OCR engine…"}
          </div>
        </div>

        <label
          className="dropzone"
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <input type="file" accept={ACCEPTED.join(",")} onChange={onInput} />
          <div className="uploadIcon">↑</div>
          <strong>{file ? "Document selected" : "Drop a document here"}</strong>
          <span>{fileDescription}</span>
          <span className="browse">{file ? "Choose another file" : "Browse files"}</span>
        </label>

        <button type="button" disabled={!file || busy || health?.ocr_ready === false} onClick={convert}>
          {busy ? "Converting…" : "Convert to editable files"}
        </button>

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
