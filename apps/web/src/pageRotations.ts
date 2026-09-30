export type RotationError = "syntax" | "duplicate" | "imagePage" | null;
export type RotationInput = { specs: string[]; error: RotationError };

/** Comma-separated UI input; emit the API's repeated, strict PAGE:DEGREES fields. */
export function parsePageRotations(value: string, filename?: string): RotationInput {
  if (!value.trim()) return { specs: [], error: null };
  if (value.length > 4096) return { specs: [], error: "syntax" };
  const values = value.split(",").map((part) => part.trim());
  const seen = new Set<number>();
  const parsed: { page: number; spec: string }[] = [];
  for (const spec of values) {
    const match = /^([1-9][0-9]*):(0|90|180|270)$/.exec(spec);
    if (!match || spec.length > 32) return { specs: [], error: "syntax" };
    const page = Number(match[1]);
    if (!Number.isSafeInteger(page)) return { specs: [], error: "syntax" };
    if (seen.has(page)) return { specs: [], error: "duplicate" };
    // The backend's current image loader processes only page 1, including TIFF.
    if (filename && !filename.toLowerCase().endsWith(".pdf") && page !== 1) {
      return { specs: [], error: "imagePage" };
    }
    seen.add(page);
    parsed.push({ page, spec });
  }
  return { specs: parsed.sort((a, b) => a.page - b.page).map((item) => item.spec), error: null };
}

export function appendPageRotations(form: FormData, input: RotationInput): void {
  if (input.error) throw new Error("Invalid manual page corrections");
  for (const spec of input.specs) form.append("rotate_page", spec);
}

/** Do not silently download an uncorrected result from an older API. */
export function confirmsPageRotations(payload: unknown, expected: string[]): boolean {
  if (expected.length === 0) return true; // Older APIs still support ordinary conversion.
  if (!Array.isArray(payload) || payload.length !== expected.length) return false;
  const observed: string[] = [];
  for (const item of payload) {
    if (!item || typeof item !== "object") return false;
    const { page, degrees_clockwise: angle } = item as Record<string, unknown>;
    if (typeof page !== "number" || !Number.isSafeInteger(page) || page < 1 ||
        typeof angle !== "number" || ![0, 90, 180, 270].includes(angle)) return false;
    observed.push(`${page}:${angle}`);
  }
  return new Set(observed).size === observed.length &&
    observed.every((spec) => expected.includes(spec));
}
