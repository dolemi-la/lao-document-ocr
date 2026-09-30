import type { Messages } from "./i18n";

type ReviewMessages = Pick<Messages, "orientationReview" | "orientationUnassessed">;

function pageNumbers(value: unknown, pageCount: number): number[] | null {
  if (!Array.isArray(value)) return null;
  if (!value.every((page) => Number.isSafeInteger(page) && page >= 1 && page <= pageCount)) {
    return null;
  }
  const pages = value as number[];
  if (new Set(pages).size !== pages.length) return null;
  return [...pages].sort((a, b) => a - b);
}

/** Fixed local messages only: never render text supplied inside server metadata. */
export function orientationReviewWarnings(
  jobStatus: string | undefined,
  payload: unknown,
  messages: ReviewMessages,
): string[] {
  if (jobStatus !== "succeeded" || !payload || typeof payload !== "object") return [];
  const review = payload as Record<string, unknown>;
  if (review.version !== "selected-line-axis-review-v1") return [];
  const count = review.page_count;
  if (typeof count !== "number" || !Number.isSafeInteger(count) || count < 0) return [];
  const flagged = pageNumbers(review.review_pages, count);
  const unknown = pageNumbers(review.unassessed_pages, count);
  if (!flagged || !unknown || flagged.some((page) => unknown.includes(page))) return [];

  // Older, absent, or malformed summaries must never become an "all upright" message.
  if (review.status === "not-requested") return [];
  if (typeof review.status !== "string" ||
      !["review-required", "incomplete", "not-assessed", "no-sideways-evidence"].includes(
        review.status,
      )) return [];
  if ((review.status === "review-required") !== (flagged.length > 0)) return [];
  if (review.status === "no-sideways-evidence" && (unknown.length > 0 || count === 0)) return [];
  if (review.status === "incomplete" && unknown.length === 0 && count > 0) return [];
  if (review.status === "not-assessed" && unknown.length !== count) return [];

  const warnings: string[] = [];
  if (flagged.length > 0) warnings.push(messages.orientationReview(flagged.join(", ")));
  if (unknown.length > 0) warnings.push(messages.orientationUnassessed(unknown.join(", ")));
  return warnings;
}
