import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { MESSAGES } from "../src/i18n.ts";
import { orientationReviewWarnings } from "../src/orientationReview.ts";

const example = {
  version: "selected-line-axis-review-v1",
  status: "review-required",
  page_count: 3,
  review_pages: [2],
  unassessed_pages: [3],
};

for (const locale of ["lo", "en"]) {
  test(`page-specific review and unknown messages are localized (${locale})`, () => {
    const messages = MESSAGES[locale];
    assert.deepEqual(orientationReviewWarnings("succeeded", example, messages), [
      messages.orientationReview("2"), messages.orientationUnassessed("3"),
    ]);
    assert.notEqual(messages.orientationReview("2"), messages.orientationUnassessed("2"));
  });
}

for (const status of ["running", "queued", "uploading", "cancelled", "failed", undefined]) {
  test(`no completed-result warning for ${status}`, () => {
    assert.deepEqual(orientationReviewWarnings(status, example, MESSAGES.en), []);
  });
}

for (const payload of [null, undefined, {}, { ...example, version: "unknown" }]) {
  test(`missing/legacy/future summaries are tolerated: ${JSON.stringify(payload)}`, () => {
    assert.deepEqual(orientationReviewWarnings("succeeded", payload, MESSAGES.en), []);
  });
}

for (const change of [
  { page_count: true }, { page_count: -1 }, { page_count: 1.5 },
  { review_pages: ["PRIVATE-TEXT"] }, { review_pages: [true] },
  { review_pages: [0] }, { review_pages: [4] }, { review_pages: [2, 2] },
  { review_pages: [] }, { unassessed_pages: [2] },
  { status: "PRIVATE-STATUS" }, { status: { toString: "not callable" } },
]) {
  test(`invalid server summary is not rendered: ${JSON.stringify(change)}`, () => {
    assert.deepEqual(orientationReviewWarnings("succeeded", { ...example, ...change }, MESSAGES.en), []);
  });
}

test("unknown assessment receives caution, while no evidence is not claimed upright", () => {
  const unknown = { ...example, status: "not-assessed", review_pages: [], unassessed_pages: [1, 2, 3] };
  assert.deepEqual(orientationReviewWarnings("succeeded", unknown, MESSAGES.en), [
    MESSAGES.en.orientationUnassessed("1, 2, 3"),
  ]);
  const noEvidence = { ...example, status: "no-sideways-evidence", review_pages: [], unassessed_pages: [] };
  assert.deepEqual(orientationReviewWarnings("succeeded", noEvidence, MESSAGES.en), []);
  assert.deepEqual(orientationReviewWarnings("succeeded", { ...unknown, status: "not-requested" }, MESSAGES.en), []);
});

test("partial assessment warns only about unknown pages", () => {
  const report = { ...example, status: "incomplete", review_pages: [] };
  assert.deepEqual(orientationReviewWarnings("succeeded", report, MESSAGES.en), [
    MESSAGES.en.orientationUnassessed("3"),
  ]);
});

test("warnings ignore metadata text, normalize page order, and do not mutate input", () => {
  const report = { ...example, review_pages: [2, 1], private: "PRIVATE-DOCUMENT", statusMessage: "PRIVATE-ERROR" };
  assert.deepEqual(orientationReviewWarnings("succeeded", report, MESSAGES.en), [
    MESSAGES.en.orientationReview("1, 2"), MESSAGES.en.orientationUnassessed("3"),
  ]);
  assert.deepEqual(report.review_pages, [2, 1]);
});

test("the actual App wires warnings into its persistent live region", () => {
  const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
  assert.match(app, /orientationReviewWarnings\(job\?\.status, job\?\.orientation_review, m\)/);
  const announcements = app.split('className="announcements"')[1].split("</div>")[0];
  assert.match(announcements, /aria-live="polite"/);
  assert.match(announcements, /reviewWarnings\.map/);
  assert.match(announcements, /className="warning"/);
  assert.match(app, /await loadJobDownload\(API_URL, session, attempt.signal\)/);
});
