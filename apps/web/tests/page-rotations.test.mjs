import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { MESSAGES } from "../src/i18n.ts";
import { orientationReviewWarnings } from "../src/orientationReview.ts";
import {
  appendPageRotations,
  confirmsPageRotations,
  parsePageRotations,
} from "../src/pageRotations.ts";

for (const empty of ["", " ", "\t\n"]) {
  test(`empty input adds no manual fields: ${JSON.stringify(empty)}`, () => {
    const parsed = parsePageRotations(empty, "source.pdf");
    assert.deepEqual(parsed, { specs: [], error: null });
    const form = new FormData();
    appendPageRotations(form, parsed);
    assert.deepEqual([...form.entries()], []);
  });
}

test("repeated form fields retain zero and sort by page number", () => {
  const parsed = parsePageRotations(" 3:180, 1:0, 2:270 ", "original.PDF");
  assert.deepEqual(parsed, { specs: ["1:0", "2:270", "3:180"], error: null });
  const form = new FormData();
  form.append("auto_orient_right_angles", "true");
  appendPageRotations(form, parsed);
  assert.deepEqual(form.getAll("rotate_page"), ["1:0", "2:270", "3:180"]);
  assert.equal(form.get("auto_orient_right_angles"), "true");
});

for (const invalid of [
  "0:90", "-1:90", "01:90", "1:360", "1:-90", "1:90.0", "1.0:90", "1:true",
  "true:90", "1: 90", "1:90 2:270", "1:90,", ",1:90", "1:90,,2:180",
  "١:90", "1e2:90", "9007199254740992:90", "9".repeat(33) + ":90", " ".repeat(4096) + "1:0",
]) {
  test(`reject invalid syntax: ${invalid.slice(0, 40)}`, () => {
    const parsed = parsePageRotations(invalid, "source.pdf");
    assert.equal(parsed.error, "syntax");
    assert.deepEqual(parsed.specs, []);
    const form = new FormData();
    assert.throws(() => appendPageRotations(form, parsed));
    assert.deepEqual([...form.entries()], []);
  });
}

for (const duplicate of ["1:90,1:90", "1:0, 1:270"]) {
  test(`even identical duplicates are rejected: ${duplicate}`, () => {
    assert.deepEqual(parsePageRotations(duplicate), { specs: [], error: "duplicate" });
  });
}

for (const name of ["phone.jpg", "image.PNG", "single.tiff", "photo.webp"]) {
  test(`only page one for current image loader: ${name}`, () => {
    assert.deepEqual(parsePageRotations("1:270", name), { specs: ["1:270"], error: null });
    assert.equal(parsePageRotations("2:90", name).error, "imagePage");
  });
}

test("PDF bounds are checked on the server rather than invented from filenames", () => {
  assert.equal(parsePageRotations("12:90", "document.pdf").error, null);
});

const expected = ["1:0", "2:270"];
const acknowledged = [{ page: 2, degrees_clockwise: 270 }, { page: 1, degrees_clockwise: 0 }];
test("the API must acknowledge the exact map, including zero", () => {
  assert.ok(confirmsPageRotations(acknowledged, expected));
  assert.ok(confirmsPageRotations(acknowledged.map((row) => ({ ...row, secret: "PRIVATE" })), expected));
  assert.deepEqual(acknowledged, [{ page: 2, degrees_clockwise: 270 }, { page: 1, degrees_clockwise: 0 }]);
  assert.ok(confirmsPageRotations(undefined, []));
});

for (const payload of [
  undefined, null, {}, [],
  [{ page: 1, degrees_clockwise: 90 }, { page: 2, degrees_clockwise: 270 }],
  [{ page: 1, degrees_clockwise: 0 }, { page: 1, degrees_clockwise: 0 }],
  [{ page: "1", degrees_clockwise: 0 }, acknowledged[0]],
  [{ page: true, degrees_clockwise: 0 }, acknowledged[0]],
  [{ page: 1, degrees_clockwise: "0" }, acknowledged[0]],
  [{ page: 1, degrees_clockwise: 360 }, acknowledged[0]],
  [null, acknowledged[0]], ["PRIVATE", acknowledged[0]],
]) {
  test(`do not accept missing/mismatched correction acknowledgement: ${JSON.stringify(payload)}`, () => {
    assert.equal(confirmsPageRotations(payload, expected), false);
  });
}

for (const locale of ["lo", "en"]) {
  test(`instructions/errors and manual review remain localized: ${locale}`, () => {
    const messages = MESSAGES[locale];
    for (const key of ["manualRotationsTitle", "manualRotationsHelp", "manualRotationsSyntax",
      "manualRotationsDuplicate", "manualRotationsImagePage", "manualRotationsNotConfirmed"]) {
      assert.equal(typeof messages[key], "string");
      assert.ok(messages[key].length > 0);
      if (locale === "lo") assert.match(messages[key], /[\u0e80-\u0eff]/);
    }
    const summary = { version: "selected-line-axis-review-v1", status: "review-required",
      page_count: 2, review_pages: [1], unassessed_pages: [2] };
    assert.equal(orientationReviewWarnings("succeeded", summary, messages).length, 2);
  });
}

test("App wires the editable field, reset, validation, acknowledgement and resubmission", () => {
  const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
  assert.match(app, /label htmlFor="manual-rotations"/);
  assert.match(app, /aria-describedby="manual-rotations-help manual-rotations-error"/);
  assert.match(app, /disabled=\{busy \|\| !file\}/);
  assert.match(app, /aria-invalid=\{Boolean\(rotationError\)\}/);
  assert.match(app, /setFile\(next\);\s*setManualRotations\(""\)/);
  assert.match(app, /appendPageRotations\(form, rotations\)/);
  assert.match(app, /confirmsPageRotations\(created.page_rotations, rotations.specs\)/);
  assert.match(app, /confirmsPageRotations\(current.page_rotations, expectedRotations\)/);
  assert.match(app, /await pollJob\(created, file, rotations.specs\)/);
  assert.match(app, /form.append\("file", file\)/);
});
