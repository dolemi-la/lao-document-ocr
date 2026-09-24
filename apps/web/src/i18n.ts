export type Locale = "lo" | "en";

export type Messages = {
  pageTitle: string;
  eyebrow: string;
  title: string;
  kicker: string;
  headline: string;
  lede: string;
  languageSelector: string;
  languageLao: string;
  languageEnglish: string;
  ready: (engine: string) => string;
  needsSetup: string;
  checking: string;
  fileHelp: string;
  selected: string;
  drop: string;
  chooseAnother: string;
  browse: string;
  autoOrientTitle: string;
  autoOrientHelp: string;
  convert: string;
  processing: string;
  cancel: string;
  queued: string;
  running: string;
  cancelling: string;
  done: string;
  cancelled: string;
  uploading: string;
  unsupported: string;
  downloadFailed: (status: number) => string;
  jobStatusFailed: (status: number) => string;
  conversionFailedStatus: (status: number) => string;
  cancellationFailedStatus: (status: number) => string;
  conversionFailed: string;
  cancellationFailed: string;
  jobStatus: (status: string) => string;
  outputs: string;
  license: string;
};

export const MESSAGES: Record<Locale, Messages> = {
  en: {
    pageTitle: "Lao Document OCR",
    eyebrow: "OPEN-SOURCE · LOCAL-FIRST",
    title: "Lao Document OCR",
    kicker: "Lao scans and PDFs → editable Word",
    headline: "Turn Lao scans and PDFs into editable documents.",
    lede:
      "No account, no credits, no cloud requirement. Run it yourself and keep your documents on infrastructure you control.",
    languageSelector: "Language",
    languageLao: "Lao",
    languageEnglish: "English",
    ready: (engine) => `OCR engine ready · ${engine}`,
    needsSetup: "API online · OCR engine needs setup",
    checking: "Checking local OCR engine…",
    fileHelp: "PDF, PNG, JPG, TIFF or WebP · up to 25 MB · PDFs up to 60 pages",
    selected: "Document selected",
    drop: "Drop a document here",
    chooseAnother: "Choose another file",
    browse: "Browse files",
    autoOrientTitle: "Auto-fix sideways or upside-down pages",
    autoOrientHelp:
      "Optional. Uses extra OCR only on pages that look ambiguous; leave off for the fastest conversion.",
    convert: "Convert to editable files",
    processing: "Processing…",
    cancel: "Cancel",
    queued: "Queued for local OCR processing…",
    running: "Reading document and building editable outputs…",
    cancelling: "Cancellation requested…",
    done: "Done. The ZIP contains DOCX, Markdown, TXT and structured JSON.",
    cancelled: "Conversion cancelled.",
    uploading: "Uploading document…",
    unsupported: "Unsupported file type.",
    downloadFailed: (status) => `Download failed (${status})`,
    jobStatusFailed: (status) => `Job status failed (${status})`,
    conversionFailedStatus: (status) => `Conversion failed (${status})`,
    cancellationFailedStatus: (status) => `Cancellation failed (${status})`,
    conversionFailed: "Conversion failed.",
    cancellationFailed: "Cancellation failed.",
    jobStatus: (status) => `Job status: ${status}`,
    outputs: "Outputs: .docx · .md · .txt · .json",
    license: "Apache-2.0",
  },
  lo: {
    pageTitle: "Lao Document OCR",
    eyebrow: "ໂອເພນຊອດ · ເນັ້ນການໃຊ້ງານໃນເຄື່ອງ",
    title: "Lao Document OCR",
    kicker: "ຈາກເອກະສານສະແກນ ແລະ PDF → Word ທີ່ແກ້ໄຂໄດ້",
    headline: "ປ່ຽນເອກະສານພາສາລາວເປັນໄຟລ໌ທີ່ແກ້ໄຂໄດ້.",
    lede:
      "ບໍ່ຕ້ອງສ້າງບັນຊີ, ບໍ່ມີເຄຣດິດ, ແລະ ບໍ່ຈຳເປັນຕ້ອງໃຊ້ຄລາວ. ສາມາດຕິດຕັ້ງໃຊ້ເອງ ແລະ ເກັບເອກະສານໄວ້ໃນລະບົບທີ່ທ່ານຄວບຄຸມ.",
    languageSelector: "ພາສາ",
    languageLao: "ລາວ",
    languageEnglish: "ອັງກິດ",
    ready: (engine) => `ເຄື່ອງ OCR ພ້ອມໃຊ້ · ${engine}`,
    needsSetup: "API ອອນລາຍ · ຕ້ອງຕັ້ງຄ່າເຄື່ອງ OCR",
    checking: "ກຳລັງກວດເຄື່ອງ OCR ໃນເຄື່ອງ…",
    fileHelp: "PDF, PNG, JPG, TIFF ຫຼື WebP · ສູງສຸດ 25 MB · PDF ສູງສຸດ 60 ໜ້າ",
    selected: "ເລືອກເອກະສານແລ້ວ",
    drop: "ລາກໄຟລ໌ມາວາງທີ່ນີ້",
    chooseAnother: "ເລືອກໄຟລ໌ອື່ນ",
    browse: "ເລືອກໄຟລ໌",
    autoOrientTitle: "ປັບໜ້າທີ່ຫັນຂ້າງ ຫຼື ກັບຫົວອັດຕະໂນມັດ",
    autoOrientHelp:
      "ຕົວເລືອກເສີມ. ໃຊ້ OCR ເພີ່ມສະເພາະໜ້າທີ່ທິດທາງບໍ່ຊັດເຈນ; ປິດໄວ້ຈະໄວກວ່າ.",
    convert: "ປ່ຽນເປັນໄຟລ໌ແກ້ໄຂໄດ້",
    processing: "ກຳລັງປະມວນຜົນ…",
    cancel: "ຍົກເລີກ",
    queued: "ເຂົ້າຄິວສຳລັບການປະມວນຜົນ OCR ແລ້ວ…",
    running: "ກຳລັງອ່ານເອກະສານ ແລະ ສ້າງໄຟລ໌ທີ່ແກ້ໄຂໄດ້…",
    cancelling: "ສົ່ງຄຳຂໍຍົກເລີກແລ້ວ…",
    done: "ສຳເລັດ. ZIP ມີ DOCX, Markdown, TXT ແລະ JSON ແບບມີໂຄງສ້າງ.",
    cancelled: "ຍົກເລີກການປ່ຽນໄຟລ໌ແລ້ວ.",
    uploading: "ກຳລັງອັບໂຫຼດເອກະສານ…",
    unsupported: "ປະເພດໄຟລ໌ນີ້ບໍ່ຮອງຮັບ.",
    downloadFailed: (status) => `ດາວໂຫຼດບໍ່ສຳເລັດ (${status})`,
    jobStatusFailed: (status) => `ກວດສະຖານະງານບໍ່ສຳເລັດ (${status})`,
    conversionFailedStatus: (status) => `ປ່ຽນໄຟລ໌ບໍ່ສຳເລັດ (${status})`,
    cancellationFailedStatus: (status) => `ຍົກເລີກບໍ່ສຳເລັດ (${status})`,
    conversionFailed: "ປ່ຽນໄຟລ໌ບໍ່ສຳເລັດ.",
    cancellationFailed: "ຍົກເລີກບໍ່ສຳເລັດ.",
    jobStatus: (status) => `ສະຖານະງານ: ${status}`,
    outputs: "ຜົນລັບ: .docx · .md · .txt · .json",
    license: "Apache-2.0",
  },
};

const STORAGE_KEY = "lao-document-ocr-locale";

export function initialLocale(): Locale {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "lo" || stored === "en") {
      return stored;
    }
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
  const preferred = navigator.languages?.[0] ?? navigator.language;
  return preferred?.toLowerCase().startsWith("lo") ? "lo" : "en";
}

export function persistLocale(locale: Locale) {
  try {
    window.localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    // Language still works for the current session without persistence.
  }
}
