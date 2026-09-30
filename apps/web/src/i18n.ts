export type Locale = "lo" | "en";

export type Messages = {
  resumeConversion: string;
  retryDownload: string;
  recovering: string;
  recoveryHelp: string;
  recoveryUnavailable: string;
  recoveryInvalid: string;
  recoveryInterrupted: string;
  submissionUnknown: string;
  requestTimedOut: string;
  cancellationUnconfirmed: string;
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
  manualRotationsTitle: string;
  manualRotationsHelp: string;
  manualRotationsSyntax: string;
  manualRotationsDuplicate: string;
  manualRotationsImagePage: string;
  manualRotationsNotConfirmed: string;
  orientationReview: (pages: string) => string;
  orientationUnassessed: (pages: string) => string;
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
    requestTimedOut: "The request timed out. Server processing may continue. Check your connection.",
    cancellationUnconfirmed: "Cancellation was not confirmed. The job may still be running. Check its status or try cancellation again.",
    resumeConversion: "Resume this conversion",
    retryDownload: "Retry download",
    recovering: "Checking the existing conversion…",
    recoveryHelp: "Retry uses the existing job and its original file and page corrections. No new upload or OCR run is started. To apply edited settings, start a new conversion.",
    recoveryUnavailable: "This job or its result is no longer available. Submit the original file again to create a new conversion.",
    recoveryInvalid: "The API returned an invalid or different job. No result was downloaded.",
    recoveryInterrupted: "Could not retrieve this conversion. Check the connection and API access.",
    submissionUnknown: "A complete upload confirmation was not received. The server may already be processing it. The app did not retry the upload; converting again creates a new job.",
    manualRotationsTitle: "Manual page corrections (optional)",
    manualRotationsHelp:
      "After reviewing the source, enter page:clockwise degrees, separated by commas. Example: 1:0, 2:270. Use 0, 90, 180 or 270; page numbers start at 1. Listed pages override auto-orientation, including 0. Convert again to apply corrections to the original upload, not the previous export.",
    manualRotationsSyntax: "Use page:degrees, such as 1:0, 2:270. Allowed degrees: 0, 90, 180, 270.",
    manualRotationsDuplicate: "Specify each page only once, even when the angles are identical.",
    manualRotationsImagePage: "Image uploads support page 1 only. Use 1:degrees.",
    manualRotationsNotConfirmed:
      "The API did not confirm your page corrections. No result was downloaded. Update the API and try again.",
    orientationReview: (pages) =>
      `Pages ${pages} may still be sideways. Check the source and exported document before using the result.`,
    orientationUnassessed: (pages) =>
      `Orientation could not be assessed on pages ${pages}. Check those pages before using the result.`,
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
    requestTimedOut: "ໝົດເວລາລໍຖ້າຄຳຕອບ. ເຊີບເວີອາດຍັງປະມວນຜົນຢູ່. ກວດສອບການເຊື່ອມຕໍ່.",
    cancellationUnconfirmed: "ຍັງບໍ່ໄດ້ຢືນຢັນການຍົກເລີກ. ວຽກອາດຍັງດຳເນີນຢູ່. ກວດສອບສະຖານະ ຫຼື ລອງຍົກເລີກອີກຄັ້ງ.",
    resumeConversion: "ສືບຕໍ່ການປ່ຽນເອກະສານນີ້",
    retryDownload: "ດາວໂຫຼດອີກຄັ້ງ",
    recovering: "ກຳລັງກວດສອບການປ່ຽນເອກະສານເດີມ…",
    recoveryHelp: "ການລອງອີກຄັ້ງໃຊ້ວຽກເດີມ ແລະ ການກຳນົດໜ້າເດີມ. ບໍ່ອັບໂຫຼດ ຫຼື ເລີ່ມ OCR ໃໝ່. ເພື່ອໃຊ້ຄ່າທີ່ແກ້ໄຂ ໃຫ້ເລີ່ມການປ່ຽນເອກະສານໃໝ່.",
    recoveryUnavailable: "ວຽກນີ້ ຫຼື ຜົນລັບບໍ່ມີໃຫ້ໃຊ້ແລ້ວ. ສົ່ງໄຟລ໌ຕົ້ນສະບັບອີກຄັ້ງເພື່ອເລີ່ມວຽກໃໝ່.",
    recoveryInvalid: "API ສົ່ງຂໍ້ມູນວຽກທີ່ບໍ່ຖືກຕ້ອງ ຫຼື ບໍ່ແມ່ນວຽກເດີມ. ບໍ່ໄດ້ດາວໂຫຼດຜົນລັບ.",
    recoveryInterrupted: "ບໍ່ສາມາດຮັບຂໍ້ມູນການປ່ຽນເອກະສານໄດ້. ກວດສອບການເຊື່ອມຕໍ່ ແລະ ການເຂົ້າເຖິງ API.",
    submissionUnknown: "ບໍ່ໄດ້ຮັບການຢືນຢັນການອັບໂຫຼດທີ່ຄົບຖ້ວນ. ເຊີບເວີອາດກຳລັງປະມວນຜົນຢູ່. ບໍ່ໄດ້ສົ່ງຊ້ຳອັດຕະໂນມັດ; ການປ່ຽນເອກະສານອີກຄັ້ງຈະສ້າງວຽກໃໝ່.",
    manualRotationsTitle: "ກຳນົດການໝຸນໜ້າດ້ວຍຕົນເອງ (ທາງເລືອກ)",
    manualRotationsHelp:
      "ກວດສອບຕົ້ນສະບັບກ່ອນ ແລ້ວປ້ອນ ເລກໜ້າ:ອົງສາຕາມເຂັມໂມງ ແຍກດ້ວຍເຄື່ອງໝາຍ , . ຕົວຢ່າງ: 1:0, 2:270. ໃຊ້ 0, 90, 180 ຫຼື 270; ເລກໜ້າເລີ່ມຈາກ 1. ໜ້າທີ່ລະບຸຈະບໍ່ໃຊ້ການໝຸນອັດຕະໂນມັດ ລວມທັງ 0. ປ່ຽນເອກະສານອີກຄັ້ງເພື່ອໃຊ້ກັບໄຟລ໌ຕົ້ນສະບັບ ບໍ່ແມ່ນຜົນລັບກ່ອນໜ້າ.",
    manualRotationsSyntax: "ໃຊ້ ເລກໜ້າ:ອົງສາ ເຊັ່ນ 1:0, 2:270. ອົງສາທີ່ຮອງຮັບ: 0, 90, 180, 270.",
    manualRotationsDuplicate: "ລະບຸແຕ່ລະໜ້າພຽງຄັ້ງດຽວ ເຖິງແມ່ນວ່າອົງສາຈະຄືກັນ.",
    manualRotationsImagePage: "ໄຟລ໌ຮູບພາບຮອງຮັບໜ້າ 1 ເທົ່ານັ້ນ. ໃຊ້ 1:ອົງສາ.",
    manualRotationsNotConfirmed:
      "API ບໍ່ໄດ້ຢືນຢັນການໝຸນໜ້າ. ບໍ່ໄດ້ດາວໂຫຼດຜົນລັບ. ກະລຸນາອັບເດດ API ແລ້ວລອງອີກຄັ້ງ.",
    orientationReview: (pages) =>
      `ໜ້າ ${pages} ອາດຍັງຫັນຂ້າງຢູ່. ກະລຸນາກວດສອບຕົ້ນສະບັບ ແລະ ເອກະສານຜົນລັບກ່ອນນຳໃຊ້.`,
    orientationUnassessed: (pages) =>
      `ບໍ່ສາມາດປະເມີນທິດທາງຂອງໜ້າ ${pages} ໄດ້. ກະລຸນາກວດສອບໜ້າເຫຼົ່ານີ້ກ່ອນນຳໃຊ້.`,
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
