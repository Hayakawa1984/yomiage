// よみあげカメラ
// ページをめくるだけで自動的に文字認識 → 読み上げが続くハンズフリー読書アプリ。
//
// 仕組み:
//   1. カメラ映像を縮小グレースケールで常時サンプリングし、フレーム差分で動きを検出
//   2. ページが静止し、かつ前回OCRした映像から変化していたら新しいページとみなしてOCR
//   3. 認識テキストが直前に読んだ内容と十分違うときだけ読み上げキューへ追加
//   4. 読み上げ中も監視は続くので、先にページをめくれば次ページが順番待ちになる

const video = document.getElementById("video");
const captureCanvas = document.getElementById("capture-canvas");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const skipBtn = document.getElementById("skip-btn");
const statusBadge = document.getElementById("status-badge");
const cameraError = document.getElementById("camera-error");
const retryCameraBtn = document.getElementById("retry-camera");
const initOverlay = document.getElementById("init-overlay");
const ocrIndicator = document.getElementById("ocr-indicator");
const nowReading = document.getElementById("now-reading");
const currentText = document.getElementById("current-text");
const hint = document.getElementById("hint");
const rateInput = document.getElementById("rate");
const rateValue = document.getElementById("rate-value");
const verticalToggle = document.getElementById("vertical-toggle");

// ---- チューニング用定数 ----
const SAMPLE_INTERVAL_MS = 400; // フレーム監視の間隔
const SAMPLE_W = 48;            // 監視用の縮小サイズ
const SAMPLE_H = 64;
const MOTION_THRESHOLD = 6;     // これ以上の差分は「動いている」(0-255の平均絶対差)
const STABLE_SAMPLES = 3;       // 静止とみなす連続サンプル数(≒1.2秒)
const NEW_PAGE_THRESHOLD = 10;  // 前回OCRした映像との差分がこれ以上なら新しいページ
const SIMILARITY_SKIP = 0.75;   // 直前の読み上げとの類似度がこれ以上なら同じページとみなす
const MIN_TEXT_LENGTH = 4;      // これより短い認識結果はノイズとして無視

// ---- 状態 ----
let stream = null;
let running = false;
let sampleTimer = null;
let ocrWorkerPromise = null;
let ocrBusy = false;
let verticalMode = false; // 縦書きの本モード
let wakeLock = null;

const sampleCanvas = document.createElement("canvas");
sampleCanvas.width = SAMPLE_W;
sampleCanvas.height = SAMPLE_H;
const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });

let prevSample = null;        // 直前サンプル(動き検出用)
let stableCount = 0;
let motionSince = false;        // 前回OCR後にページめくり(動き)があったか
let lastProcessedSample = null; // 最後にOCRしたページの映像
let lastSpokenNorm = "";        // 最後に読み上げたテキスト(正規化済み)

let pageQueue = [];   // 読み上げ待ちページ(1ページ = 文の配列)
let speakingPage = []; // いま読んでいるページの残りの文
let speakingNow = false;
let activeUtter = null;

// ---- カメラ ----

async function startCamera() {
  cameraError.classList.add("hidden");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment", // 背面カメラを優先
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.error("カメラ起動エラー:", err);
    cameraError.classList.remove("hidden");
  }
}

// ---- フレーム監視(動き・新ページ検出) ----

function grabSample() {
  if (!video.videoWidth) return null;
  sampleCtx.drawImage(video, 0, 0, SAMPLE_W, SAMPLE_H);
  const { data } = sampleCtx.getImageData(0, 0, SAMPLE_W, SAMPLE_H);
  const gray = new Uint8Array(SAMPLE_W * SAMPLE_H);
  for (let i = 0; i < gray.length; i++) {
    const p = i * 4;
    gray[i] = (data[p] * 3 + data[p + 1] * 6 + data[p + 2]) / 10;
  }
  return gray;
}

function frameDiff(a, b) {
  if (!a || !b) return 255;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

function onSampleTick() {
  if (!running || ocrBusy) return;

  const sample = grabSample();
  if (!sample) return;

  const motion = frameDiff(sample, prevSample);
  prevSample = sample;

  if (motion > MOTION_THRESHOLD) {
    stableCount = 0; // ページめくり中・手ブレ中
    motionSince = true;
    return;
  }

  stableCount++;
  if (stableCount < STABLE_SAMPLES) return;

  // ページが静止した。めくり動作(動き)のあとか、映像が前回OCR時から
  // 変わっていれば読み取る。本のページ同士は「白地に黒文字」で映像差分が
  // 小さいことがあるため、動きの有無を主な判定に使う。
  // (同じページの読み直しはテキスト類似度チェックが防ぐ)
  if (motionSince || frameDiff(sample, lastProcessedSample) > NEW_PAGE_THRESHOLD) {
    motionSince = false;
    lastProcessedSample = sample;
    recognizePage();
  }
}

// ---- OCR ----

function getOcrWorker() {
  if (!ocrWorkerPromise) {
    // 認識エンジン・言語データはすべて同梱(vendor/)しているため、外部への通信は発生しない
    const langs = verticalMode ? "jpn_vert" : "jpn+eng";
    ocrWorkerPromise = Tesseract.createWorker(langs, 1, {
      workerPath: "vendor/worker.min.js",
      corePath: "vendor/core",
      langPath: "vendor/lang",
    }).then(async (worker) => {
      if (verticalMode) {
        await worker.setParameters({
          tessedit_pageseg_mode: Tesseract.PSM.SINGLE_BLOCK_VERT_TEXT,
        });
      }
      return worker;
    });
  }
  return ocrWorkerPromise;
}

// Tesseractの日本語出力に入る文字間の余計な空白を除去する
function cleanText(raw) {
  return raw
    .split("\n")
    .map((line) =>
      line
        .replace(/([　-ヿ㐀-鿿＀-￯])\s+(?=[　-ヿ㐀-鿿＀-￯])/g, "$1")
        .trim()
    )
    .filter((line) => line.length > 0)
    .join("\n");
}

function normalizeForCompare(text) {
  return text.replace(/[\s、。・,.!?！?「」『』()()]/g, "");
}

// 文字bigramのDice係数(0〜1)。同じページを二度読まないための類似度判定
function similarity(a, b) {
  if (!a || !b) return 0;
  if (a === b) return 1;
  const bigrams = (s) => {
    const m = new Map();
    for (let i = 0; i < s.length - 1; i++) {
      const g = s.slice(i, i + 2);
      m.set(g, (m.get(g) || 0) + 1);
    }
    return m;
  };
  const ga = bigrams(a);
  const gb = bigrams(b);
  let overlap = 0;
  for (const [g, n] of ga) if (gb.has(g)) overlap += Math.min(n, gb.get(g));
  const total = (a.length - 1) + (b.length - 1);
  return total > 0 ? (2 * overlap) / total : 0;
}

// 読み上げしやすいように文単位に分割する
function splitSentences(text) {
  return text
    .split(/(?<=[。!?!?])|\n/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

async function recognizePage() {
  ocrBusy = true;
  ocrIndicator.classList.remove("hidden");

  try {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCanvas.getContext("2d").drawImage(video, 0, 0);

    const worker = await getOcrWorker();
    const { data } = await worker.recognize(captureCanvas);
    if (!running) return; // 認識中に停止された

    const text = cleanText(data.text);
    const norm = normalizeForCompare(text);

    if (norm.length < MIN_TEXT_LENGTH) return; // 文字のないページ・ノイズ
    if (similarity(norm, lastSpokenNorm) >= SIMILARITY_SKIP) return; // 同じページ

    lastSpokenNorm = norm;
    pageQueue.push({ text, sentences: splitSentences(text) });
    speakNext();
  } catch (err) {
    console.error("OCRエラー:", err);
  } finally {
    ocrBusy = false;
    ocrIndicator.classList.add("hidden");
    stableCount = 0;
  }
}

// ---- 読み上げ ----

function pickJapaneseVoice() {
  const voices = speechSynthesis.getVoices();
  return (
    voices.find((v) => v.lang === "ja-JP" && v.localService) ||
    voices.find((v) => v.lang === "ja-JP") ||
    voices.find((v) => v.lang.startsWith("ja")) ||
    null
  );
}

function speakNext() {
  if (!running) return;

  if (speakingPage.length === 0) {
    const page = pageQueue.shift();
    if (!page) {
      speakingNow = false;
      setStatus("watching");
      return;
    }
    speakingPage = page.sentences.slice();
    currentText.textContent = page.text;
    nowReading.classList.remove("hidden");
  }

  if (speakingNow) return; // すでに読み上げチェーンが動いている

  speakingNow = true;
  speakSentence();
}

function speakSentence() {
  const sentence = speakingPage.shift();
  if (sentence === undefined) {
    speakingNow = false;
    speakNext(); // 次のページが待っていれば続けて読む
    return;
  }

  setStatus("speaking");

  const utter = new SpeechSynthesisUtterance(sentence);
  utter.lang = "ja-JP";
  utter.rate = parseFloat(rateInput.value);
  const voice = pickJapaneseVoice();
  if (voice) utter.voice = voice;

  activeUtter = utter;
  utter.onend = utter.onerror = () => {
    if (activeUtter !== utter) return; // skip等で無効化済み
    activeUtter = null;
    speakSentence();
  };

  speechSynthesis.speak(utter);
}

function skipPage() {
  speakingPage = [];
  activeUtter = null; // 現在の文のonendを無効化
  speechSynthesis.cancel();
  speakingNow = false;
  speakNext();
}

function stopSpeaking() {
  pageQueue = [];
  speakingPage = [];
  activeUtter = null;
  speakingNow = false;
  speechSynthesis.cancel();
}

// ---- 画面スリープ防止 ----

async function acquireWakeLock() {
  if (!("wakeLock" in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request("screen");
  } catch (err) {
    console.warn("Wake Lockを取得できませんでした:", err);
  }
}

function releaseWakeLock() {
  if (wakeLock) {
    wakeLock.release().catch(() => {});
    wakeLock = null;
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && running) {
    acquireWakeLock(); // バックグラウンド復帰時に再取得
  }
});

// ---- 開始 / 終了 ----

function setStatus(state) {
  const labels = {
    idle: "停止中",
    watching: "ページを待っています",
    speaking: "読み上げ中",
  };
  statusBadge.textContent = labels[state];
  statusBadge.className = `badge ${state}`;
}

async function startReading() {
  if (!stream) {
    await startCamera();
    if (!stream) return;
  }

  // iOSのTTSはユーザー操作起点が必要なので、タップ直後に無音発話でアンロックする
  if ("speechSynthesis" in window) {
    const unlock = new SpeechSynthesisUtterance(" ");
    unlock.volume = 0;
    speechSynthesis.speak(unlock);
    speechSynthesis.getVoices();
  }

  startBtn.disabled = true;
  initOverlay.classList.remove("hidden");
  try {
    await getOcrWorker(); // 初回は言語データのダウンロードが走る
  } catch (err) {
    console.error("認識エンジンの初期化に失敗:", err);
    ocrWorkerPromise = null;
    alert("文字認識エンジンを読み込めませんでした。通信環境を確認して再試行してください。");
    return;
  } finally {
    initOverlay.classList.add("hidden");
    startBtn.disabled = false;
  }

  running = true;
  prevSample = null;
  stableCount = 0;
  motionSince = false;
  lastProcessedSample = null; // 開始時点で映っているページから読み始める
  lastSpokenNorm = "";

  startBtn.classList.add("hidden");
  stopBtn.classList.remove("hidden");
  hint.textContent = "ページをめくると自動で読み上げが続きます。";
  setStatus("watching");

  acquireWakeLock();
  sampleTimer = setInterval(onSampleTick, SAMPLE_INTERVAL_MS);
}

function stopReading() {
  running = false;
  clearInterval(sampleTimer);
  sampleTimer = null;
  stopSpeaking();
  releaseWakeLock();

  stopBtn.classList.add("hidden");
  startBtn.classList.remove("hidden");
  nowReading.classList.add("hidden");
  ocrIndicator.classList.add("hidden");
  hint.innerHTML =
    "本にカメラを向けて「読書をはじめる」を押してください。<br>あとはページをめくるだけで、自動で読み上げが続きます。";
  setStatus("idle");
}

// ---- イベント ----

startBtn.addEventListener("click", startReading);
stopBtn.addEventListener("click", stopReading);
skipBtn.addEventListener("click", skipPage);
retryCameraBtn.addEventListener("click", startCamera);

rateInput.addEventListener("input", () => {
  rateValue.textContent = parseFloat(rateInput.value).toFixed(1);
});

verticalToggle.addEventListener("change", async () => {
  verticalMode = verticalToggle.checked;

  // モードに合った認識エンジンに切り替える(古いワーカーは破棄)
  const oldWorkerPromise = ocrWorkerPromise;
  ocrWorkerPromise = null;
  if (oldWorkerPromise) {
    try {
      (await oldWorkerPromise).terminate();
    } catch (e) {
      /* 破棄失敗は無視 */
    }
  }

  // いま映っているページをあらためて読み取れるようにリセット
  lastProcessedSample = null;
  lastSpokenNorm = "";
  stableCount = 0;
});

// 音声一覧を非同期で読み込むブラウザ対策
if ("speechSynthesis" in window) {
  speechSynthesis.getVoices();
  speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
}

setStatus("idle");
startCamera();

// デバッグ用(開発時のみ使用)
window.__yomiageDebug = () => ({
  running,
  ocrBusy,
  stableCount,
  queueLen: pageQueue.length,
  speakingNow,
  speakingLeft: speakingPage.length,
  lastSpokenNorm,
  videoW: video.videoWidth,
});
