// よみあげカメラ — ハンズフリー自動読み上げ
//
// 「読書をはじめる」を押したあとは操作不要:
//   カメラ映像を監視 → ページが静止したら OCR → 新しい文章なら自動で読み上げ。
//   ページをめくるたびにこれを繰り返す。

// ---- 調整パラメータ ----
const SAMPLE_INTERVAL_MS = 400; // フレーム監視の間隔
const SAMPLE_W = 48; // 動き検出用の縮小サイズ
const SAMPLE_H = 64;
const PIXEL_DELTA = 25; // 1画素あたりこれを超える輝度差 = 「変化した画素」とみなす
const MOTION_FRACTION = 0.1; // 変化画素が10%を超えたら動いている(ページめくり中)
const STABLE_COUNT = 3; // 静止とみなす連続回数(約1.2秒)
const SCENE_FRACTION = 0.015; // 前回OCRした画面から1.5%以上の画素が変化していたら新しいページ
const SIMILARITY_SKIP = 0.75; // 前回読んだ文章とこれ以上似ていたら同じページとみなす

// ---- DOM ----
const video = document.getElementById("video");
const captureCanvas = document.getElementById("capture-canvas");
const statusBadge = document.getElementById("status-badge");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const skipBtn = document.getElementById("skip-btn");
const nowReading = document.getElementById("now-reading");
const currentText = document.getElementById("current-text");
const cameraError = document.getElementById("camera-error");
const retryCameraBtn = document.getElementById("retry-camera");
const initOverlay = document.getElementById("init-overlay");
const ocrIndicator = document.getElementById("ocr-indicator");
const rateInput = document.getElementById("rate");
const rateValue = document.getElementById("rate-value");
const hint = document.getElementById("hint");

// ---- 状態 ----
let stream = null;
let running = false;
let monitorTimer = null;
let ocrBusy = false;
let wakeLock = null;
let ocrWorkerPromise = null;

let prevSample = null; // 直前の監視フレーム(動き検出用)
let stableCount = 0;
let lastProcessedFrame = null; // 最後にOCRした画面(新ページ判定用)
let lastSpokenText = ""; // 最後に読み上げた文章(重複読み防止用)

let sentenceQueue = []; // 読み上げ待ちの文
let speakToken = 0; // スキップ/停止時に古い utterance の連鎖を無効化する

const sampleCanvas = document.createElement("canvas");
sampleCanvas.width = SAMPLE_W;
sampleCanvas.height = SAMPLE_H;
const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });

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
    return true;
  } catch (err) {
    console.error("カメラ起動エラー:", err);
    cameraError.classList.remove("hidden");
    return false;
  }
}

// ---- OCR ----

let verticalMode = false; // 縦書きの本(小説など)を読むモード

function getOcrWorker() {
  if (!ocrWorkerPromise) {
    // 横書き: 日本語+英語 / 縦書き: 縦書き用日本語モデル + 縦一列ブロックのレイアウト指定
    const langs = verticalMode ? "jpn_vert" : "jpn+eng";
    ocrWorkerPromise = Tesseract.createWorker(langs, 1, {
      workerPath: "vendor/worker.min.js",
      corePath: "vendor/core",
      langPath: "vendor/lang",
    })
      .then(async (worker) => {
        if (verticalMode) {
          await worker.setParameters({
            tessedit_pageseg_mode: Tesseract.PSM.SINGLE_BLOCK_VERT_TEXT,
          });
        }
        return worker;
      })
      .catch((err) => {
        ocrWorkerPromise = null; // 失敗したら次回作り直せるように
        throw err;
      });
  }
  return ocrWorkerPromise;
}

async function resetOcrWorker() {
  const old = ocrWorkerPromise;
  ocrWorkerPromise = null;
  if (old) {
    try {
      (await old).terminate();
    } catch (_) {
      /* 生成に失敗していた場合は何もしない */
    }
  }
}

// Tesseract の日本語出力に入る文字間の余計な空白を除去する
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

// ---- テキスト類似度(文字bigramのDice係数) ----

function bigrams(text) {
  const normalized = text.replace(/[\s。、．，!?！?「」『』()（)]/g, "");
  const set = new Set();
  for (let i = 0; i < normalized.length - 1; i++) {
    set.add(normalized.slice(i, i + 2));
  }
  return set;
}

function similarity(a, b) {
  const ba = bigrams(a);
  const bb = bigrams(b);
  if (ba.size === 0 || bb.size === 0) return 0;
  let common = 0;
  for (const g of ba) if (bb.has(g)) common++;
  return (2 * common) / (ba.size + bb.size);
}

// ---- フレーム監視(動き検出・新ページ判定) ----

function grabSample() {
  sampleCtx.drawImage(video, 0, 0, SAMPLE_W, SAMPLE_H);
  const { data } = sampleCtx.getImageData(0, 0, SAMPLE_W, SAMPLE_H);
  const gray = new Uint8ClampedArray(SAMPLE_W * SAMPLE_H);
  for (let i = 0; i < gray.length; i++) {
    const o = i * 4;
    gray[i] = (data[o] + data[o + 1] + data[o + 2]) / 3;
  }
  return gray;
}

// 大きく変化した画素の割合(0〜1)。
// 平均輝度差だと「白いページ上で文字だけ変わった」ケースを検出できないため、
// 画素単位の変化を数える方式にしている。
function changedFraction(a, b) {
  let changed = 0;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > PIXEL_DELTA) changed++;
  }
  return changed / a.length;
}

function monitorTick() {
  if (!running || !stream || ocrBusy || video.videoWidth === 0) return;

  const frame = grabSample();
  if (prevSample) {
    if (changedFraction(frame, prevSample) > MOTION_FRACTION) {
      stableCount = 0; // ページめくり中・手ブレ中
    } else {
      stableCount++;
    }
  }
  prevSample = frame;

  if (stableCount < STABLE_COUNT) return; // まだ画面が落ち着いていない

  // 静止した。前回OCRした画面と十分違うときだけ新しいページとして認識する
  if (
    !lastProcessedFrame ||
    changedFraction(frame, lastProcessedFrame) > SCENE_FRACTION
  ) {
    lastProcessedFrame = frame;
    recognizeCurrentFrame();
  }
}

async function recognizeCurrentFrame() {
  ocrBusy = true;
  ocrIndicator.classList.remove("hidden");
  updateStatus();

  try {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCanvas.getContext("2d").drawImage(video, 0, 0);

    const worker = await getOcrWorker();
    const { data } = await worker.recognize(captureCanvas);
    if (!running) return; // 認識中に「読書をおわる」が押された

    const text = cleanText(data.text);
    // 短すぎるものはノイズ(机や手だけが映った等)として無視
    if (text.length >= 4 && similarity(text, lastSpokenText) < SIMILARITY_SKIP) {
      lastSpokenText = text;
      enqueueText(text);
    }
  } catch (err) {
    console.error("OCRエラー:", err);
  } finally {
    ocrBusy = false;
    ocrIndicator.classList.add("hidden");
    // OCR中に手元が動いていた可能性があるので静止判定をやり直す
    stableCount = 0;
    prevSample = null;
    updateStatus();
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

// 長文を一気に speak すると途切れるブラウザがあるため文単位に分割する
function splitSentences(text) {
  return text
    .split(/(?<=[。!?！?])|\n/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function enqueueText(text) {
  currentText.textContent = text;
  nowReading.classList.remove("hidden");
  sentenceQueue.push(...splitSentences(text));
  if (!speechSynthesis.speaking) speakNext();
  updateStatus();
}

function speakNext() {
  if (!running || sentenceQueue.length === 0) {
    updateStatus();
    return;
  }
  const sentence = sentenceQueue.shift();
  const token = speakToken;

  const utter = new SpeechSynthesisUtterance(sentence);
  utter.lang = "ja-JP";
  utter.rate = parseFloat(rateInput.value);
  const voice = pickJapaneseVoice();
  if (voice) utter.voice = voice;

  utter.onend = utter.onerror = () => {
    if (token !== speakToken) return; // スキップ/停止済みの連鎖は打ち切る
    speakNext();
  };

  speechSynthesis.speak(utter);
  updateStatus();
}

function cancelSpeech() {
  speakToken++;
  sentenceQueue = [];
  speechSynthesis.cancel();
}

function skipCurrentPage() {
  cancelSpeech();
  updateStatus();
}

// ---- 画面スリープ防止 ----

async function acquireWakeLock() {
  if (!("wakeLock" in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request("screen");
  } catch (err) {
    console.warn("Wake Lock 取得失敗:", err);
  }
}

function releaseWakeLock() {
  if (wakeLock) {
    wakeLock.release().catch(() => {});
    wakeLock = null;
  }
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && running) acquireWakeLock();
});

// ---- 状態表示 ----

function updateStatus() {
  let cls, label;
  if (!running) {
    cls = "idle";
    label = "停止中";
  } else if (speechSynthesis.speaking || sentenceQueue.length > 0) {
    cls = "speaking";
    label = "🔊 読み上げ中";
  } else if (ocrBusy) {
    cls = "ocr";
    label = "🔍 文字を認識中";
  } else {
    cls = "watching";
    label = "👀 ページを待っています";
  }
  statusBadge.className = `badge ${cls}`;
  statusBadge.textContent = label;
}

// ---- 開始 / 終了 ----

async function startReading() {
  startBtn.disabled = true;

  if (!stream && !(await startCamera())) {
    startBtn.disabled = false;
    return;
  }

  // iOSのTTSはユーザー操作起点が必要なため、開始タップの中で一度発話しておく
  if ("speechSynthesis" in window) {
    speechSynthesis.getVoices();
    const unlock = new SpeechSynthesisUtterance(" ");
    unlock.volume = 0;
    speechSynthesis.speak(unlock);
  } else {
    alert("このブラウザは音声読み上げに対応していません。");
    startBtn.disabled = false;
    return;
  }

  // OCRエンジンの準備(初回は言語データのダウンロードで時間がかかる)
  initOverlay.classList.remove("hidden");
  try {
    await getOcrWorker();
  } catch (err) {
    console.error("OCRエンジン初期化エラー:", err);
    alert("文字認識エンジンを読み込めませんでした。通信環境を確認して再度お試しください。");
    initOverlay.classList.add("hidden");
    startBtn.disabled = false;
    return;
  }
  initOverlay.classList.add("hidden");

  running = true;
  prevSample = null;
  stableCount = 0;
  lastProcessedFrame = null;
  lastSpokenText = "";

  startBtn.classList.add("hidden");
  startBtn.disabled = false;
  stopBtn.classList.remove("hidden");
  hint.textContent = "ページをめくると自動で読み上げます。スマホは動かさず固定してください。";

  acquireWakeLock();
  monitorTimer = setInterval(monitorTick, SAMPLE_INTERVAL_MS);
  updateStatus();
}

function stopReading() {
  running = false;
  clearInterval(monitorTimer);
  monitorTimer = null;
  cancelSpeech();
  releaseWakeLock();

  stopBtn.classList.add("hidden");
  startBtn.classList.remove("hidden");
  nowReading.classList.add("hidden");
  ocrIndicator.classList.add("hidden");
  currentText.textContent = "";
  hint.innerHTML =
    "本にカメラを向けて「読書をはじめる」を押してください。<br>あとはページをめくるだけで、自動で読み上げが続きます。";
  updateStatus();
}

// ---- イベント ----

startBtn.addEventListener("click", startReading);
stopBtn.addEventListener("click", stopReading);
skipBtn.addEventListener("click", skipCurrentPage);
retryCameraBtn.addEventListener("click", startCamera);

rateInput.addEventListener("input", () => {
  rateValue.textContent = parseFloat(rateInput.value).toFixed(1);
});

const verticalToggle = document.getElementById("vertical-toggle");
verticalToggle.addEventListener("change", async () => {
  verticalMode = verticalToggle.checked;
  await resetOcrWorker(); // 認識モデルを切り替えるためワーカーを作り直す
  // いま映っているページを新モードで読み直せるように判定をリセット
  lastProcessedFrame = null;
  stableCount = 0;
  prevSample = null;
});

// 音声一覧を非同期で読み込むブラウザ向けに、先に読み込みを走らせておく
if ("speechSynthesis" in window) {
  speechSynthesis.getVoices();
}

// 起動時にカメラのプレビューだけ開始しておく
startCamera();
updateStatus();
