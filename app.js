// よみあげカメラ — ハンズフリー自動読み上げ v2
//
// 「読書をはじめる」を押したあとは操作不要:
//   カメラ映像を監視 → ページが静止したら文字認識 → 新しい文章なら自動で読み上げ。
//   ページをめくるたびにこれを繰り返す。
//
// 文字認識は Gemini API(APIキー設定時、縦書きも高精度)または
// 同梱の Tesseract.js(キー未設定時のフォールバック)。
//
// ジェスチャー(カメラ映像の上で):
//   タップ = 一時停止/再開 / 上下スワイプ = 速さ調整
//   2回タップ = ページ画像を保存 / 3回タップ = しおりを挟む

// ---- 調整パラメータ ----
const SAMPLE_INTERVAL_MS = 400; // フレーム監視の間隔
const SAMPLE_W = 48; // 動き検出用の縮小サイズ
const SAMPLE_H = 64;
const PIXEL_DELTA = 25; // 1画素あたりこれを超える輝度差 = 「変化した画素」とみなす
const MOTION_FRACTION = 0.1; // 変化画素がこの割合を超えたら動いている(ページめくり中)
const STABLE_COUNT = 3; // 静止とみなす連続回数(約1.2秒)
const SCENE_FRACTION = 0.015; // 前回OCRした画面からこの割合以上変化していたら新しいページ
const SIMILARITY_SKIP = 0.75; // 前回読んだ文章とこれ以上似ていたら同じページとみなす
const IDLE_OCR_MIN_MS = 4000; // 次ページ待ち中の定期再認識の最短間隔
const IDLE_OCR_MAX_MS = 15000; // 同(バックオフ後の最長間隔)

const GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"];
const GEMINI_PROMPT =
  "この画像に写っている文章を、書かれている通りにそのまま書き出してください。" +
  "縦書きの場合は右の列から左の列の順に読みます。" +
  "説明・前置き・記号の装飾は一切書かず、本文だけを出力してください。" +
  "読める文章がない場合は何も出力しないでください。";

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
const geminiKeyInput = document.getElementById("gemini-key");
const ocrModeNote = document.getElementById("ocr-mode-note");
const verticalRow = document.getElementById("vertical-row");
const verticalToggle = document.getElementById("vertical-toggle");
const cameraContainer = document.getElementById("camera-container");
const rateOverlay = document.getElementById("rate-overlay");
const pauseOverlay = document.getElementById("pause-overlay");
const flashEl = document.getElementById("flash");
const toastEl = document.getElementById("toast");
const bookmarkList = document.getElementById("bookmark-list");
const noBookmarks = document.getElementById("no-bookmarks");

// ---- 状態 ----
let stream = null;
let running = false;
let paused = false;
let monitorTimer = null;
let ocrBusy = false;
let wakeLock = null;
let ocrWorkerPromise = null;
let verticalMode = false;
let geminiModelIndex = 0;
let geminiBroken = false; // キー無効などでこのセッション中はGeminiを使わない

let prevSample = null; // 直前の監視フレーム(動き検出用)
let stableCount = 0;
let lastProcessedFrame = null; // 最後にOCRした画面(新ページ判定用)
let lastSpokenText = ""; // 最後に読み上げた文章(重複読み防止用)
let recentDiffs = []; // 直近のフレーム間差分(カメラノイズの推定用)
let lastOcrAttempt = 0;
let idleOcrInterval = IDLE_OCR_MIN_MS;

let sentenceQueue = []; // 読み上げ待ちの文
let currentSentence = null; // いま読み上げ中の文(一時停止時にキューへ戻す)
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

// ---- 文字認識(Gemini / Tesseract) ----

function getGeminiKey() {
  return (localStorage.getItem("yomiage_gemini_key") || "").trim();
}

function useGemini() {
  return !geminiBroken && getGeminiKey().length > 0;
}

function updateOcrModeNote() {
  if (useGemini()) {
    ocrModeNote.textContent = "🤖 Gemini AIで認識します(縦書きも高精度)";
    verticalRow.classList.add("hidden");
  } else {
    ocrModeNote.textContent = geminiBroken
      ? "⚠ APIキーが使えないため端末内OCRで動作中です。キーを確認してください。"
      : "いまは端末内OCRで動作中です(縦書きは精度が下がります)";
    verticalRow.classList.remove("hidden");
  }
}

// 撮影フレームを長辺1280px以下のJPEG(base64)に縮小する
function frameToJpegBase64() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  const scale = Math.min(1, 1280 / Math.max(w, h));
  captureCanvas.width = Math.round(w * scale);
  captureCanvas.height = Math.round(h * scale);
  captureCanvas.getContext("2d").drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", 0.8).split(",")[1];
}

async function geminiRecognize() {
  const key = getGeminiKey();
  const body = JSON.stringify({
    contents: [
      {
        parts: [
          { text: GEMINI_PROMPT },
          { inline_data: { mime_type: "image/jpeg", data: frameToJpegBase64() } },
        ],
      },
    ],
    generationConfig: { temperature: 0 },
  });

  for (let attempt = 0; attempt < 3; attempt++) {
    const model = GEMINI_MODELS[geminiModelIndex];
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(key)}`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body }
    );
    if (res.ok) {
      const data = await res.json();
      const parts = data.candidates?.[0]?.content?.parts || [];
      return parts.map((p) => p.text || "").join("");
    }
    if (res.status === 404 && geminiModelIndex < GEMINI_MODELS.length - 1) {
      geminiModelIndex++; // 古いモデル名にフォールバック
      continue;
    }
    if (res.status === 429) {
      await new Promise((r) => setTimeout(r, 3000)); // レート制限: 少し待って再試行
      continue;
    }
    if (res.status === 400 || res.status === 403) {
      geminiBroken = true; // キー無効: 以降は端末内OCRへ
      updateOcrModeNote();
      throw new Error(`Gemini APIキーエラー (${res.status})`);
    }
    throw new Error(`Gemini APIエラー (${res.status})`);
  }
  throw new Error("Gemini APIの再試行が上限に達しました");
}

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

async function tesseractRecognize() {
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCanvas.getContext("2d").drawImage(video, 0, 0);
  const worker = await getOcrWorker();
  const { data } = await worker.recognize(captureCanvas);
  return data.text;
}

// 認識結果の整形(Tesseractの文字間スペース除去、Geminiのコードフェンス除去)
function cleanText(raw) {
  return raw
    .replace(/^```[^\n]*\n?|```\s*$/g, "")
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

// 実機カメラは静止していても常にノイズで揺れるため、
// 直近の差分の中央値をベースラインとして閾値を引き上げる(適応閾値)
function noiseBaseline() {
  if (recentDiffs.length < 5) return 0;
  const sorted = [...recentDiffs].sort((x, y) => x - y);
  return sorted[Math.floor(sorted.length / 2)];
}

function monitorTick() {
  if (!running || !stream || ocrBusy || video.videoWidth === 0) return;

  // 読み上げキューが何らかの理由で止まっていたら再開する(番犬)
  if (
    !paused &&
    sentenceQueue.length > 0 &&
    !speechSynthesis.speaking &&
    !currentSentence
  ) {
    speakNext();
  }

  const frame = grabSample();
  if (prevSample) {
    const diff = changedFraction(frame, prevSample);
    recentDiffs.push(diff);
    if (recentDiffs.length > 12) recentDiffs.shift();
    if (diff > Math.max(MOTION_FRACTION, noiseBaseline() * 3)) {
      stableCount = 0; // ページめくり中・手ブレ中
    } else {
      stableCount++;
    }
  }
  prevSample = frame;

  if (stableCount < STABLE_COUNT) return; // まだ画面が落ち着いていない

  // 速い経路: 前回OCRした画面と十分違うときは新しいページとして即認識
  const sceneThreshold = Math.max(SCENE_FRACTION, noiseBaseline() * 2.5);
  if (
    !lastProcessedFrame ||
    changedFraction(frame, lastProcessedFrame) > sceneThreshold
  ) {
    idleOcrInterval = IDLE_OCR_MIN_MS;
    recognizeCurrentFrame(frame);
    return;
  }

  // 確実な経路: 次のページ待ち(読み上げが空)なら、映像変化を検出できなくても
  // 定期的に認識し直す。同じページなら類似度判定が読み上げをスキップする。
  const waiting =
    !paused && sentenceQueue.length === 0 && !speechSynthesis.speaking;
  if (waiting && Date.now() - lastOcrAttempt > idleOcrInterval) {
    recognizeCurrentFrame(frame);
  }
}

async function recognizeCurrentFrame(frame) {
  ocrBusy = true;
  lastOcrAttempt = Date.now();
  lastProcessedFrame = frame;
  ocrIndicator.classList.remove("hidden");
  updateStatus();

  try {
    let raw;
    if (useGemini()) {
      try {
        raw = await geminiRecognize();
      } catch (err) {
        console.error("Gemini認識エラー:", err);
        raw = await tesseractRecognize(); // 端末内OCRでリカバリー
      }
    } else {
      raw = await tesseractRecognize();
    }
    if (!running) return; // 認識中に「読書をおわる」が押された

    const text = cleanText(raw);
    // 短すぎるものはノイズ(机や手だけが映った等)として無視
    if (text.length >= 4 && similarity(text, lastSpokenText) < SIMILARITY_SKIP) {
      lastSpokenText = text;
      idleOcrInterval = IDLE_OCR_MIN_MS; // 新しいページを見つけたので間隔をリセット
      enqueueText(text);
    } else {
      // 同じページだった: 定期再認識の間隔を少しずつ伸ばす(API節約)
      idleOcrInterval = Math.min(idleOcrInterval + 2000, IDLE_OCR_MAX_MS);
    }
  } catch (err) {
    console.error("文字認識エラー:", err);
    idleOcrInterval = Math.min(idleOcrInterval + 2000, IDLE_OCR_MAX_MS);
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
  if (!speechSynthesis.speaking && !paused) speakNext();
  updateStatus();
}

function speakNext() {
  if (paused || sentenceQueue.length === 0) {
    currentSentence = null;
    updateStatus();
    return;
  }
  currentSentence = sentenceQueue.shift();
  const token = speakToken;

  const utter = new SpeechSynthesisUtterance(currentSentence);
  utter.lang = "ja-JP";
  utter.rate = parseFloat(rateInput.value);
  const voice = pickJapaneseVoice();
  if (voice) utter.voice = voice;

  utter.onend = utter.onerror = () => {
    if (token !== speakToken) return; // スキップ/停止済みの連鎖は打ち切る
    currentSentence = null;
    speakNext();
  };

  speechSynthesis.speak(utter);
  updateStatus();
}

function cancelSpeech() {
  speakToken++;
  sentenceQueue = [];
  currentSentence = null;
  speechSynthesis.cancel();
}

function skipCurrentPage() {
  cancelSpeech();
  paused = false;
  pauseOverlay.classList.add("hidden");
  updateStatus();
}

// タップによる一時停止/再開。
// speechSynthesis.pause() はiOSで不安定なため、読み上げ中の文をキューへ戻して
// cancel() し、再開時にそこから読み直す方式にしている。
function togglePause() {
  if (paused) {
    paused = false;
    pauseOverlay.classList.add("hidden");
    toast("▶ 再開");
    speakNext();
  } else {
    const hasSomething =
      speechSynthesis.speaking || sentenceQueue.length > 0 || currentSentence;
    if (!hasSomething) return; // 何も読んでいないときは何もしない
    paused = true;
    speakToken++;
    if (currentSentence) sentenceQueue.unshift(currentSentence);
    currentSentence = null;
    speechSynthesis.cancel();
    pauseOverlay.classList.remove("hidden");
    toast("⏸ 一時停止");
  }
  updateStatus();
}

// ---- ジェスチャー(カメラ映像の上) ----

let pointerDown = null;
let dragging = false;
let dragStartY = 0;
let dragStartRate = 1;
let tapCount = 0;
let tapTimer = null;
let rateOverlayTimer = null;

function setRate(rate) {
  rateInput.value = rate;
  rateValue.textContent = rate.toFixed(1);
}

function showRateOverlay(rate) {
  rateOverlay.textContent = `${rate.toFixed(1)}×`;
  rateOverlay.classList.remove("hidden");
  clearTimeout(rateOverlayTimer);
  rateOverlayTimer = setTimeout(() => rateOverlay.classList.add("hidden"), 900);
}

cameraContainer.addEventListener("pointerdown", (e) => {
  pointerDown = { x: e.clientX, y: e.clientY, id: e.pointerId };
  dragging = false;
  try {
    cameraContainer.setPointerCapture(e.pointerId);
  } catch (_) {
    /* 非対応環境は無視 */
  }
});

cameraContainer.addEventListener("pointermove", (e) => {
  if (!pointerDown || e.pointerId !== pointerDown.id) return;
  const dy = e.clientY - pointerDown.y;
  const dx = e.clientX - pointerDown.x;
  if (!dragging && Math.abs(dy) > 12 && Math.abs(dy) > Math.abs(dx)) {
    dragging = true; // 縦スワイプ開始 = 速さ調整モード
    dragStartY = e.clientY;
    dragStartRate = parseFloat(rateInput.value);
    tapCount = 0;
    clearTimeout(tapTimer);
  }
  if (dragging) {
    const half = cameraContainer.clientHeight / 2;
    let rate = dragStartRate + ((dragStartY - e.clientY) / half) * 0.5; // 上へ = 速く
    rate = Math.min(2, Math.max(0.5, Math.round(rate * 10) / 10));
    setRate(rate);
    showRateOverlay(rate);
  }
});

function endPointer(e) {
  if (!pointerDown || e.pointerId !== pointerDown.id) return;
  const wasDrag = dragging;
  const moved = Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y);
  pointerDown = null;
  dragging = false;
  if (wasDrag || moved > 10) return;

  // タップ回数の判定: 350ms以内の連打を数え、途切れたら確定
  tapCount++;
  clearTimeout(tapTimer);
  if (tapCount >= 3) {
    tapCount = 0;
    addBookmark();
    return;
  }
  tapTimer = setTimeout(() => {
    const n = tapCount;
    tapCount = 0;
    if (n === 1) togglePause();
    else if (n === 2) savePagePhoto();
  }, 350);
}

cameraContainer.addEventListener("pointerup", endPointer);
cameraContainer.addEventListener("pointercancel", () => {
  pointerDown = null;
  dragging = false;
});

// ---- ページ画像の保存(2回タップ) ----

async function savePagePhoto() {
  if (!stream || video.videoWidth === 0) return;

  // 白フラッシュ演出
  flashEl.classList.remove("hidden");
  setTimeout(() => flashEl.classList.add("hidden"), 200);

  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCanvas.getContext("2d").drawImage(video, 0, 0);
  const blob = await new Promise((r) => captureCanvas.toBlob(r, "image/jpeg", 0.92));
  if (!blob) return;

  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  const file = new File([blob], `yomiage-${stamp}.jpg`, { type: "image/jpeg" });

  // iOS等では共有シート経由で「写真に保存」できる
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file] });
      return;
    } catch (err) {
      if (err.name === "AbortError") return; // ユーザーがキャンセル
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = file.name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  toast("📷 ページを保存しました");
}

// ---- しおり(3回タップ) ----

function getBookmarks() {
  try {
    return JSON.parse(localStorage.getItem("yomiage_bookmarks") || "[]");
  } catch (_) {
    return [];
  }
}

function saveBookmarks(list) {
  localStorage.setItem("yomiage_bookmarks", JSON.stringify(list.slice(0, 100)));
}

function addBookmark() {
  const text = currentText.textContent.trim();
  if (!text) {
    toast("読み上げ中の文章がないため、しおりを挟めません");
    return;
  }
  const list = getBookmarks();
  list.unshift({ text, date: new Date().toISOString() });
  saveBookmarks(list);
  renderBookmarks();
  toast("🔖 しおりを挟みました");
}

function renderBookmarks() {
  const list = getBookmarks();
  noBookmarks.classList.toggle("hidden", list.length > 0);
  bookmarkList.innerHTML = "";
  list.forEach((item, index) => {
    const li = document.createElement("li");
    li.className = "bookmark-item";

    const head = document.createElement("button");
    head.type = "button";
    head.className = "bookmark-head";
    const date = new Date(item.date).toLocaleString("ja-JP", {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
    const snippet = item.text.replace(/\s+/g, " ").slice(0, 40);
    head.textContent = `${date} 「${snippet}${item.text.length > 40 ? "…" : ""}」`;

    const detail = document.createElement("div");
    detail.className = "bookmark-detail hidden";
    const full = document.createElement("p");
    full.textContent = item.text;
    const actions = document.createElement("div");
    actions.className = "bookmark-actions";
    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "btn secondary";
    playBtn.textContent = "🔊 読み上げ";
    playBtn.addEventListener("click", () => {
      cancelSpeech();
      paused = false;
      pauseOverlay.classList.add("hidden");
      enqueueText(item.text);
    });
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "btn secondary";
    delBtn.textContent = "🗑 削除";
    delBtn.addEventListener("click", () => {
      const l = getBookmarks();
      l.splice(index, 1);
      saveBookmarks(l);
      renderBookmarks();
    });
    actions.append(playBtn, delBtn);
    detail.append(full, actions);

    head.addEventListener("click", () => detail.classList.toggle("hidden"));
    li.append(head, detail);
    bookmarkList.appendChild(li);
  });
}

// ---- トースト ----

let toastTimer = null;
function toast(message) {
  toastEl.textContent = message;
  toastEl.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add("hidden"), 1800);
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
  } else if (paused) {
    cls = "idle";
    label = "⏸ 一時停止中";
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

  // 端末内OCRを使う場合のみエンジンを準備(初回は言語データの読み込みに時間がかかる)
  if (!useGemini()) {
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
  }

  running = true;
  paused = false;
  prevSample = null;
  stableCount = 0;
  lastProcessedFrame = null;
  lastSpokenText = "";
  recentDiffs = [];
  lastOcrAttempt = 0;
  idleOcrInterval = IDLE_OCR_MIN_MS;

  startBtn.classList.add("hidden");
  startBtn.disabled = false;
  stopBtn.classList.remove("hidden");
  hint.textContent =
    "ページをめくると自動で読み上げます。スマホは動かさず固定してください。";

  acquireWakeLock();
  monitorTimer = setInterval(monitorTick, SAMPLE_INTERVAL_MS);
  updateStatus();
}

function stopReading() {
  running = false;
  paused = false;
  clearInterval(monitorTimer);
  monitorTimer = null;
  cancelSpeech();
  releaseWakeLock();

  stopBtn.classList.add("hidden");
  startBtn.classList.remove("hidden");
  nowReading.classList.add("hidden");
  ocrIndicator.classList.add("hidden");
  pauseOverlay.classList.add("hidden");
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

geminiKeyInput.value = getGeminiKey();
geminiKeyInput.addEventListener("input", () => {
  localStorage.setItem("yomiage_gemini_key", geminiKeyInput.value.trim());
  geminiBroken = false; // 入力し直したら再度Geminiを試す
  geminiModelIndex = 0;
  updateOcrModeNote();
});

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

// バックグラウンドに回ったときに読み上げが鳴りっぱなしにならないよう止める
// (しおり再生中なども含む)
document.addEventListener("visibilitychange", () => {
  if (document.hidden && !running) cancelSpeech();
});

// 起動時にカメラのプレビューだけ開始しておく
startCamera();
updateOcrModeNote();
renderBookmarks();
updateStatus();
