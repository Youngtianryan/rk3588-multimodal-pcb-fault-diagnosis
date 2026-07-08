const cameraFeed = document.querySelector("#cameraFeed");
const thermalFeed = document.querySelector("#thermalFeed");
const fusionFeed = document.querySelector("#fusionFeed");
const cameraStage = document.querySelector(".camera-stage");
const boardCalOverlay = document.querySelector("#boardCalOverlay");
const thermalCalOverlay = document.querySelector("#thermalCalOverlay");
const cameraPlaceholder = document.querySelector("#cameraPlaceholder");
const thermalPlaceholder = document.querySelector("#thermalPlaceholder");
const cameraStatus = document.querySelector("#cameraStatus");
const thermalStatus = document.querySelector("#thermalStatus");
const modelStatus = document.querySelector("#modelStatus");
const visibleResolution = document.querySelector("#visibleResolution");
const thermalReadout = document.querySelector("#thermalReadout");
const defectCount = document.querySelector("#defectCount");
const runState = document.querySelector("#runState");
const reportBox = document.querySelector("#reportBox");
const generateReportButton = document.querySelector("#generateReport");
const reportLlmMode = document.querySelector("#reportLlmMode");
const logList = document.querySelector("#logList");
const clearLogButton = document.querySelector("#clearLog");
const tempThreshold = document.querySelector("#tempThreshold");
const fusionAlpha = document.querySelector("#fusionAlpha");
const thresholdValue = document.querySelector("#thresholdValue");
const alphaValue = document.querySelector("#alphaValue");
const detectMode = document.querySelector("#detectMode");
const netlistInput = document.querySelector("#netlistInput");
const netlistStatus = document.querySelector("#netlistStatus");
const netlistFileName = document.querySelector("#netlistFileName");
const boardOutlineStatus = document.querySelector("#boardOutlineStatus");
const componentCount = document.querySelector("#componentCount");
const manualAdjustStatus = document.querySelector("#manualAdjustStatus");
const boardCalStatus = document.querySelector("#boardCalStatus");

const API = {
  status: "/api/status",
  visible: "/api/visible.mjpg",
  thermal: "/api/thermal.mjpg",
  fusion: "/api/fusion.mjpg",
  recalibrate: "/api/recalibrate",
  runYoloObb: "/api/run-yolo-obb",
  generateReport: "/api/generate-report",
  reportProgress: "/api/report-progress",
  settings: "/api/settings",
  snapshot: "/api/snapshot.jpg",
  manualBoardPose: "/api/manual-board-pose",
  boardModel: "/api/board-model",
};

const REPORT_MODE_LABELS = {
  auto: "自动",
  api: "DeepSeek API",
  local: "本地 DeepSeek",
};

const boardCalLabels = ["左上 hole3", "右上 hole1", "右下 hole2", "左下 hole4"];

const boardModel = {
  fileName: "",
  components: [],
};

const streamState = {
  cameraConnected: false,
  thermalConnected: false,
  fusionEnabled: true,
  lastStatus: null,
  statusTimer: null,
  manualAdjust: {
    dx: 0,
    dy: 0,
    scaleX: 1,
    scaleY: 1,
    sourceLeft: 0,
    sourceRight: 0,
    sourceTop: 0,
    sourceBottom: 0,
  },
  boardCalibrating: false,
  boardCalTarget: "visible",
  boardCalPoints: [],
  pendingVisibleBoardCal: null,
  pendingThermalBoardCal: null,
  fusionAlertPending: false,
  yoloObbRunning: false,
  reportRunning: false,
  reportTask: null,
  reportProgressTimer: null,
  reportCompletionHandled: false,
  reportRenderSignature: "",
  diagnosticRenderSignature: "",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function addLog(message) {
  const item = document.createElement("li");
  const time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  item.textContent = `${time}  ${message}`;
  logList.prepend(item);
}

function setPillState(element, text, state = "default") {
  element.textContent = text;
  element.classList.remove("online", "warn");
  if (state === "online" || state === "warn") {
    element.classList.add(state);
  }
}

function togglePlaceholder(element, hidden) {
  element.classList.toggle("hidden", hidden);
}

function setImageSource(element, url) {
  element.src = `${url}?ts=${Date.now()}`;
}

function clearImageSource(element) {
  element.removeAttribute("src");
}

function setBoardCalPoints(points = []) {
  streamState.boardCalPoints = points.map((point) => ({ ...point }));
  updateBoardCalUi();
}

function clearCurrentBoardCalibrationStep() {
  streamState.boardCalibrating = false;
  streamState.boardCalPoints = [];
  updateBoardCalUi();
}

function updateRunState() {
  if (streamState.cameraConnected && streamState.thermalConnected) {
    runState.textContent = streamState.fusionEnabled ? "双路在线 / 融合开启" : "双路在线 / 融合关闭";
    return;
  }
  if (streamState.cameraConnected) {
    runState.textContent = "仅可见光";
    return;
  }
  if (streamState.thermalConnected) {
    runState.textContent = "仅热成像";
    return;
  }
  runState.textContent = "空闲";
}

function updateFusionVisibility() {
  fusionFeed.classList.toggle("is-disabled", !streamState.fusionEnabled);
  if (streamState.fusionEnabled) {
    if (!fusionFeed.src) {
      setImageSource(fusionFeed, API.fusion);
    }
    return;
  }
  clearImageSource(fusionFeed);
}

function updateBoardUi(status = streamState.lastStatus) {
  const backendFileName = status?.fusion?.boardSourceFile || "";
  const backendCount = Number(status?.fusion?.componentCount ?? 0);
  const fileName = boardModel.fileName || backendFileName;
  const parsedCount = boardModel.components.length || backendCount;
  netlistStatus.textContent = fileName ? "已导入" : "未导入";
  netlistFileName.textContent = fileName || "未选择";
  boardOutlineStatus.textContent = parsedCount ? "坐标已解析" : "未解析";
  componentCount.textContent = String(parsedCount);
}

function formatManualAdjust(adjust = streamState.manualAdjust) {
  return `dx ${adjust.dx.toFixed(0)} / dy ${adjust.dy.toFixed(0)} / sx ${adjust.scaleX.toFixed(2)} / sy ${adjust.scaleY.toFixed(2)} / L ${adjust.sourceLeft.toFixed(0)} R ${adjust.sourceRight.toFixed(0)}`;
}

function updateManualAdjustUi() {
  manualAdjustStatus.textContent = formatManualAdjust();
}

function updateBoardCalUi() {
  const activeOverlay = streamState.boardCalTarget === "thermal" ? thermalCalOverlay : boardCalOverlay;
  const count = streamState.boardCalPoints.length;
  const lastFusion = streamState.lastStatus?.fusion || {};
  const visibleCalibrating = streamState.boardCalibrating && streamState.boardCalTarget !== "thermal";
  const thermalCalibrating = streamState.boardCalibrating && streamState.boardCalTarget === "thermal";

  cameraStage.classList.toggle("calibrating-visible", visibleCalibrating);
  cameraStage.classList.toggle("calibrating-thermal", thermalCalibrating);

  boardCalOverlay.classList.add("hidden");
  thermalCalOverlay.classList.add("hidden");
  boardCalOverlay.innerHTML = "";
  thermalCalOverlay.innerHTML = "";

  if (streamState.boardCalibrating) {
    activeOverlay.classList.remove("hidden");
    streamState.boardCalPoints.forEach((point, index) => {
      const marker = document.createElement("span");
      marker.className = "board-cal-point";
      marker.dataset.label = boardCalLabels[index];
      marker.style.left = `${point.overlayX * 100}%`;
      marker.style.top = `${point.overlayY * 100}%`;
      activeOverlay.appendChild(marker);
    });
  }

  if (streamState.boardCalibrating) {
    if (count < 4) {
      boardCalStatus.textContent = `${streamState.boardCalTarget === "thermal" ? "热成像" : "可见光"}：请点 ${boardCalLabels[count]}`;
    } else {
      boardCalStatus.textContent = `${streamState.boardCalTarget === "thermal" ? "热成像" : "可见光"}：4/4，点击确定保存`;
    }
    return;
  }

  if (streamState.pendingVisibleBoardCal && streamState.pendingVisibleBoardCal.length === 4) {
    boardCalStatus.textContent = "可见光：待确认";
    return;
  }

  if (streamState.pendingThermalBoardCal && streamState.pendingThermalBoardCal.length === 4) {
    boardCalStatus.textContent = "热成像：待确认";
    return;
  }

  if (lastFusion.manualVisiblePose && lastFusion.manualThermalPose) {
    boardCalStatus.textContent = "四孔已完成，可点击融合";
    return;
  }

  if (lastFusion.manualVisiblePose) {
    boardCalStatus.textContent = "可见光已完成，请标定热成像";
    return;
  }

  boardCalStatus.textContent = "等待开始标定";
}

function renderReport() {
  const status = streamState.lastStatus;
  const reportTask = status?.reportTask || streamState.reportTask;
  const visibleSummary = status?.visible?.online
    ? `${status.visible.width} x ${status.visible.height} @ ${status.visible.fps.toFixed(1)} fps`
    : "离线";
  const thermalSummary = status?.thermal?.online
    ? `${status.thermal.width} x ${status.thermal.height} @ ${status.thermal.fps.toFixed(1)} fps`
    : "离线";
  const fusionSummary = status?.fusion?.online
    ? `${status.fusion.mode} @ ${status.fusion.fps.toFixed(1)} fps`
    : "离线";
  const progressCard = reportTask
    ? `
      <div class="report-card report-progress-card">
        <div class="report-progress-head">
          <h4>报告任务</h4>
          <span>${escapeHtml(reportTask.providerLabel || "--")}</span>
        </div>
        <p>${escapeHtml(reportTask.message || "等待任务")}</p>
        <div class="report-progress-bar"><span style="width:${Math.max(0, Math.min(100, Number(reportTask.progress ?? 0)))}%"></span></div>
        <div class="report-progress-meta">
          <span>阶段：${escapeHtml(reportTask.stage || "--")}</span>
          <span>${Math.round(Number(reportTask.progress ?? 0))}%</span>
        </div>
      </div>
    `
    : "";

  const reportHtml = `
    ${progressCard}
    <div class="report-grid">
      <div class="report-card">
        <h4>当前模式</h4>
        <p>${detectMode.value}</p>
      </div>
      <div class="report-stat-grid">
        <div class="report-stat">
          <span>可见光</span>
          <strong>${visibleSummary}</strong>
        </div>
        <div class="report-stat">
          <span>热成像</span>
          <strong>${thermalSummary}</strong>
        </div>
        <div class="report-stat">
          <span>融合</span>
          <strong>${fusionSummary}</strong>
        </div>
      </div>
      <div class="report-card">
        <h4>标定状态</h4>
        <p>${boardCalStatus.textContent}</p>
      </div>
    </div>
  `;
  if (streamState.reportRenderSignature === reportHtml) {
    return;
  }
  streamState.reportRenderSignature = reportHtml;
  reportBox.innerHTML = reportHtml;
  streamState.diagnosticRenderSignature = "";
}

function renderDiagnosticReportCard() {
  const status = streamState.lastStatus;
  const hotspot = status?.fusion?.hotspotAnalysis;
  const reportTask = status?.reportTask || streamState.reportTask;
  const nearest = hotspot?.nearestComponent;
  const netHtml = Array.isArray(nearest?.nets) && nearest.nets.length
    ? nearest.nets
        .slice(0, 6)
        .map((item) => `<span class="tag">${escapeHtml(item.net)} / pin ${escapeHtml(item.pin ?? "--")}</span>`)
        .join("")
    : `<span class="tag">暂无网表信息</span>`;
  const cardHtml = `
    <div class="report-card diagnostic-card">
      <h4>诊断摘要</h4>
      <p class="report-lead">
        ${nearest
          ? `最近器件：<strong>${escapeHtml(nearest.ref)}</strong> / ${escapeHtml(nearest.footprint || "--")} / ${escapeHtml(nearest.comment || "--")}`
          : "当前还没有稳定匹配到器件"}
      </p>
      <p>${nearest ? `匹配方式：${escapeHtml(nearest.matchMode || "nearest-component")} / 距离：${escapeHtml(nearest.distanceMm)} mm` : "请先完成融合并锁定热区位置"}</p>
      <p>大模型来源：${escapeHtml(reportTask?.providerLabel || "--")}</p>
      <div class="report-tags">${netHtml}</div>
      <p class="report-note">详细原因、概率和器件资料请直接生成诊断 PDF，由大模型结合器件信息输出。</p>
    </div>
  `;
  if (streamState.diagnosticRenderSignature === cardHtml) {
    return;
  }
  streamState.diagnosticRenderSignature = cardHtml;
  const existing = reportBox.querySelector(".diagnostic-card");
  if (existing) {
    existing.outerHTML = cardHtml;
  } else {
    reportBox.insertAdjacentHTML("beforeend", cardHtml);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch(API.status, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const status = await response.json();
    streamState.lastStatus = status;
    streamState.reportTask = status.reportTask || streamState.reportTask;
    updateBoardUi(status);

    if (status.fusion?.manualAdjust) {
      streamState.manualAdjust = {
        dx: Number(status.fusion.manualAdjust.dx ?? 0),
        dy: Number(status.fusion.manualAdjust.dy ?? 0),
        scaleX: Number(status.fusion.manualAdjust.scaleX ?? 1),
        scaleY: Number(status.fusion.manualAdjust.scaleY ?? 1),
        sourceLeft: Number(status.fusion.manualAdjust.sourceLeft ?? 0),
        sourceRight: Number(status.fusion.manualAdjust.sourceRight ?? 0),
        sourceTop: Number(status.fusion.manualAdjust.sourceTop ?? 0),
        sourceBottom: Number(status.fusion.manualAdjust.sourceBottom ?? 0),
      };
      updateManualAdjustUi();
    }

    visibleResolution.textContent = status.visible.online
      ? `${status.visible.width} x ${status.visible.height} / ${status.visible.fps.toFixed(1)} fps`
      : "--";
    thermalReadout.textContent = status.thermal.online
      ? `${status.thermal.width} x ${status.thermal.height} / ${status.thermal.fps.toFixed(1)} fps`
      : "--";
    defectCount.textContent = status.fusion?.online
      ? `${status.fusion.mode} / board:${status.fusion.boardPose ?? "--"} / refs:${status.fusion.componentCount ?? 0} / alpha:${Math.round((status.fusion.alpha ?? 0.42) * 100)}% / hotspot:${status.fusion.hotspot ? status.fusion.hotspot.map((v) => Math.round(v)).join(",") : "--"}`
      : "fusion offline";

    setPillState(
      cameraStatus,
      status.visible.online ? "可见光在线" : "可见光离线",
      status.visible.online ? "online" : "warn",
    );
    setPillState(
      thermalStatus,
      status.thermal.online ? "热成像在线" : "热成像离线",
      status.thermal.online ? "online" : "warn",
    );
    setPillState(
      modelStatus,
      status.fusion?.online
        ? `融合 ${status.fusion.mode}${status.fusion?.yoloObb?.enabled ? ` / OBB ${status.fusion.yoloObb.count ?? 0}` : ""}`
        : "融合离线",
      status.fusion?.online ? "online" : "warn",
    );

    if (
      streamState.fusionAlertPending &&
      status.fusion?.boardPose === "manual-holes-pair" &&
      status.fusion?.mode === "visual-locked-overlay"
    ) {
      streamState.fusionAlertPending = false;
      alert("融合完成");
      addLog("融合完成");
      if (streamState.fusionEnabled) {
        setImageSource(fusionFeed, API.fusion);
      }
    }

    updateBoardCalUi();
    renderReport();
    renderDiagnosticReportCard();
  } catch (error) {
    setPillState(cameraStatus, "服务器异常", "warn");
    setPillState(thermalStatus, "服务器异常", "warn");
    setPillState(modelStatus, "服务器异常", "warn");
  }
}

async function runYoloObbDetection() {
  if (streamState.yoloObbRunning) {
    addLog("YOLOv8 OBB 正在运行");
    return;
  }
  streamState.yoloObbRunning = true;
  setPillState(modelStatus, "YOLOv8 OBB 运行中", "warn");
  try {
    const response = await fetch(API.runYoloObb, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const payload = await response.json();
    if (!payload.ok) {
      throw new Error(payload.error || "YOLOv8 OBB failed");
    }
    const count = Number(payload.result?.count ?? 0);
    addLog(`YOLOv8 OBB 完成，检测到 ${count} 个目标`);
    alert(`YOLOv8 OBB 完成，检测到 ${count} 个目标`);
    await refreshStatus();
    if (streamState.fusionEnabled) {
      setImageSource(fusionFeed, API.fusion);
    }
  } catch (error) {
    addLog(`YOLOv8 OBB 失败: ${error.message}`);
    alert(`YOLOv8 OBB 失败: ${error.message}`);
    await refreshStatus();
  } finally {
    streamState.yoloObbRunning = false;
  }
}

async function generateDiagnosticReport() {
  if (streamState.reportRunning) {
    addLog("诊断报告正在生成中");
    return;
  }
  const llmMode = reportLlmMode?.value || "auto";
  const requestedModeLabel = REPORT_MODE_LABELS[llmMode] || llmMode;
  streamState.reportRunning = true;
  streamState.reportCompletionHandled = false;
  generateReportButton.disabled = true;
  generateReportButton.textContent = "生成中...";
  streamState.reportTask = {
    running: true,
    progress: 1,
    stage: "queued",
    message: `已提交报告任务（${requestedModeLabel}）`,
    providerLabel: requestedModeLabel,
    requestedMode: llmMode,
    requestedModeLabel,
  };
  renderReport();
  renderDiagnosticReportCard();
  startReportProgressPolling();
  try {
    const response = await fetch(`${API.generateReport}?llmMode=${encodeURIComponent(llmMode)}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const payload = await response.json();
    if (!payload.ok) {
      throw new Error(payload.error || "报告生成失败");
    }
    streamState.reportTask = payload.task || streamState.reportTask;
    if (streamState.lastStatus) {
      streamState.lastStatus.reportTask = streamState.reportTask;
    }
    addLog(`诊断报告任务已启动，模式: ${requestedModeLabel}`);
    renderReport();
    renderDiagnosticReportCard();
  } catch (error) {
    addLog(`诊断报告生成失败: ${error.message}`);
    alert(`诊断报告生成失败: ${error.message}`);
    streamState.reportRunning = false;
    generateReportButton.disabled = false;
    generateReportButton.textContent = "生成诊断 PDF";
    stopReportProgressPolling();
  }
}
function startReportProgressPolling() {
  stopReportProgressPolling();
  streamState.reportProgressTimer = setInterval(async () => {
    try {
      const response = await fetch(API.reportProgress, { cache: "no-store" });
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      if (payload.ok && payload.task) {
        streamState.reportTask = payload.task;
        if (streamState.lastStatus) {
          streamState.lastStatus.reportTask = payload.task;
        }
        renderReport();
        renderDiagnosticReportCard();
        if (!payload.task.running) {
          handleReportTaskFinished(payload.task);
          stopReportProgressPolling();
        }
      }
    } catch (_error) {
    }
  }, 600);
}

function handleReportTaskFinished(task) {
  if (streamState.reportCompletionHandled) {
    return;
  }
  streamState.reportCompletionHandled = true;
  streamState.reportRunning = false;
  generateReportButton.disabled = false;
  generateReportButton.textContent = "生成诊断 PDF";

  if (task.stage === "done") {
    const providerLabel = task.providerLabel || task.requestedModeLabel || "--";
    addLog(`诊断报告已生成: ${task.pdfPath || "未返回路径"} / ${providerLabel}`);
    if (task.pdfUrl) {
      window.open(task.pdfUrl, "_blank", "noopener,noreferrer");
    }
    alert(`诊断 PDF 已生成\n模型来源: ${providerLabel}`);
    return;
  }

  if (task.stage === "error") {
    addLog(`诊断报告生成失败: ${task.error || "未知错误"}`);
    alert(`诊断报告生成失败: ${task.error || "未知错误"}`);
  }
}
function stopReportProgressPolling() {
  if (streamState.reportProgressTimer) {
    clearInterval(streamState.reportProgressTimer);
    streamState.reportProgressTimer = null;
  }
}

function getImagePointFromOverlayEvent(event, overlayElement, imageElement) {
  const overlayRect = overlayElement.getBoundingClientRect();
  const imageRatio = imageElement.naturalWidth / Math.max(imageElement.naturalHeight, 1);
  const boxRatio = overlayRect.width / Math.max(overlayRect.height, 1);

  let imageLeft = overlayRect.left;
  let imageTop = overlayRect.top;
  let imageWidth = overlayRect.width;
  let imageHeight = overlayRect.height;

  if (imageRatio > boxRatio) {
    imageHeight = overlayRect.width / imageRatio;
    imageTop = overlayRect.top + (overlayRect.height - imageHeight) / 2;
  } else {
    imageWidth = overlayRect.height * imageRatio;
    imageLeft = overlayRect.left + (overlayRect.width - imageWidth) / 2;
  }

  const xNorm = (event.clientX - imageLeft) / imageWidth;
  const yNorm = (event.clientY - imageTop) / imageHeight;
  if (xNorm < 0 || xNorm > 1 || yNorm < 0 || yNorm > 1) {
    return null;
  }

  return {
    xNorm,
    yNorm,
    overlayX: (event.clientX - overlayRect.left) / overlayRect.width,
    overlayY: (event.clientY - overlayRect.top) / overlayRect.height,
  };
}

function startBoardCalibration(target = "visible") {
  streamState.boardCalibrating = true;
  streamState.boardCalTarget = target;
  streamState.boardCalPoints = [];
  if (target === "thermal") {
    streamState.pendingThermalBoardCal = null;
  } else {
    streamState.pendingVisibleBoardCal = null;
  }
  updateBoardCalUi();
  addLog(`${target === "thermal" ? "热成像" : "可见光"}四孔标定开始，请按左上、右上、右下、左下顺序点击`);
}

function captureBoardCalPoint(event, overlayElement, imageElement) {
  if (!streamState.boardCalibrating || streamState.boardCalPoints.length >= 4) {
    return;
  }

  const point = getImagePointFromOverlayEvent(event, overlayElement, imageElement);
  if (!point) {
    addLog("请点击画面区域内的孔位");
    return;
  }

  streamState.boardCalPoints.push(point);
  updateBoardCalUi();
  addLog(`已记录 ${boardCalLabels[streamState.boardCalPoints.length - 1]}`);

  if (streamState.boardCalPoints.length === 4) {
    const payload = streamState.boardCalPoints.map(({ xNorm, yNorm }) => ({ xNorm, yNorm }));
    if (streamState.boardCalTarget === "thermal") {
      streamState.pendingThermalBoardCal = payload;
      addLog("热成像四孔已记录，点击“确定热成像”保存");
    } else {
      streamState.pendingVisibleBoardCal = payload;
      addLog("可见光四孔已记录，点击“确定可见光”保存");
    }
    updateBoardCalUi();
  }
}

async function confirmBoardCalibration(target) {
  const points = target === "thermal" ? streamState.pendingThermalBoardCal : streamState.pendingVisibleBoardCal;
  if (!points || points.length !== 4) {
    addLog(`${target === "thermal" ? "热成像" : "可见光"}没有待确认的四孔数据`);
    return;
  }

  try {
    const response = await fetch(API.manualBoardPose, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target, points }),
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }

    if (target === "thermal") {
      streamState.pendingThermalBoardCal = null;
      addLog("热成像四孔标定已保存");
      alert("热成像四孔标定完成");
    } else {
      streamState.pendingVisibleBoardCal = null;
      addLog("可见光四孔标定已保存");
      alert("可见光四孔标定完成");
    }

    clearCurrentBoardCalibrationStep();
    await refreshStatus();
  } catch (error) {
    addLog(`${target === "thermal" ? "热成像" : "可见光"}标定保存失败: ${error.message}`);
  }
}

async function waitForFusionCompletion(startServerTime = 0, timeoutMs = 10000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    await refreshStatus();
    const status = streamState.lastStatus;
    const fusion = status?.fusion;
    if (
      fusion?.boardPose === "manual-holes-pair" &&
      fusion?.mode === "visual-locked-overlay" &&
      fusion?.visibleBorder &&
      fusion?.thermalBorder &&
      Number(fusion?.samples ?? 0) >= 4 &&
      Number(status?.serverTime ?? 0) > Number(startServerTime ?? 0)
    ) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

async function runManualFusionCalibration() {
  if (streamState.pendingVisibleBoardCal || streamState.pendingThermalBoardCal) {
    addLog("请先完成并确认可见光、热成像四孔标定");
    return;
  }

  const fusion = streamState.lastStatus?.fusion;
  if (!fusion?.manualVisiblePose || !fusion?.manualThermalPose) {
    addLog("请先完成可见光和热成像四孔标定");
    return;
  }

  try {
    streamState.fusionAlertPending = true;
    const startServerTime = Number(streamState.lastStatus?.serverTime ?? 0);
    addLog("开始 OpenCV 修位与重新融合");
    const response = await fetch(API.recalibrate, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    addLog("已请求 OpenCV 修位");
    setPillState(modelStatus, "OpenCV 修位中", "warn");
    const ok = await waitForFusionCompletion(startServerTime);
    if (!ok) {
      streamState.fusionAlertPending = false;
      alert("融合超时，请重试");
      addLog("融合超时");
    }
  } catch (error) {
    streamState.fusionAlertPending = false;
    addLog(`融合失败: ${error.message}`);
  }
}
async function clearBoardCalibration() {
  try {
    const visibleResponse = await fetch(`${API.settings}?clearBoardPose=1`, { cache: "no-store" });
    if (!visibleResponse.ok) {
      throw new Error(`status ${visibleResponse.status}`);
    }
    const thermalResponse = await fetch(`${API.settings}?clearThermalBoardPose=1`, { cache: "no-store" });
    if (!thermalResponse.ok) {
      throw new Error(`status ${thermalResponse.status}`);
    }

    streamState.pendingVisibleBoardCal = null;
    streamState.pendingThermalBoardCal = null;
    clearCurrentBoardCalibrationStep();
    addLog("已清除可见光和热成像四孔标定");
    await refreshStatus();
  } catch (error) {
    addLog(`清除四孔标定失败: ${error.message}`);
  }
}

function startCamera() {
  if (streamState.cameraConnected) {
    addLog("可见光视频已连接");
    return;
  }
  setImageSource(cameraFeed, API.visible);
  updateFusionVisibility();
  streamState.cameraConnected = true;
  togglePlaceholder(cameraPlaceholder, true);
  setPillState(cameraStatus, "可见光连接中");
  updateRunState();
  addLog("正在打开可见光视频");
}

function stopCamera(options = {}) {
  const { silent = false } = options;
  clearImageSource(cameraFeed);
  streamState.cameraConnected = false;
  togglePlaceholder(cameraPlaceholder, false);
  setPillState(cameraStatus, "可见光待连接");
  visibleResolution.textContent = "--";
  updateRunState();
  if (!silent) {
    addLog("可见光视频已断开");
  }
}

function startThermal() {
  if (streamState.thermalConnected) {
    addLog("热成像视频已连接");
    return;
  }
  setImageSource(thermalFeed, API.thermal);
  updateFusionVisibility();
  streamState.thermalConnected = true;
  togglePlaceholder(thermalPlaceholder, true);
  setPillState(thermalStatus, "热成像连接中");
  updateRunState();
  addLog("正在打开热成像视频");
}

function stopThermal(options = {}) {
  const { silent = false } = options;
  clearImageSource(thermalFeed);
  streamState.thermalConnected = false;
  togglePlaceholder(thermalPlaceholder, false);
  setPillState(thermalStatus, "热成像待连接");
  thermalReadout.textContent = "--";
  updateRunState();
  if (!silent) {
    addLog("热成像视频已断开");
  }
}

function captureFrame() {
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  link.download = `fusion-frame-${stamp}.jpg`;
  link.href = `${API.snapshot}?ts=${Date.now()}`;
  link.click();
  addLog("已保存当前融合截图");
}

async function updateFusionAlpha() {
  const alpha = Number(fusionAlpha.value) / 100;
  alphaValue.textContent = `${fusionAlpha.value}%`;
  try {
    const response = await fetch(`${API.settings}?alpha=${alpha.toFixed(2)}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    addLog(`融合透明度已设置为 ${fusionAlpha.value}%`);
  } catch (error) {
    addLog(`透明度更新失败: ${error.message}`);
  }
}

async function sendManualAdjust({ reset = false } = {}) {
  const params = new URLSearchParams();
  if (reset) {
    params.set("resetManual", "1");
  } else {
    params.set("dx", streamState.manualAdjust.dx.toFixed(2));
    params.set("dy", streamState.manualAdjust.dy.toFixed(2));
    params.set("scaleX", streamState.manualAdjust.scaleX.toFixed(4));
    params.set("scaleY", streamState.manualAdjust.scaleY.toFixed(4));
    params.set("sourceLeft", streamState.manualAdjust.sourceLeft.toFixed(2));
    params.set("sourceRight", streamState.manualAdjust.sourceRight.toFixed(2));
    params.set("sourceTop", streamState.manualAdjust.sourceTop.toFixed(2));
    params.set("sourceBottom", streamState.manualAdjust.sourceBottom.toFixed(2));
  }

  const response = await fetch(`${API.settings}?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`status ${response.status}`);
  }
  const payload = await response.json();
  if (payload.settings?.manualAdjust) {
    streamState.manualAdjust = {
      dx: Number(payload.settings.manualAdjust.dx ?? 0),
      dy: Number(payload.settings.manualAdjust.dy ?? 0),
      scaleX: Number(payload.settings.manualAdjust.scaleX ?? 1),
      scaleY: Number(payload.settings.manualAdjust.scaleY ?? 1),
      sourceLeft: Number(payload.settings.manualAdjust.sourceLeft ?? 0),
      sourceRight: Number(payload.settings.manualAdjust.sourceRight ?? 0),
      sourceTop: Number(payload.settings.manualAdjust.sourceTop ?? 0),
      sourceBottom: Number(payload.settings.manualAdjust.sourceBottom ?? 0),
    };
  }
  updateManualAdjustUi();
  if (streamState.fusionEnabled) {
    setImageSource(fusionFeed, API.fusion);
  }
}

async function nudgeManualAdjust(key, step) {
  streamState.manualAdjust[key] += step;
  if (key === "scaleX" || key === "scaleY") {
    streamState.manualAdjust[key] = Math.min(2.5, Math.max(0.35, streamState.manualAdjust[key]));
  }
  if (key.startsWith("source")) {
    streamState.manualAdjust[key] = Math.min(120, Math.max(-120, streamState.manualAdjust[key]));
  }

  try {
    await sendManualAdjust();
    addLog(`融合微调: ${formatManualAdjust()}`);
  } catch (error) {
    addLog(`融合微调失败: ${error.message}`);
  }
}

function parseDelimitedCoordinateText(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) {
    return [];
  }

  const parts0 = lines[0].split(/[,;\t]/).map((part) => part.trim());
  const hasHeader = parts0.some((part) => {
    const normalized = normalizeColumnName(part);
    return ["ref", "refdes", "designator", "x", "midx", "centrex", "y", "midy", "centrey", "layer"].includes(normalized);
  });

  if (hasHeader) {
    const headers = parts0.map((part) => normalizeColumnName(part));
    return lines
      .slice(1)
      .map((line) => {
        const values = line.split(/[,;\t]/).map((part) => part.trim());
        const rowMap = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
        const ref = firstPresent(rowMap, ["ref", "refdes", "designator", "reference", "位号"], "");
        const x = firstPresent(rowMap, ["x", "mid x", "center x", "centrex", "centerx", "pos x", "x(mm)", "ref x", "pad x", "坐标x"], "");
        const y = firstPresent(rowMap, ["y", "mid y", "center y", "centrey", "centery", "pos y", "y(mm)", "ref y", "pad y", "坐标y"], "");
        return {
          ref: String(ref).trim(),
          x: parseMetricNumber(x),
          y: parseMetricNumber(y),
          footprint: String(firstPresent(rowMap, ["footprint", "package", "封装"], "")).trim(),
          device: String(firstPresent(rowMap, ["device", "model", "型号"], "")).trim(),
          comment: String(firstPresent(rowMap, ["comment", "value", "参数", "描述"], "")).trim(),
          name: String(firstPresent(rowMap, ["name"], "")).trim(),
          layer: String(firstPresent(rowMap, ["layer", "side"], "T")).trim().toUpperCase(),
          rotation: parseMetricNumber(firstPresent(rowMap, ["rotation", "rot", "angle", "角度"], 0)) || 0,
          refX: parseMetricNumber(firstPresent(rowMap, ["ref x", "refx"], "")),
          refY: parseMetricNumber(firstPresent(rowMap, ["ref y", "refy"], "")),
          padX: parseMetricNumber(firstPresent(rowMap, ["pad x", "padx"], "")),
          padY: parseMetricNumber(firstPresent(rowMap, ["pad y", "pady"], "")),
          pins: parseMetricNumber(firstPresent(rowMap, ["pins", "pin"], 0)) || 0,
          smd: String(firstPresent(rowMap, ["smd"], "")).trim(),
        };
      })
      .filter((item) => item.ref && Number.isFinite(item.x) && Number.isFinite(item.y));
  }

  return lines
    .map((line) => {
      const parts = line.split(/[,;\t]/).map((part) => part.trim());
      return {
        ref: parts[0] || "",
        x: parseMetricNumber(parts[1]),
        y: parseMetricNumber(parts[2]),
        footprint: parts[3] || "",
        device: parts[4] || "",
        comment: parts[5] || "",
        name: parts[6] || "",
      };
    })
    .filter((item) => item.ref && Number.isFinite(item.x) && Number.isFinite(item.y));
}

function normalizeColumnName(name) {
  return String(name ?? "")
    .trim()
    .toLowerCase()
    .replace(/[\s_\-./()[\]:]+/g, "");
}

function firstPresent(rowMap, keys, fallback = undefined) {
  for (const key of keys) {
    const normalized = normalizeColumnName(key);
    if (Object.prototype.hasOwnProperty.call(rowMap, normalized)) {
      const value = rowMap[normalized];
      if (value !== null && value !== undefined && String(value).trim() !== "") {
        return value;
      }
    }
  }
  return fallback;
}

function parseMetricNumber(value) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : NaN;
  }
  const text = String(value ?? "").trim();
  if (!text) {
    return NaN;
  }
  const match = text.replace(/,/g, "").match(/[-+]?\d*\.?\d+(?:e[-+]?\d+)?/i);
  return match ? Number(match[0]) : NaN;
}

function parseSpreadsheetCoordinateFile(file, buffer) {
  const workbook = XLSX.read(buffer, { type: "array" });
  const components = [];

  workbook.SheetNames.forEach((sheetName) => {
    const sheet = workbook.Sheets[sheetName];
    const rows = XLSX.utils.sheet_to_json(sheet, { defval: "" });
    if (!rows.length) {
      return;
    }

    rows.forEach((row) => {
      const entries = Object.entries(row);
      const lowerMap = Object.fromEntries(entries.map(([key, value]) => [normalizeColumnName(key), value]));
      const ref =
        firstPresent(lowerMap, ["ref", "refdes", "designator", "reference", "reference designator", "位号"]) ??
        lowerMap.ref ??
        lowerMap.designator ??
        lowerMap["位号"] ??
        lowerMap["refdes"] ??
        entries[0]?.[1] ??
        "";
      const x =
        firstPresent(lowerMap, ["x", "mid x", "center x", "centrex", "centerx", "pos x", "posx", "x(mm)", "x mm", "ref x", "坐标x"]) ??
        lowerMap.x ??
        lowerMap.centerx ??
        lowerMap.posx ??
        lowerMap["x(mm)"] ??
        lowerMap["x"] ??
        entries[1]?.[1] ??
        "";
      const y =
        firstPresent(lowerMap, ["y", "mid y", "center y", "centrey", "centery", "pos y", "posy", "y(mm)", "y mm", "ref y", "坐标y"]) ??
        lowerMap.y ??
        lowerMap.centery ??
        lowerMap.posy ??
        lowerMap["y(mm)"] ??
        lowerMap["y"] ??
        entries[2]?.[1] ??
        "";
      const component = {
        ref: String(ref).trim(),
        x: parseMetricNumber(x),
        y: parseMetricNumber(y),
        footprint: String(lowerMap.footprint ?? lowerMap.package ?? lowerMap["封装"] ?? "").trim(),
        device: String(lowerMap.device ?? lowerMap.model ?? lowerMap["型号"] ?? "").trim(),
        comment: String(lowerMap.comment ?? lowerMap.value ?? lowerMap["参数"] ?? lowerMap["描述"] ?? "").trim(),
        name: String(firstPresent(lowerMap, ["name", "value", "comment", "参数"], "")).trim(),
        layer: String(lowerMap.layer ?? lowerMap.side ?? "T").trim().toUpperCase(),
        rotation: parseMetricNumber(firstPresent(lowerMap, ["rotation", "rot", "angle", "角度"], lowerMap.rotation ?? lowerMap.rot ?? 0)) || 0,
        refX: parseMetricNumber(firstPresent(lowerMap, ["ref x", "refx"], "")),
        refY: parseMetricNumber(firstPresent(lowerMap, ["ref y", "refy"], "")),
        padX: parseMetricNumber(firstPresent(lowerMap, ["pad x", "padx"], "")),
        padY: parseMetricNumber(firstPresent(lowerMap, ["pad y", "pady"], "")),
        pins: parseMetricNumber(firstPresent(lowerMap, ["pins", "pin"], lowerMap.pins ?? 0)) || 0,
        smd: String(firstPresent(lowerMap, ["smd"], "")).trim(),
      };
      if (component.ref && Number.isFinite(component.x) && Number.isFinite(component.y)) {
        components.push(component);
      }
    });
  });

  return components;
}

async function parseCoordinateFile(file) {
  const lowerName = file.name.toLowerCase();
  if (lowerName.endsWith(".json")) {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    if (Array.isArray(parsed.components)) {
      return parsed.components;
    }
    return [];
  }

  if (lowerName.endsWith(".xlsx") || lowerName.endsWith(".xls")) {
    const buffer = await file.arrayBuffer();
    return parseSpreadsheetCoordinateFile(file, buffer);
  }

  const text = await file.text();
  return parseDelimitedCoordinateText(text);
}

async function handleNetlistFileChange(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    return;
  }

  try {
    boardModel.fileName = file.name;
    boardModel.components = await parseCoordinateFile(file);
    if (!boardModel.components.length) {
      throw new Error("没有解析到器件坐标，请检查列名是否包含 Designator / RefDes / Mid X / Mid Y");
    }
    const response = await fetch(API.boardModel, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fileName: file.name,
        components: boardModel.components,
      }),
    });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    updateBoardUi();
    renderReport();
    await refreshStatus();
    addLog(`已导入坐标文件: ${file.name} (${boardModel.components.length} 条)`);
  } catch (error) {
    addLog(`坐标文件解析或上传失败: ${error.message}`);
  } finally {
    event.target.value = "";
  }
}

document.querySelector("#startCamera").addEventListener("click", startCamera);
document.querySelector("#stopCamera").addEventListener("click", () => stopCamera());
document.querySelector("#startThermal").addEventListener("click", startThermal);
document.querySelector("#stopThermal").addEventListener("click", () => stopThermal());
document.querySelector("#toggleFusion").addEventListener("click", () => {
  streamState.fusionEnabled = !streamState.fusionEnabled;
  updateFusionVisibility();
  updateRunState();
  addLog(streamState.fusionEnabled ? "融合显示已开启" : "融合显示已关闭");
});
document.querySelector("#captureFrame").addEventListener("click", captureFrame);
document.querySelector("#recalibrateFusion").addEventListener("click", runManualFusionCalibration);
document.querySelector("#runYoloObb").addEventListener("click", runYoloObbDetection);
generateReportButton.addEventListener("click", () => {
  generateDiagnosticReport();
  addLog(`开始生成诊断 PDF，模式: ${REPORT_MODE_LABELS[reportLlmMode?.value || "auto"] || (reportLlmMode?.value || "auto")}`);
});
document.querySelector("#uploadNetlist").addEventListener("click", () => {
  netlistInput.click();
});
document.querySelector("#clearNetlist").addEventListener("click", () => {
  fetch(`${API.settings}?clearRuntimeBoardModel=1`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`status ${response.status}`);
      }
      return response.json();
    })
    .then(async () => {
      boardModel.fileName = "";
      boardModel.components = [];
      updateBoardUi();
      await refreshStatus();
      addLog("坐标文件已清除");
    })
    .catch((error) => {
      addLog(`清除坐标文件失败: ${error.message}`);
    });
});
document.querySelector("#startBoardCal").addEventListener("click", () => startBoardCalibration("visible"));
document.querySelector("#startThermalBoardCal").addEventListener("click", () => startBoardCalibration("thermal"));
document.querySelector("#undoBoardCal").addEventListener("click", () => {
  if (!streamState.boardCalibrating || !streamState.boardCalPoints.length) {
    return;
  }
  streamState.boardCalPoints.pop();
  if (streamState.boardCalTarget === "thermal") {
    streamState.pendingThermalBoardCal = null;
  } else {
    streamState.pendingVisibleBoardCal = null;
  }
  updateBoardCalUi();
  addLog("已撤销上一个点");
});
document.querySelector("#confirmVisibleBoardCal").addEventListener("click", () => confirmBoardCalibration("visible"));
document.querySelector("#confirmThermalBoardCal").addEventListener("click", () => confirmBoardCalibration("thermal"));
document.querySelector("#runFusionCal").addEventListener("click", runManualFusionCalibration);
document.querySelector("#clearBoardCal").addEventListener("click", clearBoardCalibration);

boardCalOverlay.addEventListener("click", (event) => captureBoardCalPoint(event, boardCalOverlay, cameraFeed));
thermalCalOverlay.addEventListener("click", (event) => captureBoardCalPoint(event, thermalCalOverlay, thermalFeed));

document.querySelectorAll("[data-adjust]").forEach((button) => {
  button.addEventListener("click", () => {
    nudgeManualAdjust(button.dataset.adjust, Number(button.dataset.step));
  });
});

document.querySelector("#resetManualAdjust").addEventListener("click", async () => {
  try {
    await sendManualAdjust({ reset: true });
    addLog("融合微调已重置");
  } catch (error) {
    addLog(`重置融合微调失败: ${error.message}`);
  }
});

document.querySelector("#saveManualAdjust").addEventListener("click", () => {
  fetch(`${API.settings}?saveManual=1`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`status ${response.status}`);
      }
      return response.json();
    })
    .then(() => {
      addLog(`已保存融合微调参数: ${JSON.stringify(streamState.manualAdjust)}`);
    })
    .catch((error) => {
      addLog(`保存融合微调失败: ${error.message}`);
    });
});

netlistInput.addEventListener("change", handleNetlistFileChange);

tempThreshold.addEventListener("input", () => {
  thresholdValue.textContent = `${tempThreshold.value} C`;
});
tempThreshold.addEventListener("change", () => {
  addLog(`热区阈值已设置为 ${tempThreshold.value} C`);
  renderReport();
});

fusionAlpha.addEventListener("input", () => {
  alphaValue.textContent = `${fusionAlpha.value}%`;
});
fusionAlpha.addEventListener("change", updateFusionAlpha);

detectMode.addEventListener("change", () => {
  addLog(`检测模式已切换: ${detectMode.value}`);
  renderReport();
});

thresholdValue.textContent = `${tempThreshold.value} C`;
alphaValue.textContent = `${fusionAlpha.value}%`;
updateManualAdjustUi();
updateBoardUi();
updateBoardCalUi();
renderReport();
addLog("面板已加载");
startCamera();
startThermal();
refreshStatus();
streamState.statusTimer = setInterval(refreshStatus, 1200);

window.addEventListener("beforeunload", () => {
  stopCamera({ silent: true });
  stopThermal({ silent: true });
  if (streamState.statusTimer) {
    clearInterval(streamState.statusTimer);
  }
  stopReportProgressPolling();
});

clearLogButton?.addEventListener("click", () => {
  logList.innerHTML = "";
  addLog("日志已清空");
});


