// 获取DOM元素
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
const gridCanvas = document.getElementById("gridCanvas");
const gridCtx = gridCanvas.getContext("2d");
const processedImageCanvas = document.getElementById("processedImage");
const processedImageCtx = processedImageCanvas.getContext("2d");
const recognizeBtn = document.getElementById("recognizeBtn");
const clearBtn = document.getElementById("clearBtn");
const resultDiv = document.getElementById("resultValue");
const confidenceBar = document.querySelector(".confidence-level");
const confidenceText = document.querySelector(".confidence-text");
const brushSizeInput = document.getElementById("brushSize");
const brushSizeValue = document.getElementById("brushSizeValue");
const showGridCheckbox = document.getElementById("showGrid");
const modelVersionElement = document.getElementById("modelVersion");
const allPredictionsDiv = document.getElementById("allPredictions");

// 应用配置
const API_BASE_URL =
  window.location.hostname === "localhost" ? "http://localhost:5002" : "";
let brushSize = 15;
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// 初始化
function init() {
  console.log("初始化应用");

  // 设置画布背景为白色
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 设置默认画笔 - 使用更深的颜色和更大的线宽
  ctx.strokeStyle = "#000000"; // 纯黑色
  ctx.fillStyle = "#000000"; // 填充也使用黑色
  ctx.lineWidth = brushSize;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // 绘制网格线
  drawGrid();

  // 添加事件监听器
  addEventListeners();

  // 获取模型信息
  fetchModelInfo();

  // 更新画笔大小显示
  updateBrushSizeDisplay();

  // 初始化预测概率条
  initPredictionBars();

  console.log("应用初始化完成");
}

// 添加所有事件监听器
function addEventListeners() {
  console.log("添加事件监听器");

  // 鼠标事件
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseout", stopDrawing);

  // 触摸事件
  canvas.addEventListener("touchstart", handleTouchStart, false);
  canvas.addEventListener("touchmove", handleTouchMove, false);
  canvas.addEventListener("touchend", stopDrawing, false);

  // 按钮事件
  clearBtn.addEventListener("click", clearCanvas);
  recognizeBtn.addEventListener("click", recognizeDigit);

  // 其他控件事件
  brushSizeInput.addEventListener("input", updateBrushSize);
  showGridCheckbox.addEventListener("change", drawGrid);

  console.log("事件监听器添加完成");
}

// 绘制网格线
function drawGrid() {
  if (!showGridCheckbox.checked) {
    gridCtx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);
    return;
  }

  const gridSize = 28; // 28x28网格，与MNIST数据集一致
  const cellSize = canvas.width / gridSize;

  gridCtx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);
  gridCtx.beginPath();
  gridCtx.strokeStyle = "rgba(0, 0, 0, 0.1)";
  gridCtx.lineWidth = 1;

  // 绘制垂直线
  for (let i = 1; i < gridSize; i++) {
    const x = i * cellSize;
    gridCtx.moveTo(x, 0);
    gridCtx.lineTo(x, canvas.height);
  }

  // 绘制水平线
  for (let i = 1; i < gridSize; i++) {
    const y = i * cellSize;
    gridCtx.moveTo(0, y);
    gridCtx.lineTo(canvas.width, y);
  }

  gridCtx.stroke();
}

// 初始化预测概率条
function initPredictionBars() {
  allPredictionsDiv.innerHTML = "";

  for (let i = 0; i < 10; i++) {
    const row = document.createElement("div");
    row.className = "prediction-row";
    row.id = `prediction-row-${i}`;

    const label = document.createElement("div");
    label.className = "digit-label";
    label.textContent = i;

    const barContainer = document.createElement("div");
    barContainer.className = "prediction-bar-container";

    const bar = document.createElement("div");
    bar.className = "prediction-bar";
    bar.id = `prediction-bar-${i}`;
    bar.style.width = "0%";

    const value = document.createElement("div");
    value.className = "prediction-value";
    value.id = `prediction-value-${i}`;
    value.textContent = "0%";

    barContainer.appendChild(bar);
    barContainer.appendChild(value);

    row.appendChild(label);
    row.appendChild(barContainer);

    allPredictionsDiv.appendChild(row);
  }
}

// 更新所有预测概率
function updatePredictions(predictions) {
  // 移除所有活跃类
  document.querySelectorAll(".prediction-row").forEach((row) => {
    row.classList.remove("active");
  });

  // 更新每个数字的预测概率
  for (let i = 0; i < 10; i++) {
    const probability = predictions[i] * 100;
    const bar = document.getElementById(`prediction-bar-${i}`);
    const value = document.getElementById(`prediction-value-${i}`);
    const row = document.getElementById(`prediction-row-${i}`);

    bar.style.width = `${probability}%`;
    value.textContent = `${probability.toFixed(2)}%`;

    // 高亮最高概率的数字
    if (predictions[i] === Math.max(...predictions)) {
      row.classList.add("active");
    }
  }
}

// 获取模型信息
function fetchModelInfo() {
  fetch(`${API_BASE_URL}/model_info`)
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        modelVersionElement.textContent = "未知";
      } else {
        modelVersionElement.textContent = `${data.model_version} (${data.model_type})`;
      }
    })
    .catch((error) => {
      console.error("获取模型信息失败:", error);
      modelVersionElement.textContent = "未知";
    });
}

// 处理触摸开始
function handleTouchStart(e) {
  e.preventDefault();
  console.log("触摸开始");

  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;

  isDrawing = true;
  [lastX, lastY] = [x, y];

  // 确保画笔颜色设置正确
  ctx.fillStyle = "#000000";
  ctx.strokeStyle = "#000000";

  // 立即绘制一个点，以便触摸开始时也能绘制
  ctx.beginPath();
  ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
  ctx.fill();

  // 添加调试信息
  console.log("触摸点绘制完成", x, y);
}

// 处理触摸移动
function handleTouchMove(e) {
  e.preventDefault();
  if (!isDrawing) return;

  console.log("触摸移动");

  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;

  // 确保画笔颜色设置正确
  ctx.fillStyle = "#000000";
  ctx.strokeStyle = "#000000";

  // 绘制线条
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();

  // 在线条的终点绘制一个圆点，增强可见性
  ctx.beginPath();
  ctx.arc(x, y, brushSize / 4, 0, Math.PI * 2);
  ctx.fill();

  [lastX, lastY] = [x, y];

  // 添加调试信息
  console.log("触摸绘制完成", x, y);
}

// 更新画笔大小
function updateBrushSize() {
  brushSize = brushSizeInput.value;
  ctx.lineWidth = brushSize;
  updateBrushSizeDisplay();
}

// 更新画笔大小显示
function updateBrushSizeDisplay() {
  brushSizeValue.textContent = brushSize;
}

// 开始绘画
function startDrawing(e) {
  console.log("开始绘画", e.offsetX, e.offsetY);
  isDrawing = true;
  [lastX, lastY] = [e.offsetX, e.offsetY];

  // 确保画笔颜色设置正确
  ctx.fillStyle = "#000000";
  ctx.strokeStyle = "#000000";

  // 立即绘制一个点，以便单击也能绘制
  ctx.beginPath();
  ctx.arc(lastX, lastY, brushSize / 2, 0, Math.PI * 2);
  ctx.fill();

  // 添加调试信息
  console.log("绘制点完成", ctx.fillStyle);
}

// 绘画过程
function draw(e) {
  if (!isDrawing) return;

  console.log("绘画中", e.offsetX, e.offsetY);

  // 确保画笔颜色设置正确
  ctx.fillStyle = "#000000";
  ctx.strokeStyle = "#000000";

  // 绘制线条
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();

  // 在线条的终点绘制一个圆点，增强可见性
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, brushSize / 4, 0, Math.PI * 2);
  ctx.fill();

  [lastX, lastY] = [e.offsetX, e.offsetY];

  // 添加调试信息
  console.log("绘制线条完成", ctx.strokeStyle);
}

// 停止绘画
function stopDrawing() {
  console.log("停止绘画");
  isDrawing = false;
}

// 清除画布
function clearCanvas() {
  console.log("清除画布");

  // 用白色填充画布
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 重置画笔颜色为黑色
  ctx.fillStyle = "#000000";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = brushSize;

  // 重置结果显示
  resultDiv.textContent = "等待绘制...";
  confidenceBar.style.width = "0%";
  confidenceText.textContent = "置信度: 0%";

  // 清除处理后的图像
  processedImageCtx.fillStyle = "black";
  processedImageCtx.fillRect(
    0,
    0,
    processedImageCanvas.width,
    processedImageCanvas.height
  );

  // 重置所有预测概率
  for (let i = 0; i < 10; i++) {
    const bar = document.getElementById(`prediction-bar-${i}`);
    const value = document.getElementById(`prediction-value-${i}`);
    const row = document.getElementById(`prediction-row-${i}`);

    bar.style.width = "0%";
    value.textContent = "0%";
    row.classList.remove("active");
  }

  console.log("画布已清除，画笔颜色重置为", ctx.fillStyle);
}

// 显示处理后的图像
function displayProcessedImage(imageData) {
  // 解码Base64图像数据
  const img = new Image();
  img.onload = function () {
    processedImageCtx.drawImage(
      img,
      0,
      0,
      processedImageCanvas.width,
      processedImageCanvas.height
    );
  };
  img.src = imageData;
}

// 识别数字
function recognizeDigit() {
  console.log("开始识别数字");

  // 显示加载状态
  resultDiv.textContent = "识别中...";

  const imageData = canvas.toDataURL("image/png");

  fetch(`${API_BASE_URL}/recognize_with_debug`, {
    method: "POST",
    body: JSON.stringify({ image: imageData }),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      console.log("识别结果:", data);

      if (data.error) {
        resultDiv.textContent = `错误: ${data.error}`;
        confidenceBar.style.width = "0%";
        confidenceText.textContent = "置信度: 0%";
      } else {
        const confidencePercent = (data.confidence * 100).toFixed(2);
        resultDiv.textContent = data.digit;
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceText.textContent = `置信度: ${confidencePercent}%`;

        // 根据置信度设置颜色
        if (data.confidence > 0.9) {
          confidenceBar.style.backgroundColor = "#34a853"; // 绿色
        } else if (data.confidence > 0.7) {
          confidenceBar.style.backgroundColor = "#fbbc05"; // 黄色
        } else {
          confidenceBar.style.backgroundColor = "#ea4335"; // 红色
        }

        // 更新所有预测概率
        if (data.all_predictions) {
          updatePredictions(data.all_predictions);
        }

        // 显示处理后的图像
        if (data.processed_image) {
          displayProcessedImage(data.processed_image);
        }
      }
    })
    .catch((error) => {
      console.error("识别请求失败:", error);
      resultDiv.textContent = "请求失败";
    });
}

// 健康检查
function healthCheck() {
  fetch(`${API_BASE_URL}/health`)
    .then((response) => response.json())
    .then((data) => {
      console.log("服务器状态:", data.status);
    })
    .catch((error) => {
      console.error("健康检查失败:", error);
    });
}

// 在页面加载完成后初始化应用
document.addEventListener("DOMContentLoaded", function () {
  console.log("页面加载完成");
  init();
  healthCheck();
});
