// 获取DOM元素
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
const recognizeBtn = document.getElementById("recognizeBtn");
const clearBtn = document.getElementById("clearBtn");
const resultDiv = document.getElementById("resultValue");
const confidenceBar = document.querySelector(".confidence-level");
const confidenceText = document.querySelector(".confidence-text");
const brushSizeInput = document.getElementById("brushSize");
const brushSizeValue = document.getElementById("brushSizeValue");
const modelVersionElement = document.getElementById("modelVersion");

// 应用配置
const API_BASE_URL = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
let brushSize = 15;

// 初始化
function init() {
  // 设置画布背景为白色
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // 设置默认画笔
  ctx.strokeStyle = "black";
  ctx.lineWidth = brushSize;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  
  // 添加触摸支持
  addTouchSupport();
  
  // 获取模型信息
  fetchModelInfo();
  
  // 更新画笔大小显示
  updateBrushSizeDisplay();
}

// 获取模型信息
function fetchModelInfo() {
  fetch(`${API_BASE_URL}/model_info`)
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        modelVersionElement.textContent = "未知";
      } else {
        modelVersionElement.textContent = `${data.model_version} (${data.model_type})`;
      }
    })
    .catch(error => {
      console.error("获取模型信息失败:", error);
      modelVersionElement.textContent = "未知";
    });
}

// 添加触摸支持
function addTouchSupport() {
  canvas.addEventListener("touchstart", handleTouchStart, false);
  canvas.addEventListener("touchmove", handleTouchMove, false);
  canvas.addEventListener("touchend", stopDrawing, false);
}

// 处理触摸开始
function handleTouchStart(e) {
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  
  isDrawing = true;
  [lastX, lastY] = [x, y];
}

// 处理触摸移动
function handleTouchMove(e) {
  e.preventDefault();
  if (!isDrawing) return;
  
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  
  [lastX, lastY] = [x, y];
}

// 绘画状态变量
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// 事件监听器
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);
clearBtn.addEventListener("click", clearCanvas);
recognizeBtn.addEventListener("click", recognizeDigit);
brushSizeInput.addEventListener("input", updateBrushSize);

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
  isDrawing = true;
  [lastX, lastY] = [e.offsetX, e.offsetY];
}

// 绘画过程
function draw(e) {
  if (!isDrawing) return;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  [lastX, lastY] = [e.offsetX, e.offsetY];
}

// 停止绘画
function stopDrawing() {
  isDrawing = false;
}

// 清除画布
function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  resultDiv.textContent = "等待绘制...";
  confidenceBar.style.width = "0%";
  confidenceText.textContent = "置信度: 0%";
}

// 识别数字
function recognizeDigit() {
  // 显示加载状态
  resultDiv.textContent = "识别中...";
  
  const imageData = canvas.toDataURL("image/png");

  fetch(`${API_BASE_URL}/recognize`, {
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
      }
    })
    .catch((error) => {
      resultDiv.textContent = "请求失败";
      console.error("识别请求失败:", error);
    });
}

// 健康检查
function healthCheck() {
  fetch(`${API_BASE_URL}/health`)
    .then(response => response.json())
    .then(data => {
      console.log("服务器状态:", data.status);
    })
    .catch(error => {
      console.error("健康检查失败:", error);
    });
}

// 初始化应用
init();
// 执行健康检查
healthCheck();
