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

// 拍照和上传相关元素
const drawTab = document.getElementById("drawTab");
const photoTab = document.getElementById("photoTab");
const drawInputPanel = document.getElementById("drawInputPanel");
const photoInputPanel = document.getElementById("photoInputPanel");
const photoInput = document.getElementById("photoInput");
const photoPreview = document.getElementById("photoPreview");
const photoPlaceholder = document.getElementById("photoPlaceholder");
const captureBtn = document.getElementById("captureBtn");
const recognizePhotoBtn = document.getElementById("recognizePhotoBtn");
const clearPhotoBtn = document.getElementById("clearPhotoBtn");
const cameraFeed = document.getElementById("cameraFeed");
const captureCanvas = document.getElementById("captureCanvas");
const captureCtx = captureCanvas.getContext("2d");

// 多数字识别相关元素
const multipleDigitsContainer = document.getElementById(
  "multipleDigitsContainer"
);
const combinedResultSpan = document.getElementById("combinedResult");
const digitVisualization = document.getElementById("digitVisualization");
const digitsList = document.getElementById("digitsList");
const recognitionModeRadios = document.getElementsByName("recognitionMode");

// 应用配置
const API_BASE_URL =
  window.location.hostname === "localhost" ? "http://localhost:5002" : "";
let brushSize = 8; // 减小默认画笔大小
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// 初始化
function init() {
  console.log("初始化应用");

  // 检查画布元素
  console.log("绘图画布:", canvas, "网格画布:", gridCanvas);

  // 设置画布背景为白色
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 设置默认画笔 - 使用更深的颜色和更大的线宽
  ctx.strokeStyle = "#000000"; // 纯黑色
  ctx.fillStyle = "#000000"; // 填充也使用黑色
  ctx.lineWidth = brushSize;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // 增强线条可见度
  ctx.shadowColor = "#000000";
  ctx.shadowBlur = 1;

  // 检查网格复选框状态
  console.log("网格复选框状态:", showGridCheckbox.checked);

  // 强制设置复选框为选中状态
  showGridCheckbox.checked = true;

  // 绘制网格线
  setTimeout(() => {
    console.log("延迟调用drawGrid");
    drawGrid();
  }, 100);

  // 添加事件监听器
  addEventListeners();

  // 获取模型信息
  fetchModelInfo();

  // 更新画笔大小显示
  updateBrushSizeDisplay();

  // 初始化预测概率条
  initPredictionBars();

  // 根据当前选择的模式设置显示区域
  const mode = getRecognitionMode();
  if (mode === "multiple") {
    multipleDigitsContainer.style.display = "block";
    document.querySelector(".result-container").style.display = "none";
    document.querySelector(".processed-image-container").style.display = "none";
  } else {
    multipleDigitsContainer.style.display = "none";
    document.querySelector(".result-container").style.display = "block";
    document.querySelector(".processed-image-container").style.display =
      "block";
  }

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

  // 标签切换事件
  drawTab.addEventListener("click", () => switchTab("draw"));
  photoTab.addEventListener("click", () => switchTab("photo"));

  // 拍照和上传相关事件
  photoInput.addEventListener("change", handlePhotoUpload);
  captureBtn.addEventListener("click", function () {
    if (cameraActive) {
      capturePhoto();
    } else {
      startCamera();
    }
  });
  recognizePhotoBtn.addEventListener("click", recognizePhoto);
  clearPhotoBtn.addEventListener("click", clearPhoto);

  console.log("事件监听器添加完成");
}

// 绘制网格线
function drawGrid() {
  console.log("开始绘制网格线");

  // 先清除网格画布
  gridCtx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);

  // 如果复选框未选中，则不绘制网格
  if (!showGridCheckbox.checked) {
    console.log("复选框未选中，不绘制网格");
    return;
  }

  // 恢复原始网格样式
  const gridSize = 28; // 28x28网格，与MNIST数据集一致
  const cellSize = canvas.width / gridSize;

  // 绘制网格线
  gridCtx.beginPath();
  gridCtx.strokeStyle = "rgba(0, 0, 0, 0.2)"; // 增强网格颜色的可见度
  gridCtx.lineWidth = 0.5;

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

  console.log("网格线绘制完成");
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
  ctx.shadowColor = "#000000";
  ctx.shadowBlur = 2; // 增强阴影效果

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
  ctx.shadowColor = "#000000";
  ctx.shadowBlur = 1;

  // 绘制线条
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.lineWidth = brushSize; // 使用标准线宽
  ctx.stroke();

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
  ctx.shadowColor = "#000000";
  ctx.shadowBlur = 2; // 增强阴影效果

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
  ctx.shadowColor = "#000000";
  ctx.shadowBlur = 1;

  // 绘制线条
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.lineWidth = brushSize; // 使用标准线宽
  ctx.stroke();

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

  // 重新绘制网格
  drawGrid();

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

// 切换输入模式标签
function switchTab(tabName) {
  console.log(`切换到${tabName}模式`);

  // 重置所有标签和面板
  drawTab.classList.remove("active");
  photoTab.classList.remove("active");
  drawInputPanel.classList.remove("active");
  photoInputPanel.classList.remove("active");

  // 激活选中的标签和面板
  if (tabName === "draw") {
    drawTab.classList.add("active");
    drawInputPanel.classList.add("active");
    // 如果有摄像头流正在运行，停止它
    stopCamera();
  } else if (tabName === "photo") {
    photoTab.classList.add("active");
    photoInputPanel.classList.add("active");
  }
}

// 处理图片上传
function handlePhotoUpload(e) {
  console.log("处理图片上传");

  const file = e.target.files[0];
  if (!file) return;

  // 检查是否为图片文件
  if (!file.type.match("image.*")) {
    alert("请选择图片文件");
    return;
  }

  // 读取文件并显示预览
  const reader = new FileReader();
  reader.onload = function (event) {
    photoPreview.src = event.target.result;
    photoPreview.style.display = "block";
    photoPlaceholder.style.display = "none";

    // 如果摄像头流正在运行，停止它
    stopCamera();
  };
  reader.readAsDataURL(file);
}

// 切换摄像头状态
let cameraActive = false;
let stream = null;

function toggleCamera() {
  if (cameraActive) {
    stopCamera();
  } else {
    startCamera();
  }
}

// 检测是否为移动设备
function isMobileDevice() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
}

// 启动摄像头
function startCamera() {
  console.log("启动摄像头");

  // 在移动设备上，直接触发文件选择器
  if (isMobileDevice()) {
    console.log("检测到移动设备，使用文件选择器");
    photoInput.click();
    return;
  }

  // 在PC端使用getUserMedia API
  console.log("在PC端使用摄像头API");
  // 设置摄像头画布大小
  captureCanvas.width = 280;
  captureCanvas.height = 280;

  // 请求摄像头权限
  navigator.mediaDevices
    .getUserMedia({ video: { facingMode: "environment" } })
    .then(function (mediaStream) {
      stream = mediaStream;
      cameraFeed.srcObject = mediaStream;
      cameraFeed.style.display = "block";
      photoPreview.style.display = "none";
      photoPlaceholder.style.display = "none";
      cameraFeed.play();
      cameraActive = true;
      captureBtn.textContent = "拍摄照片";
    })
    .catch(function (err) {
      console.error("无法访问摄像头: ", err);
      alert("无法访问摄像头，请检查摄像头权限或尝试上传图片");
    });
}

// 停止摄像头
function stopCamera() {
  console.log("停止摄像头");

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  cameraFeed.style.display = "none";
  cameraActive = false;
  captureBtn.textContent = "📷 拍照";

  // 如果没有预览图片，显示占位符
  if (photoPreview.src === "" || photoPreview.style.display === "none") {
    photoPlaceholder.style.display = "block";
  }
}

// 拍摄照片
function capturePhoto() {
  console.log("拍摄照片");

  if (!cameraActive) return;

  // 将视频帧绘制到画布上
  captureCtx.drawImage(
    cameraFeed,
    0,
    0,
    captureCanvas.width,
    captureCanvas.height
  );

  // 将画布转换为数据 URL
  const imageData = captureCanvas.toDataURL("image/png");

  // 显示拍摄的照片
  photoPreview.src = imageData;
  photoPreview.style.display = "block";
  cameraFeed.style.display = "none";
  photoPlaceholder.style.display = "none";

  // 停止摄像头
  stopCamera();
}

// 清除照片
function clearPhoto() {
  console.log("清除照片");

  photoPreview.src = "";
  photoPreview.style.display = "none";
  photoPlaceholder.style.display = "block";
  photoInput.value = "";

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

  // 清除多数字识别结果
  multipleDigitsContainer.style.display = "none";
  digitsList.innerHTML = "";
  combinedResultSpan.textContent = "";
  digitVisualization.src = "";
}

// 获取当前选中的识别模式
function getRecognitionMode() {
  for (const radio of recognitionModeRadios) {
    if (radio.checked) {
      return radio.value;
    }
  }
  return "single"; // 默认为单数字模式
}

// 识别照片中的数字
function recognizePhoto() {
  console.log("开始识别照片");

  // 检查是否有图片
  if (photoPreview.src === "" || photoPreview.style.display === "none") {
    alert("请先上传或拍摄一张图片");
    return;
  }

  // 显示加载状态
  resultDiv.textContent = "识别中...";

  // 获取当前识别模式
  const mode = getRecognitionMode();

  // 根据模式选择不同的API端点
  const endpoint =
    mode === "multiple" ? "/recognize_multiple_debug" : "/recognize_with_debug";

  // 重置显示区域
  if (mode === "multiple") {
    // 显示多数字结果区域，隐藏单数字结果相关元素
    multipleDigitsContainer.style.display = "block";
    document.querySelector(".result-container").style.display = "none";
    document.querySelector(".processed-image-container").style.display = "none";

    // 清除之前的结果
    digitsList.innerHTML = "";
    combinedResultSpan.textContent = "识别中...";
    digitVisualization.src = "";
  } else {
    // 隐藏多数字结果区域，显示单数字结果
    multipleDigitsContainer.style.display = "none";
    document.querySelector(".result-container").style.display = "block";
    document.querySelector(".processed-image-container").style.display =
      "block";
  }

  // 发送图片数据到服务器
  fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: JSON.stringify({ image: photoPreview.src }),
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
      console.log(`识别结果 (${mode} 模式):`, data);

      if (data.error) {
        resultDiv.textContent = `错误: ${data.error}`;
        confidenceBar.style.width = "0%";
        confidenceText.textContent = "置信度: 0%";
        if (mode === "multiple") {
          combinedResultSpan.textContent = `错误: ${data.error}`;
        }
      } else if (mode === "single") {
        // 处理单数字识别结果
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
      } else {
        // 处理多数字识别结果
        displayMultipleDigitsResults(data);
      }
    })
    .catch((error) => {
      console.error("识别请求失败:", error);
      resultDiv.textContent = "请求失败";
      if (mode === "multiple") {
        combinedResultSpan.textContent = "请求失败";
      }
    });
}

// 显示多数字识别结果
function displayMultipleDigitsResults(data) {
  // 显示组合结果
  combinedResultSpan.textContent = data.combined_result;

  // 显示可视化图像
  if (data.visualization) {
    digitVisualization.src = data.visualization;
    digitVisualization.style.display = "block";
  }

  // 清除之前的数字列表
  digitsList.innerHTML = "";

  // 添加每个识别出的数字
  data.digits.forEach((digit) => {
    const digitItem = document.createElement("div");
    digitItem.className = "digit-item";

    const digitNumber = document.createElement("div");
    digitNumber.className = "digit-item-number";
    digitNumber.textContent = digit.digit;

    const digitConfidence = document.createElement("div");
    digitConfidence.className = "digit-item-confidence";
    digitConfidence.textContent = `置信度: ${(digit.confidence * 100).toFixed(
      2
    )}%`;

    const digitImage = document.createElement("img");
    digitImage.className = "digit-item-image";
    digitImage.src = digit.processed_image;
    digitImage.alt = `数字 ${digit.digit}`;

    digitItem.appendChild(digitNumber);
    digitItem.appendChild(digitConfidence);
    digitItem.appendChild(digitImage);

    digitsList.appendChild(digitItem);
  });

  // 在多数字模式下，不需要更新单数字识别区域
  // 因为单数字识别区域已经被隐藏
}

// 在页面加载完成后初始化应用
document.addEventListener("DOMContentLoaded", function () {
  console.log("页面加载完成");
  init();
  healthCheck();
});
