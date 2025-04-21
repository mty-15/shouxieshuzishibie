# 改进版手写数字识别系统

这是一个基于深度学习的手写数字识别系统，使用了优化的卷积神经网络模型，能够准确识别手写数字。支持 MNIST 和 EMNIST 数据集训练，并提供了改进的用户界面和画布交互体验。

## 功能特点

- 高精度的手写数字识别（准确率超过 99%）
- 支持多数字识别，可以识别一张图片上的多个数字
- 支持图片拍照和上传功能
- 支持 MNIST 和 EMNIST 数据集训练
- 优化的卷积神经网络模型，支持 ResNet 架构
- 实时识别与反馈，显示所有数字的预测概率
- 改进的画布交互，支持画笔粗细调节和网格显示
- 支持批量识别
- 模型量化优化
- 完善的日志记录
- 跨域支持
- 响应式前端界面
- 环境变量配置

## 环境要求

- Python 3.10+
- TensorFlow 2.13.0
- Flask 2.3.3
- NumPy 1.24.3
- Pillow 10.0.1
- Matplotlib 3.7.2
- 其他依赖见 requirements.txt

## 安装与运行

1. 克隆仓库：

   ```bash
   git clone https://github.com/mty-15/shouxieshuzishibie.git
   cd shouxieshuzishibie
   ```

2. 创建并激活虚拟环境（可选）：

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

4. 创建环境变量文件：

   ```bash
   cp .env.example .env
   # 根据需要编辑 .env 文件
   ```

5. 运行服务器：

   ```bash
   # 开发环境
   python app.py

   # 如果依赖版本不兼容，可以使用以下命令忽略依赖检查
   python app.py --ignore-dependencies

   # 生产环境（使用 Waitress）
   python run_with_waitress.py
   ```

6. 访问应用：
   ```
   http://localhost:5002
   ```

## API 接口

### 1. 识别单个数字

- **URL**: `/recognize`
- **方法**: POST
- **请求体**:
  ```json
  {
    "image": "base64编码的图像数据"
  }
  ```
- **响应**:
  ```json
  {
    "digit": 5,
    "confidence": 0.9987,
    "model_version": "1.0.0"
  }
  ```

### 2. 批量识别数字

- **URL**: `/recognize_batch`
- **方法**: POST
- **请求体**:
  ```json
  {
    "images": ["base64编码的图像数据1", "base64编码的图像数据2", ...]
  }
  ```
- **响应**:
  ```json
  {
    "results": [
      {
        "digit": 5,
        "confidence": 0.9987
      },
      {
        "digit": 3,
        "confidence": 0.9876
      }
    ],
    "model_version": "1.0.0"
  }
  ```

### 3. 多数字识别

- **URL**: `/recognize_multiple`
- **方法**: POST
- **请求体**:
  ```json
  {
    "image": "base64编码的图像数据"
  }
  ```
- **响应**:
  ```json
  {
    "combined_result": "123",
    "digits": [
      {
        "digit": 1,
        "confidence": 0.9987,
        "position": { "x": 10, "y": 15, "width": 20, "height": 30 },
        "processed_image": "base64编码的处理后图像",
        "original_segment": "base64编码的原始分割图像"
      },
      {
        "digit": 2,
        "confidence": 0.9876,
        "position": { "x": 40, "y": 15, "width": 20, "height": 30 },
        "processed_image": "base64编码的处理后图像",
        "original_segment": "base64编码的原始分割图像"
      },
      {
        "digit": 3,
        "confidence": 0.9765,
        "position": { "x": 70, "y": 15, "width": 20, "height": 30 },
        "processed_image": "base64编码的处理后图像",
        "original_segment": "base64编码的原始分割图像"
      }
    ],
    "count": 3,
    "model_version": "1.0.0"
  }
  ```

### 4. 多数字调试识别

- **URL**: `/recognize_multiple_debug`
- **方法**: POST
- **请求体**:
  ```json
  {
    "image": "base64编码的图像数据"
  }
  ```
- **响应**:
  ```json
  {
    "combined_result": "123",
    "digits": [
      {
        "digit": 1,
        "confidence": 0.9987,
        "position": { "x": 10, "y": 15, "width": 20, "height": 30 },
        "processed_image": "base64编码的处理后图像",
        "original_segment": "base64编码的原始分割图像",
        "all_predictions": [
          0.001, 0.9987, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
      }
      // 其他数字...
    ],
    "count": 3,
    "visualization": "base64编码的分割可视化图像",
    "model_version": "1.0.0"
  }
  ```

### 5. 获取模型信息

- **URL**: `/model_info`
- **方法**: GET
- **响应**:
  ```json
  {
    "model_type": "量化CNN",
    "model_version": "1.0.0",
    "input_shape": "28x28x1",
    "model_size_mb": 2.35,
    "last_modified": "2023-04-14 15:30:45"
  }
  ```

### 6. 健康检查

- **URL**: `/health`
- **方法**: GET
- **响应**:
  ```json
  {
    "status": "ok",
    "version": "1.0.0",
    "timestamp": "2023-04-14T15:30:45.123456"
  }
  ```

## 模型优化

本项目使用了多种优化技术来提高模型性能：

1. **模型架构优化**：使用了深层卷积神经网络，包含批归一化和 Dropout 层，支持 ResNet 架构
2. **数据集扩展**：支持 MNIST 和 EMNIST 数据集训练，提高模型泛化能力
3. **数据增强**：通过旋转、缩放、平移等方式增强训练数据
4. **图像预处理**：包括边缘增强、二值化、归一化等
5. **模型量化**：减小模型体积，提高推理速度
6. **学习率调度**：使用指数衰减的学习率调度
7. **早停策略**：避免过拟合，保存最佳模型

## 环境变量配置

项目支持通过环境变量或 `.env` 文件进行配置：

| 环境变量                | 描述             | 默认值        |
| ----------------------- | ---------------- | ------------- |
| `FLASK_APP`             | Flask 应用入口   | `app.py`      |
| `FLASK_ENV`             | 环境类型         | `development` |
| `DEBUG`                 | 是否开启调试模式 | `True`        |
| `PORT`                  | 服务端口         | `5002`        |
| `HOST`                  | 服务主机         | `0.0.0.0`     |
| `MODEL_DIR`             | 模型存储目录     | `models`      |
| `LOG_LEVEL`             | 日志级别         | `INFO`        |
| `CACHE_TYPE`            | 缓存类型         | `simple`      |
| `CACHE_DEFAULT_TIMEOUT` | 缓存超时时间     | `3600`        |
| `IGNORE_DEPENDENCIES`   | 是否忽略依赖检查 | `False`       |
| `MODEL_VERSION`         | 模型版本号       | `1.0.0`       |

## 日志与监控

系统自动记录日志到 `logs/app.log` 文件，包括：

- 模型加载和训练信息
- API 请求和响应
- 错误和异常
- 性能指标

此外，系统还支持 TensorBoard 日志，可以通过以下命令查看：

```bash
tensorboard --logdir=logs/tensorboard
```

## 训练模型

项目提供了多个训练脚本，可以轻松训练不同的模型：

```bash
# 训练改进版MNIST模型
python train_advanced_model.py

# 训练EMNIST模型
python train_emnist_model.py

# 一键式训练和运行
python improve_and_run.py  # MNIST
python train_and_run_emnist.py  # EMNIST
```

训练好的模型将保存在`models`目录下，应用会自动加载最新的模型。

## 前端界面改进

改进版的前端界面提供了多项增强功能：

1. **多数字识别**：可以识别一张图片上的多个数字，并显示分割可视化结果
2. **拍照/上传功能**：支持使用摄像头拍照或上传图片进行识别
3. **模式切换**：可以在单数字识别和多数字识别模式之间切换
4. **画笔粗细调节**：可以通过滑块调节画笔粗细
5. **网格显示**：可以开启/关闭网格线，辅助绘制
6. **实时预览**：显示模型实际看到的图像
7. **所有数字的预测概率**：直观地显示每个数字的预测概率
8. **美观的响应式设计**：适应不同屏幕尺寸

## 许可证

MIT
