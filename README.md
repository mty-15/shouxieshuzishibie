# 手写数字识别系统

这是一个基于深度学习的手写数字识别系统，使用了优化的卷积神经网络模型，能够准确识别手写数字。

## 功能特点

- 高精度的手写数字识别（准确率超过 99%）
- 优化的卷积神经网络模型
- 实时识别与反馈
- 支持批量识别
- 模型量化优化
- 完善的日志记录
- 跨域支持
- 响应式前端界面
- 环境变量配置
- Docker 支持

## 环境要求

- Python 3.10+
- TensorFlow 2.13.0+
- Flask 2.3.3+
- 其他依赖见 requirements.txt

## 安装与运行

### 方法一：使用 Docker（推荐）

1. 克隆仓库：

   ```bash
   git clone https://github.com/yourusername/shouxieshuzishibie.git
   cd shouxieshuzishibie
   ```

2. 使用 Docker Compose 启动：

   ```bash
   docker-compose up -d
   ```

3. 访问应用：
   ```
   http://localhost:5000
   ```

### 方法二：手动安装

1. 克隆仓库：

   ```bash
   git clone https://github.com/yourusername/shouxieshuzishibie.git
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

   # 生产环境（需要安装 gunicorn）
   gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
   ```

6. 访问应用：
   ```
   http://localhost:5000
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

### 3. 获取模型信息

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

### 4. 健康检查

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

1. **模型架构优化**：使用了深层卷积神经网络，包含批归一化和 Dropout 层
2. **数据增强**：通过旋转、缩放、平移等方式增强训练数据
3. **图像预处理**：包括边缘增强、高斯模糊、归一化等
4. **模型量化**：减小模型体积，提高推理速度
5. **学习率调度**：使用指数衰减的学习率调度
6. **早停策略**：避免过拟合，保存最佳模型

## 环境变量配置

项目支持通过环境变量或 `.env` 文件进行配置：

| 环境变量                | 描述              | 默认值        |
| ----------------------- | ----------------- | ------------- |
| `FLASK_APP`             | Flask 应用入口    | `app.py`      |
| `FLASK_ENV`             | 环境类型          | `development` |
| `DEBUG`                 | 是否开启调试模式  | `True`        |
| `PORT`                  | 服务端口          | `5000`        |
| `HOST`                  | 服务主机          | `0.0.0.0`     |
| `MODEL_DIR`             | 模型存储目录      | `models`      |
| `LOG_LEVEL`             | 日志级别          | `INFO`        |
| `CACHE_TYPE`            | 缓存类型          | `simple`      |
| `CACHE_DEFAULT_TIMEOUT` | 缓存超时时间      | `3600`        |
| `GUNICORN`              | 是否使用 Gunicorn | `False`       |
| `IGNORE_DEPENDENCIES`   | 是否忽略依赖检查  | `False`       |

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

## Docker 部署

项目提供了 Dockerfile 和 docker-compose.yml 文件，可以轻松地使用 Docker 进行部署：

```bash
# 构建镜像
docker build -t shouxieshuzishibie .

# 运行容器
docker run -p 5000:5000 shouxieshuzishibie

# 或者使用 Docker Compose
docker-compose up -d
```

## 许可证

MIT

## 联系方式

如有问题或建议，请联系：your.email@example.com
