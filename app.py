import os
import io
import re
import sys
import base64
import logging
import datetime
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_caching import Cache

from PIL import Image, ImageOps, ImageFilter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 加载环境变量
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# 模型版本控制
MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0.0')

# 创建 Flask 应用
app = Flask(__name__, static_folder='static')

# 配置应用
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', os.urandom(24).hex()),
    DEBUG=os.getenv('DEBUG', 'False').lower() in ('true', '1', 't'),
    CACHE_TYPE=os.getenv('CACHE_TYPE', 'simple'),
    CACHE_DEFAULT_TIMEOUT=int(os.getenv('CACHE_DEFAULT_TIMEOUT', 3600))
)

# 启用跨域支持
CORS(app)

# 配置缓存
cache = Cache(app)

# 配置日志
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO'))
log_file = os.getenv('LOG_FILE', 'logs/app.log')
log_dir = os.path.dirname(log_file)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

handler = RotatingFileHandler(log_file, maxBytes=10000, backupCount=5)
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 添加控制台日志
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter)

app.logger.addHandler(handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(log_level)

# 设置 TensorFlow 日志级别
tf.get_logger().setLevel(log_level)

# 改进的图像预处理
def preprocess_image(img):
    # 调整大小确保尺寸一致
    img = img.resize((28, 28), Image.LANCZOS)
    # 转换为灰度图
    img = img.convert('L')
    # 反色处理(手写数字通常白底黑字，MNIST是黑底白字)
    img = ImageOps.invert(img)
    # 边缘增强
    img = img.filter(ImageFilter.EDGE_ENHANCE())
    # 高斯模糊去噪
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    # 归一化
    img = np.array(img).astype('float32') / 255.0
    # 中心化
    img = (img - np.mean(img)) / (np.std(img) + 1e-7)  # 添加小值防止除零
    return img.reshape(1, 28, 28, 1)

# 数据增强生成器
def create_augmenter():
    return ImageDataGenerator(
        rotation_range=15,  # 增加旋转范围
        width_shift_range=0.15,  # 增加位移范围
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,  # 添加剪切变换
        fill_mode='nearest',  # 填充模式
        brightness_range=[0.8, 1.2]  # 亮度变化
    )

# 构建优化后的模型
def build_improved_model():
    model = Sequential([
        # 添加更多特征提取能力
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),  # 增加一层
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # 使用更先进的优化器和学习率调度
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 模型量化函数
def quantize_model(model):
    try:
        import tensorflow_model_optimization as tfmot

        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(model)
        q_aware_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return q_aware_model
    except ImportError:
        app.logger.warning("TensorFlow Model Optimization 库未安装，跳过模型量化")
        return model

# 训练优化后的模型
def train_improved_model():
    app.logger.info("开始加载 MNIST 数据集")
    try:
        # 加载数据集
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 数据预处理
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        app.logger.info(f"数据集加载成功: 训练集 {x_train.shape[0]} 样本, 测试集 {x_test.shape[0]} 样本")

        # 构建模型
        app.logger.info("构建改进的模型")
        model = build_improved_model()
        model.summary(print_fn=lambda x: app.logger.debug(x))

        # 创建数据增强器
        augmenter = create_augmenter()

        # 添加早停回调
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )

        # 添加模型检查点
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        # 添加TensorBoard日志
        log_dir = os.path.join('logs', 'tensorboard', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )

        # 使用数据增强训练模型
        app.logger.info(f"开始训练模型: 批次大小={BATCH_SIZE}, 迭代次数={EPOCHS}")
        history = model.fit(
            augmenter.flow(x_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // BATCH_SIZE,
            callbacks=[early_stopping, checkpoint, tensorboard_callback],
            verbose=1
        )

        # 评估模型
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        app.logger.info(f"模型训练完成: 测试集准确率={test_acc:.4f}, 损失={test_loss:.4f}")

        # 保存标准模型
        model.save(STANDARD_MODEL_PATH)
        app.logger.info(f"标准模型保存成功: {STANDARD_MODEL_PATH}")

        # 尝试量化模型
        try:
            app.logger.info("尝试量化模型")
            quantized_model = quantize_model(model)
            quantized_model.save(QUANTIZED_MODEL_PATH)
            app.logger.info(f"量化模型保存成功: {QUANTIZED_MODEL_PATH}")
            return quantized_model
        except Exception as e:
            app.logger.warning(f"模型量化失败: {str(e)}")
            return model

    except Exception as e:
        app.logger.error(f"训练模型时出错: {str(e)}")
        raise

# 模型目录配置
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 模型文件路径
QUANTIZED_MODEL_PATH = os.path.join(MODEL_DIR, 'quantized_mnist_model.h5')
STANDARD_MODEL_PATH = os.path.join(MODEL_DIR, 'improved_mnist_model.h5')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_mnist_model.h5')

# 训练参数
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
EPOCHS = int(os.getenv('EPOCHS', 15))

# 使用缓存加速模型加载
@cache.cached(timeout=int(os.getenv('CACHE_DEFAULT_TIMEOUT', 3600)))
def get_model():
    try:
        # 先尝试加载量化模型
        if os.path.exists(QUANTIZED_MODEL_PATH):
            app.logger.info(f"加载量化模型: {QUANTIZED_MODEL_PATH}")
            return load_model(QUANTIZED_MODEL_PATH)

        # 尝试加载最佳模型
        elif os.path.exists(BEST_MODEL_PATH):
            app.logger.info(f"加载最佳模型: {BEST_MODEL_PATH}")
            return load_model(BEST_MODEL_PATH)

        # 尝试加载标准模型
        elif os.path.exists(STANDARD_MODEL_PATH):
            app.logger.info(f"加载标准模型: {STANDARD_MODEL_PATH}")
            return load_model(STANDARD_MODEL_PATH)

        # 没有找到模型，训练新模型
        else:
            app.logger.info("没有找到预训练模型，开始训练新模型")
            return train_improved_model()
    except Exception as e:
        app.logger.warning(f"加载模型失败: {str(e)}")
        app.logger.info("开始训练新模型")
        return train_improved_model()

# 在应用启动时预热模型
app.logger.info("加载模型...")
model = get_model()

# 预热模型，避免第一次请求延迟
app.logger.info("预热模型...")
try:
    dummy_input = np.zeros((1, 28, 28, 1))
    _ = model.predict(dummy_input)
    app.logger.info("模型预热成功")
except Exception as e:
    app.logger.error(f"模型预热失败: {str(e)}")

app.logger.info("模型准备就绪")

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            app.logger.warning('接收到无效请求数据')
            return jsonify({'error': '无效的请求数据'}), 400

        image_data = re.sub('^data:image/.+;base64,', '', data['image'])
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # 使用改进的预处理
        processed_img = preprocess_image(img)

        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        confidence = float(np.max(prediction))

        app.logger.info(f'成功识别数字: {digit}, 置信度: {confidence:.4f}')
        return jsonify({
            'digit': int(digit),
            'confidence': confidence,
            'model_version': MODEL_VERSION
        })
    except Exception as e:
        app.logger.error(f'识别过程中出错: {str(e)}')
        return jsonify({'error': str(e)}), 500

# 添加批量预测功能
@app.route('/recognize_batch', methods=['POST'])
def recognize_batch():
    try:
        data = request.get_json()
        if not data or 'images' not in data or not isinstance(data['images'], list):
            app.logger.warning('接收到无效的批量请求数据')
            return jsonify({'error': '无效的批量请求数据'}), 400

        results = []

        for item in data['images']:
            try:
                image_data = re.sub('^data:image/.+;base64,', '', item)
                img = Image.open(io.BytesIO(base64.b64decode(image_data)))
                processed_img = preprocess_image(img)
                prediction = model.predict(processed_img)
                digit = np.argmax(prediction)
                confidence = float(np.max(prediction))
                results.append({
                    'digit': int(digit),
                    'confidence': confidence
                })
            except Exception as e:
                results.append({
                    'error': str(e)
                })

        app.logger.info(f'批量识别完成: {len(results)} 个图像')
        return jsonify({
            'results': results,
            'model_version': MODEL_VERSION
        })
    except Exception as e:
        app.logger.error(f'批量识别过程中出错: {str(e)}')
        return jsonify({'error': str(e)}), 500

# 添加模型信息端点
@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        # 获取模型文件大小
        model_size = 0
        model_path = 'improved_mnist_model.h5'
        quantized_model_path = 'quantized_mnist_model.h5'

        if os.path.exists(quantized_model_path):
            model_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
            model_type = '量化CNN'
            model_path = quantized_model_path
        elif os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            model_type = 'CNN'

        # 获取模型最后修改时间
        last_modified = ''
        if os.path.exists(model_path):
            timestamp = os.path.getmtime(model_path)
            last_modified = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            'model_type': model_type,
            'model_version': MODEL_VERSION,
            'input_shape': '28x28x1',
            'model_size_mb': round(model_size, 2),
            'last_modified': last_modified
        })
    except Exception as e:
        app.logger.error(f'获取模型信息时出错: {str(e)}')
        return jsonify({'error': str(e)}), 500

# 添加静态文件路由
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# 添加健康检查端点
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'version': MODEL_VERSION,
        'timestamp': datetime.datetime.now().isoformat()
    })

# 添加错误处理
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

# 检查依赖版本
def check_dependencies():
    try:
        from check_dependencies import check_dependencies as check
        return check()
    except ImportError:
        app.logger.warning("依赖检查模块未找到，跳过依赖检查")
        return True

if __name__ == '__main__':
    # 检查依赖版本
    ignore_deps = os.getenv('IGNORE_DEPENDENCIES', 'False').lower() in ('true', '1', 't') or '--ignore-dependencies' in sys.argv

    if not check_dependencies() and not ignore_deps:
        app.logger.error("依赖检查失败，请安装兼容的依赖版本或设置 IGNORE_DEPENDENCIES=True")
        sys.exit(1)

    # 从环境变量获取端口，便于云部署
    port = int(os.getenv('PORT', 5000))
    # 生产环境禁用调试模式
    debug = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    # 允许外部访问
    host = os.getenv('HOST', '0.0.0.0')

    app.logger.info(f"启动服务器: 主机={host}, 端口={port}, 调试模式={debug}")

    # 如果存在 gunicorn，使用 gunicorn 启动
    if os.getenv('GUNICORN', 'False').lower() in ('true', '1', 't'):
        # 这里不需要调用 app.run()，因为 Gunicorn 会处理这个
        app.logger.info("使用 Gunicorn 启动应用")
    else:
        # 开发环境使用 Flask 内置服务器
        app.run(host=host, port=port, debug=debug)