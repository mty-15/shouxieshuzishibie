import base64
import datetime
import io
import logging
import os
import re
import sys
from datetime import datetime as dt
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
# 尝试导入 TensorFlow 和 Keras
import tensorflow as tf

# 尝试从tensorflow导入keras，如果失败则从独立的keras包导入
try:
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                         Dropout, Flatten, MaxPooling2D)
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
except ImportError:
    # 如果tensorflow.keras导入失败，尝试从独立的keras包导入
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential, load_model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from keras.utils import to_categorical
    from keras.preprocessing.image import ImageDataGenerator

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_caching import Cache
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageOps

# 根据 TensorFlow 版本选择正确的 Keras 导入方式
try:
    # 对于 TensorFlow 2.x
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                         Dropout, Flatten, MaxPooling2D)
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
except ImportError:
    # 对于独立的 Keras
    import keras
    from keras.datasets import mnist
    from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                              Flatten, MaxPooling2D)
    from keras.models import Sequential, load_model
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical


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
logging.getLogger('tensorflow').setLevel(log_level)

# 改进的图像预处理
def preprocess_image(img):
    # 记录原始尺寸
    app.logger.info(f"原始图像尺寸: {img.size}")

    # 先将图像转换为灰度图
    img = img.convert('L')

    # 二值化处理，增强对比度
    threshold = 200  # 调整阈值以获得更好的结果
    img = img.point(lambda p: 255 if p > threshold else 0)

    # 裁剪图像，去除多余的空白区域
    bbox = ImageOps.invert(img).getbbox()
    if bbox:
        img = img.crop(bbox)

    # 添加空白填充，保持数字居中
    width, height = img.size
    max_dim = max(width, height)
    new_size = int(max_dim * 1.2)  # 添加20%的空白填充

    # 创建新的正方形图像，白色背景
    new_img = Image.new('L', (new_size, new_size), 255)
    # 将原图像粘贴到中心
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2
    new_img.paste(img, (paste_x, paste_y))
    img = new_img

    # 调整大小为28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # 反色处理(手写数字通常白底黑字，MNIST是黑底白字)
    img = ImageOps.invert(img)

    # 边缘增强
    img = img.filter(ImageFilter.EDGE_ENHANCE())

    # 归一化
    img_array = np.array(img).astype('float32') / 255.0

    # 中心化，使用MNIST数据集的均值和标准差
    # MNIST均值约为0.1307，标准差约为0.3081
    img_array = (img_array - 0.1307) / 0.3081

    # 记录处理后的图像统计信息
    app.logger.info(f"处理后图像统计: 均值={np.mean(img_array):.4f}, 标准差={np.std(img_array):.4f}, 最大值={np.max(img_array):.4f}, 最小值={np.min(img_array):.4f}")

    return img_array.reshape(1, 28, 28, 1)

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
    try:
        # 尝试使用 tensorflow.keras
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    except AttributeError:
        # 如果使用独立的keras
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
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

        # 根据keras的导入方式选择正确的优化器
        try:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        except AttributeError:
            # 如果使用独立的keras
            optimizer = keras.optimizers.Adam(learning_rate=0.001)

        q_aware_model.compile(
            optimizer=optimizer,
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

        # 添加回调函数
        try:
            # 尝试使用 tensorflow.keras
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
        except AttributeError:
            # 如果使用独立的keras
            # 添加早停回调
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )

            # 添加模型检查点
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=BEST_MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )

            # 添加TensorBoard日志
            log_dir = os.path.join('logs', 'tensorboard', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = keras.callbacks.TensorBoard(
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

# EMNIST模型文件路径
EMNIST_MODEL_PATH = os.path.join(MODEL_DIR, 'emnist_model.h5')
QUANTIZED_EMNIST_MODEL_PATH = os.path.join(MODEL_DIR, 'quantized_emnist_model.h5')
BEST_EMNIST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_emnist_model.h5')

# 训练参数
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
EPOCHS = int(os.getenv('EPOCHS', 15))

# 使用缓存加速模型加载
@cache.cached(timeout=int(os.getenv('CACHE_DEFAULT_TIMEOUT', 3600)))
def get_model():
    try:
        # 优先尝试加载 EMNIST 模型
        if os.path.exists(QUANTIZED_EMNIST_MODEL_PATH):
            app.logger.info(f"加载量化EMNIST模型: {QUANTIZED_EMNIST_MODEL_PATH}")
            return load_model(QUANTIZED_EMNIST_MODEL_PATH)

        elif os.path.exists(BEST_EMNIST_MODEL_PATH):
            app.logger.info(f"加载最佳EMNIST模型: {BEST_EMNIST_MODEL_PATH}")
            return load_model(BEST_EMNIST_MODEL_PATH)

        elif os.path.exists(EMNIST_MODEL_PATH):
            app.logger.info(f"加载EMNIST模型: {EMNIST_MODEL_PATH}")
            return load_model(EMNIST_MODEL_PATH)

        # 如果没有EMNIST模型，尝试加载 MNIST 模型
        elif os.path.exists(QUANTIZED_MODEL_PATH):
            app.logger.info(f"加载量化MNIST模型: {QUANTIZED_MODEL_PATH}")
            return load_model(QUANTIZED_MODEL_PATH)

        elif os.path.exists(BEST_MODEL_PATH):
            app.logger.info(f"加载最佳MNIST模型: {BEST_MODEL_PATH}")
            return load_model(BEST_MODEL_PATH)

        elif os.path.exists(STANDARD_MODEL_PATH):
            app.logger.info(f"加载标准MNIST模型: {STANDARD_MODEL_PATH}")
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

@app.route('/recognize_with_debug', methods=['POST'])
def recognize_with_debug():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            app.logger.warning('接收到无效请求数据')
            return jsonify({'error': '无效的请求数据'}), 400

        image_data = re.sub('^data:image/.+;base64,', '', data['image'])
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # 使用改进的预处理
        processed_img = preprocess_image(img)

        # 将处理后的图像转换为可视化的格式
        # 将归一化的图像转换回0-255范围
        visual_img = np.squeeze(processed_img.copy())
        # 反转标准化
        visual_img = visual_img * 0.3081 + 0.1307
        # 转换回0-1范围
        visual_img = np.clip(visual_img, 0, 1)
        # 反色，使数字显示为白色，背景为黑色
        visual_img = visual_img * 255

        # 将NumPy数组转换为图像
        visual_pil = Image.fromarray(visual_img.astype('uint8'))

        # 将图像转换为Base64字符串
        buffered = io.BytesIO()
        visual_pil.save(buffered, format="PNG")
        processed_image_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # 将预测概率转换为列表
        all_predictions = prediction.flatten().tolist()

        app.logger.info(f'成功识别数字: {digit}, 置信度: {confidence:.4f}')
        return jsonify({
            'digit': int(digit),
            'confidence': confidence,
            'model_version': MODEL_VERSION,
            'all_predictions': all_predictions,
            'processed_image': processed_image_base64
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
    # 检查当前使用的模型类型
    model_type = '手写数字识别CNN'
    model_path = None
    dataset_type = 'MNIST'

    # 检查EMNIST模型
    if os.path.exists(QUANTIZED_EMNIST_MODEL_PATH):
        model_type = '量化EMNIST-ResNet'
        model_path = QUANTIZED_EMNIST_MODEL_PATH
        dataset_type = 'EMNIST'
    elif os.path.exists(BEST_EMNIST_MODEL_PATH):
        model_type = 'EMNIST-ResNet'
        model_path = BEST_EMNIST_MODEL_PATH
        dataset_type = 'EMNIST'
    elif os.path.exists(EMNIST_MODEL_PATH):
        model_type = 'EMNIST-CNN'
        model_path = EMNIST_MODEL_PATH
        dataset_type = 'EMNIST'
    # 检查MNIST模型
    elif os.path.exists(QUANTIZED_MODEL_PATH):
        model_type = '量化MNIST-CNN'
        model_path = QUANTIZED_MODEL_PATH
    elif os.path.exists(BEST_MODEL_PATH):
        model_type = 'MNIST-CNN'
        model_path = BEST_MODEL_PATH
    elif os.path.exists(STANDARD_MODEL_PATH):
        model_type = 'MNIST-CNN'
        model_path = STANDARD_MODEL_PATH

    # 获取模型大小
    model_size = 5.0  # 默认值
    if model_path and os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

    # 获取模型最后修改时间
    last_modified = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    if model_path and os.path.exists(model_path):
        timestamp = os.path.getmtime(model_path)
        last_modified = dt.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    return jsonify({
        'model_type': model_type,
        'model_version': MODEL_VERSION,
        'dataset': dataset_type,
        'input_shape': '28x28x1',
        'model_size_mb': round(model_size, 2),
        'last_modified': last_modified
    })

# 添加静态文件路由
@app.route('/')
def index():
    return send_from_directory('static', 'fixed_index.html')

@app.route('/css/<path:path>')
def serve_css(path):
    return send_from_directory('static/css', path)

@app.route('/js/<path:path>')
def serve_js(path):
    return send_from_directory('static/js', path)

# favicon.ico文件不存在，移除路由

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
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    app.logger.error(f"Server error: {str(error)}")
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
    port = int(os.getenv('PORT', 5002))
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
