import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/emnist_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EMNIST训练")

# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 配置参数
BATCH_SIZE = 128
EPOCHS = 50
MODEL_DIR = 'models'
LOG_DIR = os.path.join('logs', 'tensorboard')

# 确保目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# 模型文件路径
EMNIST_MODEL_PATH = os.path.join(MODEL_DIR, 'emnist_model.h5')
BEST_EMNIST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_emnist_model.h5')
QUANTIZED_EMNIST_MODEL_PATH = os.path.join(MODEL_DIR, 'quantized_emnist_model.h5')

def load_emnist_from_local():
    """从本地文件加载EMNIST数据集"""
    logger.info("尝试从本地文件加载EMNIST数据集...")
    try:
        # 检查数据目录是否存在
        data_dir = os.path.join(os.getcwd(), 'data', 'emnist')
        if not os.path.exists(data_dir):
            logger.error(f"数据目录不存在: {data_dir}")
            logger.error("请先运行 python download_emnist.py 下载数据集")
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 导入必要的库
        import gzip
        import struct

        import numpy as np

        # 定义读取函数，支持gzip压缩文件
        def read_idx(filename):
            # 检查是否为gzip文件
            is_gzip = filename.lower().endswith('.gz')
            open_func = gzip.open if is_gzip else open

            with open_func(filename, 'rb') as f:
                # 读取文件头
                magic = f.read(4)
                if len(magic) != 4:
                    raise ValueError("Invalid IDX file format")

                # 解析文件头
                magic_number = int.from_bytes(magic, byteorder='big')
                data_type = (magic_number >> 8) & 0xff
                dims = magic_number & 0xff

                # 读取各维度大小
                shape = []
                for i in range(dims):
                    dim_size = int.from_bytes(f.read(4), byteorder='big')
                    shape.append(dim_size)

                # 读取数据
                data = f.read()

                # 根据数据类型选择numpy类型
                dtype_map = {
                    0x08: np.uint8,   # unsigned byte
                    0x09: np.int8,    # signed byte
                    0x0B: np.int16,   # short (2 bytes)
                    0x0C: np.int32,   # int (4 bytes)
                    0x0D: np.float32, # float (4 bytes)
                    0x0E: np.float64  # double (8 bytes)
                }

                if data_type not in dtype_map:
                    raise ValueError(f"Unknown data type: {data_type}")

                # 转换为numpy数组并重塑
                return np.frombuffer(data, dtype=dtype_map[data_type]).reshape(shape)

        # 尝试查找训练和测试文件，优先使用digits数据集
        train_images_file = None
        train_labels_file = None
        test_images_file = None
        test_labels_file = None

        # 递归搜索所有文件
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # 优先选择digits数据集
                if 'digits' in file.lower():
                    if 'train' in file.lower() and 'images' in file.lower():
                        train_images_file = file_path
                    elif 'train' in file.lower() and 'labels' in file.lower():
                        train_labels_file = file_path
                    elif 'test' in file.lower() and 'images' in file.lower():
                        test_images_file = file_path
                    elif 'test' in file.lower() and 'labels' in file.lower():
                        test_labels_file = file_path

        # 如果没有找到digits数据集，则使用任何可用的数据集
        if not all([train_images_file, train_labels_file, test_images_file, test_labels_file]):
            logger.info("未找到digits数据集，尝试使用其他EMNIST数据集...")
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if not train_images_file and 'train' in file.lower() and 'images' in file.lower():
                        train_images_file = file_path
                    elif not train_labels_file and 'train' in file.lower() and 'labels' in file.lower():
                        train_labels_file = file_path
                    elif not test_images_file and 'test' in file.lower() and 'images' in file.lower():
                        test_images_file = file_path
                    elif not test_labels_file and 'test' in file.lower() and 'labels' in file.lower():
                        test_labels_file = file_path

        # 检查是否找到所有必要的文件
        if not all([train_images_file, train_labels_file, test_images_file, test_labels_file]):
            missing = []
            if not train_images_file: missing.append("训练图像文件")
            if not train_labels_file: missing.append("训练标签文件")
            if not test_images_file: missing.append("测试图像文件")
            if not test_labels_file: missing.append("测试标签文件")
            logger.error(f"缺少必要的数据文件: {', '.join(missing)}")
            logger.error("请先运行 python download_emnist.py 下载数据集")
            raise FileNotFoundError(f"缺少必要的数据文件: {', '.join(missing)}")

        # 读取数据
        logger.info(f"读取训练图像: {train_images_file}")
        train_images = read_idx(train_images_file)
        logger.info(f"读取训练标签: {train_labels_file}")
        train_labels = read_idx(train_labels_file)
        logger.info(f"读取测试图像: {test_images_file}")
        test_images = read_idx(test_images_file)
        logger.info(f"读取测试标签: {test_labels_file}")
        test_labels = read_idx(test_labels_file)

        # 转换为TensorFlow数据集
        train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
        test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')

        # 创建数据集
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

        logger.info(f"EMNIST数据集加载成功: {len(train_images)} 训练样本, {len(test_images)} 测试样本")
        return train_ds, test_ds
    except Exception as e:
        logger.error(f"从本地加载EMNIST数据集失败: {str(e)}")
        raise

def download_emnist():
    """下载EMNIST数据集"""
    logger.info("开始加载EMNIST数据集...")

    # 首先尝试从本地文件加载
    try:
        return load_emnist_from_local()
    except Exception as local_error:
        logger.warning(f"从本地加载失败: {str(local_error)}")
        logger.error("根据用户要求，不使用MNIST数据集替代EMNIST数据集")
        logger.error("请运行 python download_emnist.py 下载数据集")
        logger.error("然后重新运行本程序")
        raise Exception("请先运行 python download_emnist.py 下载数据集")

def preprocess_emnist(image, label):
    """预处理EMNIST数据"""
    # EMNIST图像需要转置和翻转以匹配MNIST的方向
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)

    # 归一化到[0,1]
    image = tf.cast(image, tf.float32) / 255.0

    # 标准化
    mean = 0.1736
    std = 0.3317
    image = (image - mean) / std

    # 独热编码标签
    label = tf.one_hot(label, 10)

    return image, label

def prepare_dataset(ds_train, ds_test):
    """准备数据集用于训练"""
    logger.info("准备数据集...")

    # 应用预处理
    ds_train = ds_train.map(preprocess_emnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_emnist, num_parallel_calls=tf.data.AUTOTUNE)

    # 缓存、打乱和批处理
    ds_train = ds_train.cache().shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    logger.info("数据集准备完成")
    return ds_train, ds_test

def create_data_augmentation():
    """创建数据增强层"""
    return keras.Sequential([
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.GaussianNoise(0.1)
    ])

def build_improved_model():
    """构建改进的CNN模型"""
    logger.info("构建改进的CNN模型...")

    # 创建数据增强层
    data_augmentation = create_data_augmentation()

    # 构建模型
    inputs = keras.Input(shape=(28, 28, 1))

    # 数据增强 - 仅在训练时应用
    x = data_augmentation(inputs)

    # 第一个卷积块
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)

    # 第二个卷积块
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)

    # 第三个卷积块
    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)

    # 全连接层
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)

    # 输出层
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("模型构建完成")
    return model

def build_resnet_model():
    """构建ResNet风格的模型"""
    logger.info("构建ResNet风格模型...")

    def residual_block(x, filters, kernel_size=3, strides=1, use_conv_shortcut=False):
        shortcut = x

        # 第一个卷积层
        x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        # 第二个卷积层
        x = keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        # 如果需要，使用卷积快捷连接
        if use_conv_shortcut:
            shortcut = keras.layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)

        # 添加残差连接
        x = keras.layers.add([x, shortcut])
        x = keras.layers.Activation('relu')(x)

        return x

    # 创建数据增强层
    data_augmentation = create_data_augmentation()

    # 构建模型
    inputs = keras.Input(shape=(28, 28, 1))

    # 数据增强 - 仅在训练时应用
    x = data_augmentation(inputs)

    # 初始卷积层
    x = keras.layers.Conv2D(32, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # 第一组残差块
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.25)(x)

    # 第二组残差块
    x = residual_block(x, 64, use_conv_shortcut=True)
    x = residual_block(x, 64)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.25)(x)

    # 第三组残差块
    x = residual_block(x, 128, use_conv_shortcut=True)
    x = residual_block(x, 128)

    # 全局平均池化
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)

    # 全连接层
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)

    # 输出层
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("ResNet模型构建完成")
    return model

def train_model(model, ds_train, ds_test, model_type="CNN"):
    """训练模型并保存"""
    logger.info(f"开始训练{model_type}模型...")

    # 创建回调函数
    callbacks = [
        # 早停
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 模型检查点
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_EMNIST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # 学习率调度
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard日志
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, f"emnist_{model_type.lower()}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            histogram_freq=1
        )
    ]

    # 训练模型
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test,
        callbacks=callbacks,
        verbose=1
    )

    # 保存模型
    model.save(EMNIST_MODEL_PATH)
    logger.info(f"模型保存成功: {EMNIST_MODEL_PATH}")

    return history, model

def quantize_model(model):
    """量化模型以减小大小"""
    logger.info("尝试量化模型...")
    try:
        import tensorflow_model_optimization as tfmot

        # 应用量化感知训练
        quantize_model = tfmot.quantization.keras.quantize_model

        # 克隆模型以避免修改原始模型
        q_aware_model = quantize_model(model)

        # 编译量化模型
        q_aware_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 保存量化模型
        q_aware_model.save(QUANTIZED_EMNIST_MODEL_PATH)
        logger.info(f"量化模型保存成功: {QUANTIZED_EMNIST_MODEL_PATH}")

        # 比较模型大小
        original_size = os.path.getsize(EMNIST_MODEL_PATH) / (1024 * 1024)
        quantized_size = os.path.getsize(QUANTIZED_EMNIST_MODEL_PATH) / (1024 * 1024)
        logger.info(f"原始模型大小: {original_size:.2f} MB")
        logger.info(f"量化模型大小: {quantized_size:.2f} MB")
        logger.info(f"大小减少: {(1 - quantized_size/original_size) * 100:.2f}%")

        return q_aware_model
    except ImportError:
        logger.warning("tensorflow-model-optimization 未安装，跳过模型量化")
        return model
    except Exception as e:
        logger.error(f"模型量化失败: {str(e)}")
        return model

def plot_history(history, model_type="CNN"):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))

    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title(f'EMNIST {model_type} 模型准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()

    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'EMNIST {model_type} 模型损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'emnist_{model_type.lower()}_training_history.png'))
    logger.info(f"训练历史图表已保存到 {os.path.join(MODEL_DIR, f'emnist_{model_type.lower()}_training_history.png')}")

def visualize_predictions(model, ds_test):
    """可视化模型预测"""
    logger.info("可视化模型预测...")

    # 获取一批测试数据
    for images, labels in ds_test.take(1):
        # 选择前16个样本
        images = images[:16]
        labels = labels[:16]

        # 获取预测
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels.numpy(), axis=1)

        # 绘制图像和预测
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i].numpy().reshape(28, 28), cmap='gray')
            color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
            plt.title(f"预测: {predicted_labels[i]}\n真实: {true_labels[i]}", color=color)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'emnist_predictions.png'))
        logger.info(f"预测可视化已保存到 {os.path.join(MODEL_DIR, 'emnist_predictions.png')}")
        break

def evaluate_model(model, ds_test):
    """评估模型性能"""
    logger.info("评估模型性能...")

    # 在测试集上评估
    loss, accuracy = model.evaluate(ds_test)
    logger.info(f"测试集损失: {loss:.4f}")
    logger.info(f"测试集准确率: {accuracy:.4f}")

    # 计算混淆矩阵
    y_true = []
    y_pred = []

    for images, labels in ds_test:
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    # 绘制混淆矩阵
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(MODEL_DIR, 'emnist_confusion_matrix.png'))
    logger.info(f"混淆矩阵已保存到 {os.path.join(MODEL_DIR, 'emnist_confusion_matrix.png')}")

    # 打印分类报告
    report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"分类报告:\n{report}")

    # 保存分类报告到文件
    with open(os.path.join(MODEL_DIR, 'emnist_classification_report.txt'), 'w') as f:
        f.write(report)

    return accuracy

def main():
    """主函数"""
    global EPOCHS, BATCH_SIZE

    parser = argparse.ArgumentParser(description='训练EMNIST手写数字识别模型')
    parser.add_argument('--model', type=str, default='resnet', choices=['cnn', 'resnet'],
                        help='选择模型类型: cnn或resnet (默认: resnet)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f'训练轮次 (默认: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'批次大小 (默认: {BATCH_SIZE})')
    parser.add_argument('--quantize', action='store_true', help='是否量化模型')

    args = parser.parse_args()

    # 更新全局参数
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    logger.info(f"开始EMNIST模型训练: 模型类型={args.model}, 轮次={EPOCHS}, 批次大小={BATCH_SIZE}")

    try:
        # 下载并准备数据集
        ds_train, ds_test = download_emnist()
        ds_train, ds_test = prepare_dataset(ds_train, ds_test)

        # 构建模型
        if args.model == 'resnet':
            model = build_resnet_model()
            model_type = "ResNet"
        else:
            model = build_improved_model()
            model_type = "CNN"

        # 打印模型摘要
        model.summary()

        # 训练模型
        history, trained_model = train_model(model, ds_train, ds_test, model_type)

        # 绘制训练历史
        plot_history(history, model_type)

        # 评估模型
        accuracy = evaluate_model(trained_model, ds_test)

        # 可视化预测
        visualize_predictions(trained_model, ds_test)

        # 量化模型
        if args.quantize:
            quantized_model = quantize_model(trained_model)
            # 评估量化模型
            logger.info("评估量化模型性能...")
            q_accuracy = evaluate_model(quantized_model, ds_test)
            logger.info(f"量化前准确率: {accuracy:.4f}, 量化后准确率: {q_accuracy:.4f}")

        logger.info("EMNIST模型训练完成!")

    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}", exc_info=True)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
