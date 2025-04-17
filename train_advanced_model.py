import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, 
    Dropout, Flatten, Dense, Add, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
import datetime
import matplotlib.pyplot as plt

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

# 模型文件路径
RESNET_MODEL_PATH = os.path.join(MODEL_DIR, 'resnet_mnist_model.h5')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_mnist_model.h5')

# 加载和预处理数据
def load_and_preprocess_data():
    print("加载MNIST数据集...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 重塑数据
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # 标准化 - 使用MNIST的均值和标准差
    mean = 0.1307
    std = 0.3081
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # 独热编码
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"数据集加载完成: 训练集 {x_train.shape[0]} 样本, 测试集 {x_test.shape[0]} 样本")
    return (x_train, y_train), (x_test, y_test)

# 创建增强的数据生成器
def create_advanced_augmenter():
    return ImageDataGenerator(
        rotation_range=15,           # 随机旋转
        width_shift_range=0.15,      # 水平平移
        height_shift_range=0.15,     # 垂直平移
        zoom_range=0.15,             # 随机缩放
        shear_range=0.15,            # 剪切变换
        fill_mode='nearest',         # 填充模式
        brightness_range=[0.8, 1.2], # 亮度变化
        # 添加更多增强
        channel_shift_range=0.1,     # 通道偏移
        horizontal_flip=False,       # 不进行水平翻转（数字翻转会改变含义）
        vertical_flip=False,         # 不进行垂直翻转
        preprocessing_function=lambda x: x + np.random.normal(0, 0.05, x.shape)  # 添加噪声
    )

# 创建ResNet风格的残差块
def residual_block(x, filters, kernel_size=3, strides=1, use_conv_shortcut=False):
    shortcut = x
    
    # 第一个卷积层
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)
    
    # 第二个卷积层
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # 如果需要，使用卷积快捷连接
    if use_conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # 添加残差连接
    x = Add()([x, shortcut])
    x = keras.activations.relu(x)
    
    return x

# 构建ResNet风格的模型
def build_resnet_model():
    inputs = Input(shape=(28, 28, 1))
    
    # 初始卷积层
    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)
    
    # 第一组残差块
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)
    
    # 第二组残差块
    x = residual_block(x, 64, use_conv_shortcut=True)
    x = residual_block(x, 64)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)
    
    # 第三组残差块
    x = residual_block(x, 128, use_conv_shortcut=True)
    x = residual_block(x, 128)
    
    # 全局平均池化
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    # 全连接层
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # 输出层
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # 使用学习率调度的优化器
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 训练模型并保存训练历史
def train_model(model, x_train, y_train, x_test, y_test):
    print("开始训练模型...")
    
    # 创建数据增强器
    augmenter = create_advanced_augmenter()
    
    # 创建回调函数
    callbacks = [
        # 早停
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 模型检查点
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # 学习率调度
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard日志
        TensorBoard(
            log_dir=os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # 训练模型
    history = model.fit(
        augmenter.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"模型训练完成: 测试集准确率={test_acc:.4f}, 损失={test_loss:.4f}")
    
    # 保存模型
    model.save(RESNET_MODEL_PATH)
    print(f"模型保存成功: {RESNET_MODEL_PATH}")
    
    return history

# 绘制训练历史
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()

# 测试模型在不同变换下的性能
def test_model_robustness(model, x_test, y_test):
    print("测试模型鲁棒性...")
    
    # 原始测试集
    original_loss, original_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"原始测试集: 准确率={original_acc:.4f}, 损失={original_loss:.4f}")
    
    # 旋转变换
    rotation_gen = ImageDataGenerator(rotation_range=20)
    rotation_iterator = rotation_gen.flow(x_test, y_test, batch_size=len(x_test), shuffle=False)
    x_rotation, y_rotation = next(rotation_iterator)
    rotation_loss, rotation_acc = model.evaluate(x_rotation, y_rotation, verbose=0)
    print(f"旋转变换: 准确率={rotation_acc:.4f}, 损失={rotation_loss:.4f}")
    
    # 缩放变换
    zoom_gen = ImageDataGenerator(zoom_range=0.2)
    zoom_iterator = zoom_gen.flow(x_test, y_test, batch_size=len(x_test), shuffle=False)
    x_zoom, y_zoom = next(zoom_iterator)
    zoom_loss, zoom_acc = model.evaluate(x_zoom, y_zoom, verbose=0)
    print(f"缩放变换: 准确率={zoom_acc:.4f}, 损失={zoom_loss:.4f}")
    
    # 添加噪声
    noise = np.random.normal(0, 0.1, x_test.shape)
    x_noise = np.clip(x_test + noise, 0, 1)
    noise_loss, noise_acc = model.evaluate(x_noise, y_test, verbose=0)
    print(f"添加噪声: 准确率={noise_acc:.4f}, 损失={noise_loss:.4f}")

# 主函数
def main():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 构建模型
    model = build_resnet_model()
    model.summary()
    
    # 训练模型
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # 绘制训练历史
    plot_history(history)
    
    # 测试模型鲁棒性
    test_model_robustness(model, x_test, y_test)

if __name__ == "__main__":
    main()
