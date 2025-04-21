import logging
import os
import shutil
import sys
import zipfile

import requests

# 尝试导入tqdm
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，安装它
    print("正在安装tqdm包...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm
    print("tqdm安装成功")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/emnist_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EMNIST下载")

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

def download_file(url, save_path, chunk_size=8192):
    """下载文件并显示进度条"""
    try:
        # 禁用SSL证书验证的警告
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 下载文件
        logger.info(f"开始下载: {url}")
        logger.info(f"保存到: {save_path}")

        # 使用requests下载，禁用SSL验证
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()  # 确保请求成功

        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))

        # 使用tqdm显示进度条
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # 过滤掉保持连接alive的空chunks
                    f.write(chunk)
                    bar.update(len(chunk))

        logger.info(f"下载完成: {save_path}")
        return True
    except Exception as e:
        logger.error(f"下载失败: {str(e)}")
        return False

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    try:
        logger.info(f"开始解压: {zip_path}")
        logger.info(f"解压到: {extract_to}")

        # 创建解压目录
        os.makedirs(extract_to, exist_ok=True)

        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        logger.info(f"解压完成")
        return True
    except Exception as e:
        logger.error(f"解压失败: {str(e)}")
        return False

def main():
    # EMNIST数据集URL
    emnist_url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

    # 保存路径
    data_dir = os.path.join(os.getcwd(), "data")
    emnist_dir = os.path.join(data_dir, "emnist")
    zip_path = os.path.join(data_dir, "emnist.zip")

    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(emnist_dir, exist_ok=True)

    # 下载EMNIST数据集
    if not download_file(emnist_url, zip_path):
        logger.error("EMNIST数据集下载失败")
        return False

    # 解压EMNIST数据集
    if not extract_zip(zip_path, emnist_dir):
        logger.error("EMNIST数据集解压失败")
        return False

    logger.info("EMNIST数据集下载和解压完成")
    logger.info(f"数据集保存在: {emnist_dir}")

    # 列出解压后的文件
    logger.info("解压后的文件列表:")
    for root, dirs, files in os.walk(emnist_dir):
        for file in files:
            logger.info(f" - {os.path.join(root, file)}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
