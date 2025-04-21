import os
import sys
import subprocess

# 设置环境变量以禁用SSL验证
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''

# 构建命令行参数
cmd = [sys.executable, 'train_emnist_model.py']
cmd.extend(sys.argv[1:])  # 添加传递给此脚本的所有参数

print(f"运行命令: {' '.join(cmd)}")
print("已禁用SSL证书验证")

# 运行train_emnist_model.py
subprocess.run(cmd)
