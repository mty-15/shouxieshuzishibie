import os
import sys
import subprocess
import time

def print_header(message):
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(command, description):
    print_header(description)
    print(f"执行命令: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"命令执行成功!")
        if result.stdout:
            print("\n输出:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        return True
    else:
        print(f"命令执行失败，错误码: {result.returncode}")
        if result.stderr:
            print("\n错误输出:")
            print(result.stderr)
        return False

def main():
    print_header("手写数字识别系统改进计划")
    
    # 检查依赖
    print("检查依赖...")
    try:
        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from flask import Flask
        from waitress import serve
        print("所有必要的依赖已安装!")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        choice = input("是否安装缺失的依赖? (y/n): ")
        if choice.lower() == 'y':
            run_command("pip install -r requirements.txt", "安装依赖")
        else:
            print("未安装依赖，退出程序")
            return
    
    # 询问用户是否训练新模型
    train_model = input("是否训练新的ResNet模型? 这可能需要较长时间 (y/n): ")
    if train_model.lower() == 'y':
        print_header("训练新模型")
        run_command("python train_advanced_model.py", "训练ResNet模型")
    
    # 询问用户是否启动应用
    start_app = input("是否启动改进版应用? (y/n): ")
    if start_app.lower() == 'y':
        print_header("启动应用")
        print("应用将在 http://localhost:5002 启动")
        print("按 Ctrl+C 停止应用")
        time.sleep(2)
        os.system("python run_with_waitress.py")
    
    print_header("程序结束")

if __name__ == "__main__":
    main()
