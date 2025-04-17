import os
import sys
import subprocess
import time
import argparse

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
    parser = argparse.ArgumentParser(description='训练EMNIST模型并运行应用')
    parser.add_argument('--skip-deps', action='store_true', help='跳过依赖安装')
    parser.add_argument('--skip-train', action='store_true', help='跳过模型训练')
    parser.add_argument('--model', type=str, default='resnet', choices=['cnn', 'resnet'], 
                        help='选择模型类型: cnn或resnet (默认: resnet)')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮次 (默认: 20)')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小 (默认: 128)')
    parser.add_argument('--quantize', action='store_true', help='是否量化模型')
    parser.add_argument('--port', type=int, default=5002, help='应用运行端口 (默认: 5002)')
    
    args = parser.parse_args()
    
    print_header("EMNIST手写数字识别系统训练与部署")
    
    # 安装依赖
    if not args.skip_deps:
        if not run_command("python install_emnist_deps.py", "安装EMNIST依赖"):
            print("依赖安装失败，请手动安装依赖后重试")
            return False
    
    # 训练模型
    if not args.skip_train:
        train_cmd = f"python train_emnist_model.py --model {args.model} --epochs {args.epochs} --batch-size {args.batch_size}"
        if args.quantize:
            train_cmd += " --quantize"
        
        if not run_command(train_cmd, f"训练EMNIST {args.model.upper()} 模型"):
            print("模型训练失败，请检查错误信息")
            return False
    
    # 更新端口
    if args.port != 5002:
        # 更新run_with_waitress.py中的端口
        with open('run_with_waitress.py', 'r') as f:
            content = f.read()
        
        content = content.replace('port=5002', f'port={args.port}')
        
        with open('run_with_waitress.py', 'w') as f:
            f.write(content)
        
        print(f"应用端口已更新为: {args.port}")
    
    # 启动应用
    print_header("启动应用")
    print(f"应用将在 http://localhost:{args.port} 启动")
    print("按 Ctrl+C 停止应用")
    time.sleep(2)
    
    # 使用subprocess.Popen启动应用，这样不会阻塞当前进程
    process = subprocess.Popen(["python", "run_with_waitress.py"])
    
    # 等待应用启动
    time.sleep(5)
    
    # 打开浏览器
    if sys.platform == 'win32':
        os.system(f"start http://localhost:{args.port}")
    elif sys.platform == 'darwin':  # macOS
        os.system(f"open http://localhost:{args.port}")
    else:  # Linux
        os.system(f"xdg-open http://localhost:{args.port}")
    
    print("\n应用已启动，请在浏览器中使用。按Ctrl+C停止应用。")
    
    try:
        # 保持脚本运行，直到用户按Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n接收到停止信号，正在关闭应用...")
        process.terminate()
        process.wait()
        print("应用已关闭")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
