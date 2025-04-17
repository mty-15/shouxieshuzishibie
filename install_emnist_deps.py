import subprocess
import sys
import os

def print_header(message):
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(command):
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
    print_header("安装EMNIST数据集训练所需依赖")
    
    # 安装tensorflow-datasets
    print("安装tensorflow-datasets...")
    if not run_command("pip install tensorflow-datasets"):
        print("安装tensorflow-datasets失败，请手动安装")
        return False
    
    # 安装scikit-learn
    print("安装scikit-learn...")
    if not run_command("pip install scikit-learn"):
        print("安装scikit-learn失败，请手动安装")
        return False
    
    # 安装seaborn
    print("安装seaborn...")
    if not run_command("pip install seaborn"):
        print("安装seaborn失败，请手动安装")
        return False
    
    # 安装opencv-python
    print("安装opencv-python...")
    if not run_command("pip install opencv-python"):
        print("安装opencv-python失败，请手动安装")
        return False
    
    # 安装tensorflow-model-optimization
    print("安装tensorflow-model-optimization...")
    if not run_command("pip install tensorflow-model-optimization"):
        print("安装tensorflow-model-optimization失败，请手动安装")
        return False
    
    print_header("所有依赖安装完成")
    print("现在您可以运行 python train_emnist_model.py 来训练EMNIST模型")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
