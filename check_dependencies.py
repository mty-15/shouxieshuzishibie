import importlib
import pkg_resources
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("依赖检查")

# 定义依赖版本范围
DEPENDENCIES = {
    'flask': {'min': '2.3.0', 'max': '2.4.0', 'recommended': '2.3.3'},
    'numpy': {'min': '1.24.0', 'max': '1.25.0', 'recommended': '1.24.3'},
    'pillow': {'min': '10.0.0', 'max': '10.1.0', 'recommended': '10.0.1'},
    'tensorflow': {'min': '2.13.0', 'max': '2.14.0', 'recommended': '2.13.0'},
    'matplotlib': {'min': '3.7.0', 'max': '3.8.0', 'recommended': '3.7.5'},
    'flask_cors': {'min': '4.0.0', 'max': '4.1.0', 'recommended': '4.0.0'},
    'flask_caching': {'min': '2.0.0', 'max': '2.1.0', 'recommended': '2.0.2'},
    'tensorflow_model_optimization': {'min': '0.7.0', 'max': '0.8.0', 'recommended': '0.7.5'},
    'python-dotenv': {'min': '1.0.0', 'max': '1.1.0', 'recommended': '1.0.0'},
    'gunicorn': {'min': '21.0.0', 'max': '22.0.0', 'recommended': '21.2.0'},
    'werkzeug': {'min': '2.3.0', 'max': '2.4.0', 'recommended': '2.3.7'},
}

def check_dependencies():
    """检查依赖版本是否兼容"""
    incompatible = []
    warnings = []
    
    for package_name, version_info in DEPENDENCIES.items():
        try:
            # 处理包名中的下划线和连字符
            dist_name = package_name.replace('_', '-')
            pkg_version = pkg_resources.get_distribution(dist_name).version
            
            min_version = version_info.get('min')
            max_version = version_info.get('max')
            recommended = version_info.get('recommended')
            
            # 检查版本是否在范围内
            if min_version and pkg_resources.parse_version(pkg_version) < pkg_resources.parse_version(min_version):
                incompatible.append(f"{package_name} 版本 {pkg_version} 低于最低要求 {min_version}")
            elif max_version and pkg_resources.parse_version(pkg_version) >= pkg_resources.parse_version(max_version):
                warnings.append(f"{package_name} 版本 {pkg_version} 高于推荐的最大版本 {max_version}")
            elif recommended and pkg_version != recommended:
                warnings.append(f"{package_name} 版本 {pkg_version} 不是推荐版本 {recommended}")
            else:
                logger.info(f"{package_name} 版本 {pkg_version} 兼容")
                
        except pkg_resources.DistributionNotFound:
            incompatible.append(f"{package_name} 未安装")
        except Exception as e:
            warnings.append(f"检查 {package_name} 时出错: {str(e)}")
    
    # 输出结果
    if incompatible:
        logger.error("发现不兼容的依赖:")
        for item in incompatible:
            logger.error(f"  - {item}")
        
        if '--ignore-dependencies' not in sys.argv:
            logger.error("请安装兼容的依赖版本，或使用 --ignore-dependencies 参数忽略此检查")
            sys.exit(1)
        else:
            logger.warning("忽略依赖检查，继续运行...")
    
    if warnings:
        logger.warning("发现潜在问题:")
        for item in warnings:
            logger.warning(f"  - {item}")
    
    if not incompatible and not warnings:
        logger.info("所有依赖版本兼容")
    
    return not incompatible

if __name__ == "__main__":
    check_dependencies()
