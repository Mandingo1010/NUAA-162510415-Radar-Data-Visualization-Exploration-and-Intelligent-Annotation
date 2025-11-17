#!/usr/bin/env python3
"""
启动脚本
Startup Script for Radar Annotation System
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_dependencies():
    """检查依赖项"""
    print("检查系统依赖...")

    required_packages = [
        'numpy',
        'pandas',
        'opencv-python',
        'Pillow',
        'open3d',
        'matplotlib',
        'plotly',
        'streamlit',
        'torch',
        'ultralytics',
        'scipy',
        'scikit-learn',
        'pyyaml'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")

    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    print("\n所有依赖项检查通过!")
    return True


def create_directories():
    """创建必要的目录"""
    directories = [
        'data/raw',
        'data/processed',
        'data/sample',
        'models/pretrained',
        'models/custom',
        'outputs',
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")


def download_models():
    """下载预训练模型"""
    print("\n检查预训练模型...")

    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)

    # YOLO模型会在首次使用时自动下载
    print("✓ YOLO模型将在首次使用时自动下载")

    # 检查是否存在其他模型文件
    sam_model = models_dir / "sam_vit_h_4b8939.pth"
    if sam_model.exists():
        print("✓ SAM模型已存在")
    else:
        print("ℹ SAM模型未找到，如需使用请手动下载")

    print("模型检查完成")


def setup_config():
    """设置配置文件"""
    config_path = Path("configs/default_config.json")

    if not config_path.exists():
        print("\n创建默认配置文件...")
        sys.path.append('src')
        from utils.config_manager import ConfigManager

        config_manager = ConfigManager()
        config_manager.create_default_config_file(str(config_path))
        print("✓ 默认配置文件已创建")
    else:
        print("✓ 配置文件已存在")


def run_streamlit_app():
    """启动Streamlit应用"""
    print("\n启动雷达标注系统...")
    print("浏览器将自动打开应用界面")
    print("如未自动打开，请访问: http://localhost:8501")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动应用失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n应用已停止")


def run_demo():
    """运行演示"""
    print("\n运行演示程序...")
    sys.path.append('src')

    try:
        # 导入并运行数据处理演示
        from data_processing.radar_processor import main as radar_demo
        from data_processing.image_processor import main as image_demo
        from visualization.point_cloud_viz import main as viz_demo
        from annotation.object_detector import main as detection_demo
        from annotation.radar_image_fusion import main as fusion_demo

        print("1. 运行雷达数据处理演示...")
        radar_demo()

        print("\n2. 运行图像处理演示...")
        image_demo()

        print("\n3. 运行可视化演示...")
        viz_demo()

        print("\n4. 运行目标检测演示...")
        detection_demo()

        print("\n5. 运行融合标注演示...")
        fusion_demo()

        print("\n演示程序运行完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="雷达数据可视化与智能标注系统")
    parser.add_argument(
        '--mode',
        choices=['app', 'demo', 'check', 'setup'],
        default='app',
        help='运行模式'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='跳过依赖检查'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("雷达数据可视化与智能标注系统")
    print("Radar Data Visualization and Intelligent Annotation System")
    print("=" * 60)

    # 切换到项目目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    if args.mode == 'check':
        # 检查模式
        success = check_dependencies()
        sys.exit(0 if success else 1)

    elif args.mode == 'setup':
        # 设置模式
        print("初始化项目环境...")

        if not args.skip_deps:
            if not check_dependencies():
                print("依赖检查失败，请先安装必要的包")
                sys.exit(1)

        create_directories()
        download_models()
        setup_config()

        print("\n项目设置完成!")
        print("运行 'python run.py --mode app' 启动应用")

    elif args.mode == 'demo':
        # 演示模式
        if not args.skip_deps:
            if not check_dependencies():
                print("依赖检查失败，请先安装必要的包")
                sys.exit(1)

        create_directories()
        setup_config()
        run_demo()

    else:
        # 应用模式 (默认)
        if not args.skip_deps:
            if not check_dependencies():
                print("依赖检查失败，请先安装必要的包")
                print("或运行 'python run.py --skip-deps' 跳过检查")
                sys.exit(1)

        create_directories()
        setup_config()
        run_streamlit_app()


if __name__ == "__main__":
    main()