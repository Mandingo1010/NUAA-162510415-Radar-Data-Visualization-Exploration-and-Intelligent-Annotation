#!/usr/bin/env python3
"""
Web端雷达标注系统启动脚本
Web-based Radar Annotation System Startup Script
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查Web系统依赖...")

    required_packages = [
        'flask',
        'werkzeug',
        'opencv-python',
        'pillow',
        'numpy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'werkzeug':
                __import__('werkzeug')
            else:
                __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")

    if missing_packages:
        print(f"\n❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install flask opencv-python pillow numpy")
        return False

    print("\n✅ 所有依赖项检查通过!")
    return True

def setup_environment():
    """设置环境"""
    print("🛠️  设置Web环境...")

    # 确保目录存在
    directories = [
        'web_interface/templates',
        'web_interface/static/js',
        'web_interface/static/css',
        'web_interface/static/images',
        'uploads',
        'temp'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")

    print("✅ 环境设置完成!")

def start_flask_app():
    """启动Flask应用"""
    print("🚀 启动Web应用...")
    print("=" * 60)
    print("雷达数据智能标注系统 - Web版")
    print("Radar Data Intelligent Annotation System - Web Edition")
    print("=" * 60)

    # 切换到web_interface目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    try:
        # 启动Flask应用
        subprocess.run([sys.executable, "app.py"], check=True)

    except KeyboardInterrupt:
        print("\n\n🛑 用户停止应用")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 应用启动失败: {e}")
        print("\n故障排除建议:")
        print("1. 检查端口5000是否被占用")
        print("2. 确保所有依赖包已正确安装")
        print("3. 检查Python版本是否兼容")
        return False
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        return False

    return True

def open_browser():
    """打开浏览器"""
    print("🌐 正在打开浏览器...")
    time.sleep(2)  # 等待服务器启动

    try:
        webbrowser.open('http://localhost:5000')
        print("✅ 浏览器已打开: http://localhost:5000")
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
        print("请手动访问: http://localhost:5000")

def main():
    """主函数"""
    print("🎯 雷达标注系统Web版启动器")
    print("=" * 40)

    # 检查依赖
    if not check_dependencies():
        input("\n按回车键退出...")
        return

    # 设置环境
    setup_environment()

    print("\n📝 使用说明:")
    print("1. 启动后访问 http://localhost:5000")
    print("2. 拖拽文件夹到上传区域")
    print("3. 系统会自动分类文件")
    print("4. 点击'开始处理'进行智能标注")
    print("5. 在右侧面板进行人工审核")
    print("6. 导出审核结果")

    print("\n⚠️  注意事项:")
    print("- 支持拖拽文件夹上传")
    print("- 自动识别雷达数据、图像、配置文件")
    print("- 按 Ctrl+C 停止服务器")

    # 询问是否启动
    response = input("\n是否现在启动Web应用? (y/n): ").lower()
    if response not in ['y', 'yes', '是']:
        print("启动已取消")
        return

    # 在新线程中打开浏览器
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # 启动Flask应用
    success = start_flask_app()

    if success:
        print("\n✅ 应用已安全关闭")
    else:
        print("\n❌ 应用启动失败")

    input("按回车键退出...")

if __name__ == "__main__":
    main()