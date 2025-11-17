@echo off
chcp 65001 >nul
title 雷达标注系统 - Web版启动器

echo.
echo =====================================
echo    雷达数据智能标注系统 - Web版
echo           一键启动工具
echo =====================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未检测到Python！
    echo.
    echo 请先安装Python：
    echo 1. 访问 https://www.python.org/downloads/
    echo 2. 下载并安装Python 3.8+
    echo 3. 安装时务必勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo ✅ Python已安装
python --version

echo.
echo 📦 正在检查Web版依赖包...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Flask未安装，正在自动安装...
    echo 这可能需要几分钟时间，请耐心等待...
    echo.
    pip install flask opencv-python pillow numpy werkzeug
    if errorlevel 1 (
        echo ❌ 依赖包安装失败！
        echo 请检查网络连接或手动安装：
        echo pip install flask opencv-python pillow numpy werkzeug
        pause
        exit /b 1
    )
    echo ✅ 依赖包安装完成！
) else (
    echo ✅ Web版依赖包已安装
)

echo.
echo 🚀 正在启动Web版系统...
echo.
echo ⚠️  重要提示：
echo 1. 浏览器将自动打开 http://localhost:5000
echo 2. 如果没有自动打开，请手动访问该地址
echo 3. 支持拖拽文件夹上传
echo 4. 可以自动识别雷达数据、图像、配置文件
echo 5. 关闭此窗口将停止Web服务
echo.

REM 进入web_interface目录并启动Flask应用
cd /d "%~dp0web_interface"
if exist "start_web.py" (
    echo 🌐 启动Web界面...
    python start_web.py
) else (
    echo 🌐 直接启动Flask应用...
    python app.py
)

echo.
echo Web服务已停止
pause