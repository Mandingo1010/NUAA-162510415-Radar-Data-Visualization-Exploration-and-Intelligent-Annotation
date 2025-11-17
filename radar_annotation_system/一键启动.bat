@echo off
chcp 65001 >nul
title 雷达标注系统 - 一键启动

echo.
echo =====================================
echo    雷达数据可视化与智能标注系统
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
echo 📦 正在检查依赖包...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo ⚠️  依赖包未安装，正在自动安装...
    echo 这可能需要10-20分钟，请耐心等待...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖包安装失败！
        echo 请检查网络连接或手动安装
        pause
        exit /b 1
    )
    echo ✅ 依赖包安装完成！
) else (
    echo ✅ 依赖包已安装
)

echo.
echo 🚀 正在启动系统...
echo.
echo 浏览器将自动打开，请稍等...
echo 如果浏览器没有自动打开，请手动访问：http://localhost:8501
echo.
echo ⚠️  关闭此窗口将停止系统运行
echo.

REM 启动Streamlit应用
python -m streamlit run main.py --server.port 8501 --server.headless false

echo.
echo 系统已停止运行
pause