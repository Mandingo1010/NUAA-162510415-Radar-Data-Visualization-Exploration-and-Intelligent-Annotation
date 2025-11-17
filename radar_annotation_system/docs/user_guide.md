# 用户使用指南

## 系统概述

雷达数据可视化与智能标注系统是一个集成了毫米波雷达数据处理、三维可视化、多模态融合和智能标注功能的综合性平台。

## 系统要求

### 硬件要求
- CPU: Intel i5 或同等性能以上
- 内存: 8GB RAM (推荐16GB)
- 显卡: 支持OpenGL 3.3+的显卡 (用于3D可视化)
- 存储: 至少5GB可用空间

### 软件要求
- Python 3.8+
- 操作系统: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

## 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd radar_annotation_system
```

### 2. 创建虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 初始化项目
```bash
python run.py --mode setup
```

## 快速开始

### 启动应用
```bash
python run.py --mode app
```

应用将在浏览器中打开: http://localhost:8501

### 运行演示
```bash
python run.py --mode demo
```

## 功能模块

### 1. 数据处理

#### 雷达数据处理
- 支持多种数据格式 (CSV, NPY, JSON)
- 自动噪声过滤和点云生成
- DBSCAN聚类去噪

#### 图像数据处理
- 支持常见图像格式 (JPG, PNG)
- 图像预处理和特征提取
- 相机畸变校正

### 2. 点云可视化

#### 可视化方式
- **Matplotlib 2D**: 基础2D投影可视化
- **Plotly 3D**: 交互式3D可视化
- **Open3D**: 高性能3D渲染

#### 可视化选项
- 点大小调节
- 多种颜色方案 (单色、强度、高度、随机)
- 边界框显示

### 3. 多模态融合

#### 融合功能
- 雷达点云投影到图像平面
- 图像检测结果映射到3D空间
- 同屏多视角展示

#### 参数配置
- 相机内参矩阵设置
- 外参矩阵标定
- 投影参数调节

### 4. 智能标注

#### 检测器支持
- **YOLO**: 实时目标检测
- **SAM**: 实例分割 (可选)
- **GPT-4V**: 语义理解 (可选)

#### 融合标注
- 自动关联雷达点和图像检测
- 3D边界框生成
- 半自动标注流程

### 5. 结果导出

#### 支持格式
- **JSON**: 结构化数据导出
- **CSV**: 表格数据导出
- **HDF5**: 高性能数据存储
- **PLY**: 点云格式导出

## 使用流程

### 标准工作流程

1. **数据准备**
   - 上传雷达数据文件
   - 上载对应图像文件
   - 系统自动进行预处理

2. **可视化探索**
   - 选择合适的可视化方法
   - 调整可视化参数
   - 初步了解数据分布

3. **多模态融合**
   - 配置相机参数
   - 生成融合可视化
   - 验证数据对齐效果

4. **智能标注**
   - 启用目标检测器
   - 执行检测任务
   - 进行融合标注

5. **结果导出**
   - 选择导出格式
   - 下载标注结果
   - 保存项目配置

### 高级功能

#### 自定义配置
```json
{
  "radar": {
    "noise_threshold": 0.1,
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 5
  },
  "detection": {
    "yolo": {
      "confidence_threshold": 0.7
    }
  }
}
```

#### 批量处理
- 支持多帧数据处理
- 自动化标注流程
- 批量结果导出

## 常见问题

### Q: 如何处理不同格式的雷达数据？
A: 系统支持CSV、NPY和JSON格式。确保数据包含range、azimuth、doppler和intensity字段。

### Q: 相机参数如何获取？
A: 需要进行相机标定。可以使用OpenCV的标定工具或标定板进行标定。

### Q: 检测效果不理想怎么办？
A: 可以调整置信度阈值、使用更大型的YOLO模型、或尝试其他检测器。

### Q: 如何提高标注精度？
A: 优化相机标定参数、调整融合阈值、进行人工校正。

### Q: 系统运行缓慢如何优化？
A: 减少点云密度、关闭不必要的可视化选项、使用GPU加速。

## 技术支持

### 日志文件
系统日志保存在 `logs/` 目录下，包含：
- 错误日志
- 性能统计
- 操作记录

### 配置文件
默认配置文件位于 `configs/default_config.json`，可以复制并自定义。

### 模型文件
预训练模型自动下载到 `models/pretrained/` 目录。

## 扩展开发

### 添加新的检测器
1. 在 `src/annotation/` 目录下创建新的检测器类
2. 实现 `detect` 方法
3. 在 `ObjectDetectionPipeline` 中注册

### 自定义可视化
1. 继承 `PointCloudVisualizer` 或 `MultimodalVisualizer`
2. 实现自定义可视化方法
3. 在主界面中添加选项

### 数据格式扩展
1. 在 `data_processing/` 中添加新的解析器
2. 实现标准化的数据接口
3. 更新配置文件格式

## 性能优化建议

1. **硬件优化**
   - 使用SSD存储
   - 增加内存容量
   - 使用独立GPU

2. **软件优化**
   - 安装CUDA加速库
   - 调整点云密度
   - 优化可视化参数

3. **数据优化**
   - 预处理和缓存数据
   - 使用适当的数据格式
   - 定期清理临时文件

## 版本更新

### 更新方法
```bash
git pull origin main
pip install -r requirements.txt --upgrade
python run.py --mode setup
```

### 版本兼容性
- 配置文件自动迁移
- 数据格式向后兼容
- 模型文件自动更新

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。