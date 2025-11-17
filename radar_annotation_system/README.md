# 雷达数据可视化与智能标注系统
# Radar Data Visualization and Intelligent Annotation System

## 项目结构
```
radar_annotation_system/
├── src/                     # 源代码
│   ├── data_processing/     # 数据处理模块
│   ├── visualization/       # 可视化模块
│   ├── annotation/         # 标注模块
│   └── utils/              # 工具函数
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── sample/            # 示例数据
├── models/                # 模型文件
│   ├── pretrained/        # 预训练模型
│   └── custom/           # 自定义模型
├── configs/              # 配置文件
├── tests/                # 测试代码
├── docs/                 # 文档
└── outputs/              # 输出结果
```

## 依赖安装
```bash
pip install -r requirements.txt
```

## 快速开始
```bash
python main.py
```