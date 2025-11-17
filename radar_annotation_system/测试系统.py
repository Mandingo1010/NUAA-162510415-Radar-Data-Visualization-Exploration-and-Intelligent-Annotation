#!/usr/bin/env python3
"""
系统测试脚本 - 验证各模块是否正常工作
"""

import sys
import os
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """测试导入"""
    print("🔍 测试模块导入...")

    try:
        from data_processing.radar_processor import RadarProcessor
        print("✅ 雷达处理模块导入成功")
    except Exception as e:
        print(f"❌ 雷达处理模块导入失败: {e}")
        return False

    try:
        from data_processing.image_processor import ImageProcessor
        print("✅ 图像处理模块导入成功")
    except Exception as e:
        print(f"❌ 图像处理模块导入失败: {e}")
        return False

    try:
        from visualization.point_cloud_viz import PointCloudVisualizer
        print("✅ 点云可视化模块导入成功")
    except Exception as e:
        print(f"❌ 点云可视化模块导入失败: {e}")
        return False

    try:
        from annotation.object_detector import ObjectDetectionPipeline
        print("✅ 目标检测模块导入成功")
    except Exception as e:
        print(f"❌ 目标检测模块导入失败: {e}")
        return False

    try:
        from annotation.radar_image_fusion import RadarImageFusion
        print("✅ 融合标注模块导入成功")
    except Exception as e:
        print(f"❌ 融合标注模块导入失败: {e}")
        return False

    return True

def test_radar_processing():
    """测试雷达数据处理"""
    print("\n🎯 测试雷达数据处理...")

    try:
        # 创建雷达处理器
        config = {
            'noise_threshold': 0.1,
            'dbscan_eps': 0.5,
            'dbscan_min_samples': 5
        }
        processor = RadarProcessor(config)

        # 生成测试数据
        np.random.seed(42)
        test_data = np.random.rand(50, 4)
        test_data[:, 0] *= 30  # range: 0-30m
        test_data[:, 1] *= 2 * np.pi  # azimuth
        test_data[:, 2] *= 5  # doppler
        test_data[:, 3] = test_data[:, 3] * 0.8 + 0.2  # intensity

        # 处理数据
        radar_points = processor.process_frame(test_data)
        point_cloud = processor.create_point_cloud(radar_points)

        print(f"✅ 雷达数据处理成功！")
        print(f"   输入数据点数: {len(test_data)}")
        print(f"   输出点云点数: {len(point_cloud)}")

        return True

    except Exception as e:
        print(f"❌ 雷达数据处理失败: {e}")
        return False

def test_image_processing():
    """测试图像数据处理"""
    print("\n🖼️  测试图像数据处理...")

    try:
        # 创建图像处理器
        config = {
            'image_size': (320, 240),
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
        processor = ImageProcessor(config)

        # 生成测试图像
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        # 处理图像
        resized_image = processor.resize_image(test_image, (160, 120))
        normalized_image = processor.normalize_image(test_image)
        features = processor.extract_features(test_image)

        print(f"✅ 图像数据处理成功！")
        print(f"   原始图像尺寸: {test_image.shape}")
        print(f"   调整后尺寸: {resized_image.shape}")
        print(f"   特征数量: {len(features)}")

        return True

    except Exception as e:
        print(f"❌ 图像数据处理失败: {e}")
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n📊 测试可视化功能...")

    try:
        # 创建可视化器
        config = {
            'window_size': (640, 480),
            'background_color': [0.1, 0.1, 0.1],
            'point_size': 2.0
        }
        visualizer = PointCloudVisualizer(config)

        # 生成测试点云
        np.random.seed(42)
        test_points = np.random.randn(20, 3) * 5
        test_colors = np.random.rand(20)

        print("✅ 可视化模块创建成功！")
        print(f"   测试点云点数: {len(test_points)}")
        print("   注意：实际的可视化窗口不会在测试中打开")

        # 清理
        visualizer.close()

        return True

    except Exception as e:
        print(f"❌ 可视化功能测试失败: {e}")
        return False

def test_fusion():
    """测试融合功能"""
    print("\n🔗 测试融合功能...")

    try:
        # 创建融合器
        config = {
            'camera_matrix': [
                [500, 0, 160],
                [0, 500, 120],
                [0, 0, 1]
            ],
            'extrinsic_matrix': [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]
            ],
            'distance_threshold': 2.0,
            'angular_threshold': 0.1,
            'confidence_weight': 0.7
        }
        fusion = RadarImageFusion(config)

        # 生成测试数据
        radar_points = np.random.randn(10, 3) * 5
        radar_points[:, 2] = np.abs(radar_points[:, 2]) + 5

        image_detections = [
            {
                'bbox': [50, 50, 40, 60],
                'center': [70, 80],
                'confidence': 0.8,
                'class_name': 'test_object'
            }
        ]

        # 测试融合
        fused_detections = fusion.generate_fused_detections(radar_points, image_detections)

        print(f"✅ 融合功能测试成功！")
        print(f"   雷达点数: {len(radar_points)}")
        print(f"   图像检测数: {len(image_detections)}")
        print(f"   融合结果数: {len(fused_detections)}")

        return True

    except Exception as e:
        print(f"❌ 融合功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 雷达标注系统 - 功能测试")
    print("=" * 60)

    all_tests_passed = True

    # 运行所有测试
    tests = [
        test_imports,
        test_radar_processing,
        test_image_processing,
        test_visualization,
        test_fusion
    ]

    for test in tests:
        if not test():
            all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 所有测试通过！系统运行正常！")
        print("\n你可以安全地启动主应用了：")
        print("1. 双击'一键启动.bat'")
        print("2. 或者运行：python run.py --mode app")
    else:
        print("❌ 部分测试失败！")
        print("\n请检查：")
        print("1. 是否正确安装了所有依赖包")
        print("2. Python版本是否为3.8+")
        print("3. 运行：pip install -r requirements.txt")

    print("=" * 60)

    if not all_tests_passed:
        input("\n按回车键退出...")
    else:
        input("\n按回车键启动主应用...")
        try:
            import subprocess
            subprocess.run([sys.executable, "run.py", "--mode", "app"], check=True)
        except:
            print("启动失败，请手动运行：python run.py --mode app")

if __name__ == "__main__":
    main()