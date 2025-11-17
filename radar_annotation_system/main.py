"""
主程序入口
Main Application Entry Point
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import os
import sys
from typing import Dict, List, Optional

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.radar_processor import RadarProcessor
from data_processing.image_processor import ImageProcessor
from visualization.point_cloud_viz import PointCloudVisualizer
from visualization.multimodal_viz import MultimodalVisualizer
from annotation.object_detector import ObjectDetectionPipeline
from annotation.radar_image_fusion import RadarImageFusion
from utils.config_manager import ConfigManager


def main():
    """主函数"""
    st.set_page_config(
        page_title="雷达数据可视化与智能标注系统",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📡 雷达数据可视化与智能标注系统")
    st.markdown("---")

    # 侧边栏配置
    with st.sidebar:
        st.header("系统配置")

        # 加载配置
        config_file = st.file_uploader("上传配置文件", type=['json'])
        if config_file:
            config = json.load(config_file)
            st.success("配置文件已加载")
        else:
            # 默认配置
            config = load_default_config()
            st.info("使用默认配置")

        # 数据源选择
        st.subheader("数据源")
        data_source = st.selectbox(
            "选择数据源",
            ["示例数据", "上传数据", "实时数据"]
        )

    # 主界面选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "数据处理", "点云可视化", "多模态融合", "智能标注", "结果导出"
    ])

    with tab1:
        data_processing_interface(config, data_source)

    with tab2:
        point_cloud_visualization_interface(config)

    with tab3:
        multimodal_fusion_interface(config)

    with tab4:
        intelligent_annotation_interface(config)

    with tab5:
        export_interface(config)


def load_default_config() -> Dict:
    """加载默认配置"""
    return {
        "radar": {
            "noise_threshold": 0.1,
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 5
        },
        "image": {
            "image_size": [640, 480],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225]
        },
        "camera": {
            "matrix": [
                [500, 0, 320],
                [0, 500, 240],
                [0, 0, 1]
            ],
            "extrinsic": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]
            ]
        },
        "detection": {
            "yolo": {
                "enabled": True,
                "model_path": "yolov8n.pt",
                "confidence_threshold": 0.5
            },
            "sam": {
                "enabled": False
            },
            "gpt4v": {
                "enabled": False
            }
        },
        "fusion": {
            "distance_threshold": 2.0,
            "angular_threshold": 0.1,
            "confidence_weight": 0.7
        }
    }


def data_processing_interface(config: Dict, data_source: str):
    """数据处理界面"""
    st.header("🔧 数据处理")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("雷达数据")

        if data_source == "上传数据":
            radar_file = st.file_uploader("上传雷达数据", type=['csv', 'npy', 'json'])
            if radar_file:
                # 处理雷达数据
                radar_processor = RadarProcessor(config['radar'])

                try:
                    # 根据文件类型加载数据
                    if radar_file.name.endswith('.csv'):
                        import pandas as pd
                        raw_data = pd.read_csv(radar_file).values
                    elif radar_file.name.endswith('.npy'):
                        raw_data = np.load(radar_file)
                    else:
                        raw_data = np.array(json.load(radar_file))

                    # 处理数据
                    radar_points = radar_processor.process_frame(raw_data)
                    point_cloud = radar_processor.create_point_cloud(radar_points)

                    st.session_state['radar_data'] = point_cloud
                    st.success(f"成功处理 {len(point_cloud)} 个雷达点")

                    # 显示统计信息
                    st.json({
                        "点数": len(point_cloud),
                        "数据形状": point_cloud.shape,
                        "处理配置": config['radar']
                    })

                except Exception as e:
                    st.error(f"处理雷达数据时出错: {e}")

        elif data_source == "示例数据":
            if st.button("生成示例雷达数据"):
                # 生成示例数据
                radar_processor = RadarProcessor(config['radar'])

                np.random.seed(42)
                test_data = np.random.rand(100, 4)
                test_data[:, 0] *= 50  # range: 0-50m
                test_data[:, 1] *= 2 * np.pi  # azimuth
                test_data[:, 2] *= 10  # doppler
                test_data[:, 3] = test_data[:, 3] * 0.8 + 0.2  # intensity

                radar_points = radar_processor.process_frame(test_data)
                point_cloud = radar_processor.create_point_cloud(radar_points)

                st.session_state['radar_data'] = point_cloud
                st.success(f"生成 {len(point_cloud)} 个示例雷达点")

    with col2:
        st.subheader("图像数据")

        if data_source == "上传数据":
            image_file = st.file_uploader("上传图像", type=['jpg', 'jpeg', 'png'])
            if image_file:
                image = Image.open(image_file)
                st.image(image, caption="原始图像", use_column_width=True)

                # 处理图像
                image_processor = ImageProcessor(config['image'])
                image_array = np.array(image)

                st.session_state['image_data'] = image_array
                st.success("图像加载成功")

                # 显示图像信息
                st.json({
                    "图像尺寸": image_array.shape,
                    "数据类型": str(image_array.dtype),
                    "数值范围": [float(image_array.min()), float(image_array.max())]
                })

        elif data_source == "示例数据":
            if st.button("生成示例图像"):
                # 生成示例图像
                image_processor = ImageProcessor(config['image'])

                # 创建彩色噪声图像
                sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # 添加一些几何形状
                cv2.rectangle(sample_image, (100, 100), (200, 200), (255, 0, 0), -1)
                cv2.circle(sample_image, (400, 300), 50, (0, 255, 0), -1)

                st.session_state['image_data'] = sample_image
                st.image(sample_image, caption="示例图像", use_column_width=True)
                st.success("示例图像生成成功")


def point_cloud_visualization_interface(config: Dict):
    """点云可视化界面"""
    st.header("🌐 点云可视化")

    if 'radar_data' not in st.session_state:
        st.warning("请先在数据处理页面加载雷达数据")
        return

    radar_data = st.session_state['radar_data']
    st.write(f"当前点云包含 {len(radar_data)} 个点")

    # 可视化选项
    viz_method = st.selectbox(
        "选择可视化方法",
        ["Matplotlib 2D", "Plotly 3D", "Open3D 交互式"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("可视化设置")

        # 点云设置
        point_size = st.slider("点大小", 1, 10, 2)
        color_scheme = st.selectbox(
            "颜色方案",
            ["单色", "强度", "高度", "随机"]
        )

        # 生成颜色
        if color_scheme == "单色":
            colors = None
        elif color_scheme == "强度" and radar_data.shape[1] > 3:
            colors = radar_data[:, 3]  # 使用强度值
        elif color_scheme == "高度":
            colors = radar_data[:, 2]  # 使用Z坐标
        else:
            colors = np.random.rand(len(radar_data))

    with col2:
        st.subheader("可视化结果")

        try:
            if viz_method == "Matplotlib 2D":
                visualizer = PointCloudVisualizer(config.get('visualization', {}))
                fig = visualizer.plot_with_matplotlib(
                    radar_data[:, :3], colors,
                    "雷达点云可视化", save_path="temp_plot.png"
                )
                st.pyplot(fig)

            elif viz_method == "Plotly 3D":
                visualizer = PointCloudVisualizer(config.get('visualization', {}))
                visualizer.plot_with_plotly(radar_data[:, :3], colors, "交互式雷达点云")
                st.success("Plotly图表已在新窗口中打开")

            elif viz_method == "Open3D 交互式":
                if st.button("启动Open3D可视化"):
                    visualizer = PointCloudVisualizer(config.get('visualization', {}))
                    visualizer.visualize_single_frame(radar_data[:, :3], colors)
                    st.success("Open3D窗口已打开，请关闭窗口继续")

        except Exception as e:
            st.error(f"可视化时出错: {e}")


def multimodal_fusion_interface(config: Dict):
    """多模态融合界面"""
    st.header("🔗 多模态融合")

    if 'radar_data' not in st.session_state or 'image_data' not in st.session_state:
        st.warning("请先在数据处理页面加载雷达数据和图像数据")
        return

    radar_data = st.session_state['radar_data']
    image_data = st.session_state['image_data']

    # 相机参数设置
    with st.expander("相机参数设置"):
        camera_matrix = np.array(config['camera']['matrix'])
        extrinsic_matrix = np.array(config['camera']['extrinsic'])

        st.write("相机内参矩阵:")
        st.json(camera_matrix.tolist())

        st.write("外参矩阵:")
        st.json(extrinsic_matrix.tolist())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("融合设置")

        show_projection = st.checkbox("显示点云投影", True)
        show_bboxes = st.checkbox("显示边界框", False)

        if show_bboxes:
            # 添加示例边界框
            example_bboxes = [
                {
                    'bbox': [100, 100, 80, 120],
                    'label': '示例物体1',
                    'color': 'red'
                },
                {
                    'bbox': [300, 200, 60, 100],
                    'label': '示例物体2',
                    'color': 'blue'
                }
            ]
        else:
            example_bboxes = None

    with col2:
        st.subheader("融合可视化")

        if st.button("生成融合可视化"):
            try:
                multimodal_viz = MultimodalVisualizer(config.get('visualization', {}))

                # 生成融合可视化
                fig = multimodal_viz.visualize_fusion(
                    image_data,
                    radar_data[:, :3],
                    camera_matrix,
                    extrinsic_matrix,
                    example_bboxes if show_bboxes else None
                )

                st.pyplot(fig)
                st.success("融合可视化生成成功")

            except Exception as e:
                st.error(f"生成融合可视化时出错: {e}")


def intelligent_annotation_interface(config: Dict):
    """智能标注界面"""
    st.header("🤖 智能标注")

    if 'image_data' not in st.session_state:
        st.warning("请先在数据处理页面加载图像数据")
        return

    image_data = st.session_state['image_data']

    # 检测器配置
    with st.expander("检测器配置"):
        detector_enabled = {}
        for detector_name, detector_config in config['detection'].items():
            detector_enabled[detector_name] = st.checkbox(
                f"启用 {detector_name.upper()} 检测器",
                detector_config.get('enabled', False)
            )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("目标检测")

        if st.button("开始检测"):
            enabled_detectors = [name for name, enabled in detector_enabled.items() if enabled]

            if not enabled_detectors:
                st.warning("请至少启用一个检测器")
            else:
                try:
                    # 创建检测流水线
                    detection_pipeline = ObjectDetectionPipeline(config)

                    # 进行检测
                    detection_results = detection_pipeline.detect(image_data, enabled_detectors)

                    st.session_state['detection_results'] = detection_results

                    # 显示检测结果
                    for detector_name, detections in detection_results.items():
                        st.write(f"**{detector_name.upper()}** 检测到 {len(detections)} 个目标")

                        if detections:
                            # 显示检测结果表格
                            import pandas as pd
                            df_data = []
                            for det in detections:
                                df_data.append({
                                    '类别': det['class_name'],
                                    '置信度': f"{det['confidence']:.2f}",
                                    '边界框': f"({det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]})"
                                })

                            df = pd.DataFrame(df_data)
                            st.dataframe(df)

                except Exception as e:
                    st.error(f"检测过程中出错: {e}")

    with col2:
        st.subheader("雷达-图像融合")

        if 'radar_data' in st.session_state and 'detection_results' in st.session_state:
            if st.button("开始融合标注"):
                try:
                    radar_data = st.session_state['radar_data']
                    detection_results = st.session_state['detection_results']

                    # 合并所有检测结果
                    all_detections = []
                    for detections in detection_results.values():
                        all_detections.extend(detections)

                    # 创建融合器
                    fusion = RadarImageFusion(config['fusion'])

                    # 生成融合检测结果
                    fused_detections = fusion.generate_fused_detections(
                        radar_data[:, :3], all_detections
                    )

                    st.session_state['fused_detections'] = fused_detections

                    st.success(f"生成 {len(fused_detections)} 个融合标注")

                    # 显示融合结果
                    for i, detection in enumerate(fused_detections):
                        with st.expander(f"标注 {i+1}: {detection.class_name}"):
                            st.write(f"**3D中心点**: {detection.center_3d}")
                            st.write(f"**3D尺寸**: {detection.size_3d}")
                            st.write(f"**置信度**: {detection.confidence:.2f}")
                            st.write(f"**来源**: {detection.source}")

                except Exception as e:
                    st.error(f"融合过程中出错: {e}")

        else:
            st.info("请先完成目标检测")


def export_interface(config: Dict):
    """结果导出界面"""
    st.header("💾 结果导出")

    # 检查可用的结果
    available_results = []

    if 'radar_data' in st.session_state:
        available_results.append("雷达点云数据")
    if 'image_data' in st.session_state:
        available_results.append("处理后的图像")
    if 'detection_results' in st.session_state:
        available_results.append("目标检测结果")
    if 'fused_detections' in st.session_state:
        available_results.append("融合标注结果")

    if not available_results:
        st.warning("暂无可导出的结果，请先完成相应的处理步骤")
        return

    st.write(f"可导出的结果: {', '.join(available_results)}")

    # 导出选项
    export_format = st.selectbox(
        "选择导出格式",
        ["JSON", "CSV", "HDF5", "PLY (点云)"]
    )

    export_items = st.multiselect(
        "选择要导出的项目",
        available_results,
        default=available_results
    )

    if st.button("导出结果"):
        try:
            export_data = {}

            for item in export_items:
                if item == "雷达点云数据" and 'radar_data' in st.session_state:
                    export_data['radar_points'] = st.session_state['radar_data'].tolist()

                elif item == "目标检测结果" and 'detection_results' in st.session_state:
                    # 转换检测结果为可序列化格式
                    serializable_detections = {}
                    for detector_name, detections in st.session_state['detection_results'].items():
                        serializable_detections[detector_name] = []
                        for det in detections:
                            serializable_det = {
                                'bbox': det['bbox'],
                                'confidence': det['confidence'],
                                'class_name': det['class_name'],
                                'center': det['center']
                            }
                            serializable_detections[detector_name].append(serializable_det)
                    export_data['detection_results'] = serializable_detections

                elif item == "融合标注结果" and 'fused_detections' in st.session_state:
                    # 转换融合检测结果
                    export_data['fused_annotations'] = []
                    for detection in st.session_state['fused_detections']:
                        annotation = {
                            'center_3d': detection.center_3d.tolist(),
                            'size_3d': detection.size_3d.tolist(),
                            'confidence': detection.confidence,
                            'class_name': detection.class_name,
                            'source': detection.source
                        }
                        export_data['fused_annotations'].append(annotation)

            # 根据格式导出
            if export_format == "JSON":
                export_json(export_data)
            elif export_format == "CSV":
                export_csv(export_data)
            elif export_format == "HDF5":
                export_hdf5(export_data)
            elif export_format == "PLY (点云)":
                export_ply(export_data)

            st.success(f"结果已导出为 {export_format} 格式")

        except Exception as e:
            st.error(f"导出过程中出错: {e}")


def export_json(data: Dict):
    """导出JSON格式"""
    import json
    from datetime import datetime

    filename = f"export_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    st.download_button(
        label="下载JSON文件",
        data=json.dumps(data, indent=2, ensure_ascii=False),
        file_name=filename,
        mime="application/json"
    )


def export_csv(data: Dict):
    """导出CSV格式"""
    import pandas as pd
    from datetime import datetime
    import io

    filename = f"export_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # 创建CSV内容
    output = io.StringIO()

    for key, value in data.items():
        output.write(f"# {key}\n")

        if isinstance(value, list) and value:
            if isinstance(value[0], dict):
                # 字典列表
                df = pd.DataFrame(value)
                df.to_csv(output, index=False)
            else:
                # 数值列表
                df = pd.DataFrame(value)
                df.to_csv(output, index=False, header=False)

        output.write("\n")

    csv_content = output.getvalue()

    st.download_button(
        label="下载CSV文件",
        data=csv_content,
        file_name=filename,
        mime="text/csv"
    )


def export_hdf5(data: Dict):
    """导出HDF5格式"""
    import h5py
    from datetime import datetime
    import tempfile
    import os

    filename = f"export_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
        with h5py.File(tmp_file.name, 'w') as f:
            for key, value in data.items():
                if isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        # 字典列表 - 转换为结构化数组
                        # 这里简化处理，实际需要更复杂的转换
                        f.create_dataset(key, data=str(value))
                    else:
                        # 数值列表
                        f.create_dataset(key, data=value)
                else:
                    f.create_dataset(key, data=str(value))

        # 读取文件内容
        with open(tmp_file.name, 'rb') as f:
            file_content = f.read()

        os.unlink(tmp_file.name)

    st.download_button(
        label="下载HDF5文件",
        data=file_content,
        file_name=filename,
        mime="application/octet-stream"
    )


def export_ply(data: Dict):
    """导出PLY点云格式"""
    from datetime import datetime
    import tempfile
    import os

    if 'radar_points' not in data:
        st.warning("没有雷达点云数据可导出")
        return

    filename = f"point_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"

    points = np.array(data['radar_points'])

    # 创建PLY内容
    ply_content = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property float intensity
end_header
"""

    for point in points:
        if len(point) >= 4:
            ply_content += f"{point[0]} {point[1]} {point[2]} {point[3]}\n"
        else:
            ply_content += f"{point[0]} {point[1]} {point[2]} 1.0\n"

    st.download_button(
        label="下载PLY文件",
        data=ply_content,
        file_name=filename,
        mime="text/plain"
    )


if __name__ == "__main__":
    main()