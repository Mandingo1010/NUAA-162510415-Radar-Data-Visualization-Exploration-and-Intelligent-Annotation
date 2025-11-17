"""
多模态可视化模块
Multimodal Visualization Module
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import open3d as o3d


class MultimodalVisualizer:
    """多模态可视化器"""

    def __init__(self, config: Dict):
        """
        初始化多模态可视化器

        Args:
            config: 配置参数字典
        """
        self.config = config
        self.image_size = config.get('image_size', (640, 480))
        self.point_size = config.get('point_size', 2)

    def project_points_to_image(self, points_3d: np.ndarray,
                               camera_matrix: np.ndarray,
                               extrinsic_matrix: np.ndarray) -> np.ndarray:
        """
        将3D点投影到图像平面

        Args:
            points_3d: 3D点云 [N, 3]
            camera_matrix: 相机内参矩阵 [3, 3]
            extrinsic_matrix: 外参矩阵 [3, 4] 或 [4, 4]

        Returns:
            2D投影点 [N, 2] 和深度信息
        """
        if len(points_3d) == 0:
            return np.array([])

        # 齐次坐标
        points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])

        # 外参变换 (世界坐标系到相机坐标系)
        if extrinsic_matrix.shape == (3, 4):
            points_camera = extrinsic_matrix @ points_homo.T
        else:  # 4x4 matrix
            points_camera = extrinsic_matrix @ points_homo.T

        # 去除齐次坐标
        points_camera = points_camera[:3, :].T

        # 过滤掉相机后面的点
        valid_mask = points_camera[:, 2] > 0
        points_camera = points_camera[valid_mask]

        if len(points_camera) == 0:
            return np.array([])

        # 内参投影到图像平面
        points_image_homo = camera_matrix @ points_camera.T
        points_image = points_image_homo[:2, :].T / points_image_homo[2, :]

        return points_image

    def visualize_fusion(self, image: np.ndarray, points_3d: np.ndarray,
                        camera_matrix: np.ndarray, extrinsic_matrix: np.ndarray,
                        bounding_boxes_2d: Optional[List[Dict]] = None,
                        bounding_boxes_3d: Optional[List[Dict]] = None) -> plt.Figure:
        """
        可视化图像-雷达融合

        Args:
            image: RGB图像
            points_3d: 3D点云
            camera_matrix: 相机内参
            extrinsic_matrix: 外参矩阵
            bounding_boxes_2d: 2D边界框列表
            bounding_boxes_3d: 3D边界框列表

        Returns:
            Matplotlib图形对象
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 绘制2D边界框
        if bounding_boxes_2d:
            for bbox in bounding_boxes_2d:
                x, y, w, h = bbox['bbox']
                rect = Rectangle((x, y), w, h, linewidth=2,
                               edgecolor=bbox.get('color', 'red'),
                               facecolor='none')
                axes[0].add_patch(rect)
                if 'label' in bbox:
                    axes[0].text(x, y-5, bbox['label'], color='red', fontsize=8)

        # 2. 点云投影到图像
        axes[1].imshow(image)
        axes[1].set_title('Radar Points Projection')
        axes[1].axis('off')

        # 投影3D点到图像
        projected_points = self.project_points_to_image(points_3d, camera_matrix, extrinsic_matrix)
        if len(projected_points) > 0:
            # 过滤在图像范围内的点
            h, w = image.shape[:2]
            valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) & \
                        (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
            valid_points = projected_points[valid_mask]

            if len(valid_points) > 0:
                axes[1].scatter(valid_points[:, 0], valid_points[:, 1],
                              c='red', s=self.point_size, alpha=0.7)

        # 3. 融合显示
        axes[2].imshow(image)
        axes[2].set_title('Fused Visualization')
        axes[2].axis('off')

        # 显示投影点
        if len(projected_points) > 0:
            h, w = image.shape[:2]
            valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) & \
                        (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
            valid_points = projected_points[valid_mask]

            if len(valid_points) > 0:
                axes[2].scatter(valid_points[:, 0], valid_points[:, 1],
                              c='lime', s=self.point_size*2, alpha=0.8)

        # 绘制融合后的边界框
        if bounding_boxes_2d:
            for bbox in bounding_boxes_2d:
                x, y, w, h = bbox['bbox']
                rect = Rectangle((x, y), w, h, linewidth=2,
                               edgecolor=bbox.get('color', 'blue'),
                               facecolor='none', linestyle='--')
                axes[2].add_patch(rect)
                if 'label' in bbox:
                    axes[2].text(x, y-5, f"[Fusion] {bbox['label']}",
                                color='blue', fontsize=8, weight='bold')

        plt.tight_layout()
        return fig

    def create_plotly_fusion(self, image: np.ndarray, points_3d: np.ndarray,
                           camera_matrix: np.ndarray, extrinsic_matrix: np.ndarray,
                           bounding_boxes_2d: Optional[List[Dict]] = None) -> go.Figure:
        """
        创建Plotly交互式融合可视化

        Args:
            image: RGB图像
            points_3d: 3D点云
            camera_matrix: 相机内参
            extrinsic_matrix: 外参矩阵
            bounding_boxes_2d: 2D边界框列表

        Returns:
            Plotly图形对象
        """
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Image with Detection', 'Radar-Image Fusion'),
            specs=[[{"type": "xy"}, {"type": "xy"}]]
        )

        # 原始图像
        fig.add_trace(
            go.Image(z=image),
            row=1, col=1
        )

        # 添加2D边界框
        if bounding_boxes_2d:
            for i, bbox in enumerate(bounding_boxes_2d):
                x, y, w, h = bbox['bbox']
                label = bbox.get('label', f'Object {i+1}')

                # 添加矩形框
                fig.add_shape(
                    type="rect",
                    x0=x, y0=y,
                    x1=x+w, y1=y+h,
                    line=dict(color="red", width=2),
                    row=1, col=1
                )

                # 添加标签
                fig.add_annotation(
                    x=x, y=y-5,
                    text=label,
                    showarrow=False,
                    font=dict(color="red", size=10),
                    row=1, col=1
                )

        # 融合图像
        fig.add_trace(
            go.Image(z=image),
            row=1, col=2
        )

        # 投影雷达点
        projected_points = self.project_points_to_image(points_3d, camera_matrix, extrinsic_matrix)
        if len(projected_points) > 0:
            h, w = image.shape[:2]
            valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) & \
                        (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
            valid_points = projected_points[valid_mask]

            if len(valid_points) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=valid_points[:, 0],
                        y=valid_points[:, 1],
                        mode='markers',
                        marker=dict(color='lime', size=4),
                        name='Radar Points',
                        hovertemplate='X: %{x:.0f}px<br>Y: %{y:.0f}px<extra></extra>'
                    ),
                    row=1, col=2
                )

        # 更新布局
        fig.update_layout(
            title="Interactive Radar-Image Fusion Visualization",
            height=600,
            showlegend=True
        )

        # 更新坐标轴
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2)

        return fig

    def visualize_3d_fusion(self, points_3d: np.ndarray, image: np.ndarray,
                           camera_matrix: np.ndarray, extrinsic_matrix: np.ndarray,
                           bounding_boxes_3d: Optional[List[Dict]] = None,
                           image_depth: float = 10.0) -> o3d.geometry.PointCloud:
        """
        3D空间中的融合可视化

        Args:
            points_3d: 雷达点云
            image: 图像
            camera_matrix: 相机内参
            extrinsic_matrix: 外参矩阵
            bounding_boxes_3d: 3D边界框列表
            image_depth: 图像平面深度

        Returns:
            Open3D可视化对象
        """
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window('3D Fusion Visualization', 1200, 800)

        # 添加雷达点云
        if len(points_3d) > 0:
            pcd_radar = o3d.geometry.PointCloud()
            pcd_radar.points = o3d.utility.Vector3dVector(points_3d)
            pcd_radar.paint_uniform_color([1, 0, 0])  # 红色
            vis.add_geometry(pcd_radar)

        # 创建图像平面的3D表示
        h, w = image.shape[:2]
        image_plane_3d = self.create_image_plane_3d(
            image, camera_matrix, extrinsic_matrix, image_depth
        )
        vis.add_geometry(image_plane_3d)

        # 添加3D边界框
        if bounding_boxes_3d:
            line_sets = self.create_3d_bounding_boxes(bounding_boxes_3d)
            for line_set in line_sets:
                vis.add_geometry(line_set)

        # 设置视图
        vis.get_render_option().background_color = [0.1, 0.1, 0.1]
        vis.get_render_option().point_size = 2.0

        return vis

    def create_image_plane_3d(self, image: np.ndarray, camera_matrix: np.ndarray,
                             extrinsic_matrix: np.ndarray, depth: float) -> o3d.geometry.PointCloud:
        """
        创建3D图像平面

        Args:
            image: 图像
            camera_matrix: 相机内参
            extrinsic_matrix: 外参矩阵
            depth: 图像平面深度

        Returns:
            3D图像平面点云
        """
        h, w = image.shape[:2]

        # 在图像平面上创建网格点
        y_coords, x_coords = np.mgrid[0:h:10, 0:w:10]
        image_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])

        # 将图像点转换到相机坐标系
        image_points_homo = np.column_stack([image_points, np.ones(len(image_points))])
        camera_points_homo = np.linalg.inv(camera_matrix) @ image_points_homo.T
        camera_points = (camera_points_homo[:2, :].T / camera_points_homo[2, :]) * depth
        camera_points = np.column_stack([camera_points, np.full(len(camera_points), depth)])

        # 转换到世界坐标系
        if extrinsic_matrix.shape == (3, 4):
            world_points_homo = np.linalg.inv(np.vstack([extrinsic_matrix, [0, 0, 0, 1]])) @ \
                               np.column_stack([camera_points, np.ones(len(camera_points))]).T
        else:
            world_points_homo = np.linalg.inv(extrinsic_matrix) @ \
                               np.column_stack([camera_points, np.ones(len(camera_points))]).T

        world_points = world_points_homo[:3, :].T

        # 创建点云
        image_pcd = o3d.geometry.PointCloud()
        image_pcd.points = o3d.utility.Vector3dVector(world_points)

        # 获取对应的颜色
        colors = image[image_points[:, 1], image_points[:, 0]] / 255.0
        image_pcd.colors = o3d.utility.Vector3dVector(colors)

        return image_pcd

    def create_3d_bounding_boxes(self, bounding_boxes_3d: List[Dict]) -> List[o3d.geometry.LineSet]:
        """
        创建3D边界框线框

        Args:
            bounding_boxes_3d: 3D边界框列表

        Returns:
            Open3D线框对象列表
        """
        line_sets = []

        for bbox in bounding_boxes_3d:
            center = bbox['center']
            size = bbox['size']
            color = bbox.get('color', [0, 1, 0])  # 默认绿色

            # 计算顶点
            x_min, x_max = center[0] - size[0]/2, center[0] + size[0]/2
            y_min, y_max = center[1] - size[1]/2, center[1] + size[1]/2
            z_min, z_max = center[2] - size[2]/2, center[2] + size[2]/2

            vertices = [
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]

            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

            line_sets.append(line_set)

        return line_sets


def main():
    """测试函数"""
    # 配置参数
    config = {
        'image_size': (640, 480),
        'point_size': 3
    }

    # 创建多模态可视化器
    visualizer = MultimodalVisualizer(config)

    # 生成测试数据
    # 1. 测试图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 2. 测试3D点
    points_3d = np.random.randn(100, 3) * 5
    points_3d[:, 2] = np.abs(points_3d[:, 2]) + 5  # 确保在相机前方

    # 3. 相机参数
    camera_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])

    extrinsic_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    # 4. 测试边界框
    bounding_boxes_2d = [
        {'bbox': [100, 100, 50, 80], 'label': 'Car', 'color': 'red'},
        {'bbox': [300, 200, 60, 90], 'label': 'Person', 'color': 'blue'}
    ]

    print("Testing multimodal visualization...")

    # 测试融合可视化
    fig = visualizer.visualize_fusion(
        image, points_3d, camera_matrix, extrinsic_matrix, bounding_boxes_2d
    )
    plt.show()

    # 测试Plotly可视化
    plotly_fig = visualizer.create_plotly_fusion(
        image, points_3d, camera_matrix, extrinsic_matrix, bounding_boxes_2d
    )
    plotly_fig.show()

    print("Multimodal visualization tests completed!")


if __name__ == "__main__":
    main()