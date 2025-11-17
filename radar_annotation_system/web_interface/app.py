#!/usr/bin/env python3
"""
雷达数据智能标注系统 - Flask Web应用
Radar Data Intelligent Annotation System - Flask Web Application
"""

import os
import sys
import json
import uuid
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
import io
import base64

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_processing.radar_processor import RadarProcessor
    from data_processing.image_processor import ImageProcessor
    from annotation.object_detector import ObjectDetectionPipeline
    from annotation.radar_image_fusion import RadarImageFusion
    from utils.config_manager import ConfigManager
    from visualization.point_cloud_viz import PointCloudVisualizer
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Please ensure all dependencies are installed")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'radar_annotation_system_2025'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {
    'radar': ['.csv', '.npy', '.json', '.txt', '.bin'],
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
    'config': ['.yaml', '.yml', '.xml', '.json', '.config', '.txt'],
    'general': ['.zip', '.tar', '.gz']
}

# 全局变量
session_data = {}
config_manager = None


def allowed_file(filename, category='general'):
    """检查文件是否被允许"""
    if not filename:
        return False

    ext = os.path.splitext(filename)[1].lower()
    if category in ALLOWED_EXTENSIONS:
        return ext in ALLOWED_EXTENSIONS[category]
    return ext in ALLOWED_EXTENSIONS['general']


def get_file_category(filename):
    """根据文件名和扩展名确定文件类别"""
    name = filename.lower()
    ext = os.path.splitext(filename)[1].lower()

    # 雷达数据文件
    if (ext in ALLOWED_EXTENSIONS['radar'] or
        'radar' in name or 'lidar' in name or 'pointcloud' in name or
        'velodyne' in name or 'pcloud' in name):
        return 'radar'

    # 图像文件
    if ext in ALLOWED_EXTENSIONS['image']:
        return 'image'

    # 配置文件
    if (ext in ALLOWED_EXTENSIONS['config'] or
        'calib' in name or 'param' in name or 'config' in name):
        return 'config'

    return 'unknown'


def process_zip_file(zip_path):
    """处理压缩文件"""
    extracted_files = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.is_dir():
                # 提取文件内容
                with zip_ref.open(file_info) as file:
                    content = file.read()

                # 创建临时文件
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(content)
                temp_file.close()

                extracted_files.append({
                    'name': file_info.filename,
                    'path': temp_file.name,
                    'size': file_info.file_size,
                    'category': get_file_category(file_info.filename)
                })

    return extracted_files


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """处理文件上传"""
    try:
        session_id = str(uuid.uuid4())
        session_files = {
            'radar': [],
            'image': [],
            'config': [],
            'unknown': []
        }

        if 'files' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400

        files = request.files.getlist('files')
        uploaded_count = 0

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                category = get_file_category(filename)

                # 保存文件
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
                file.save(file_path)

                file_info = {
                    'name': filename,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'category': category,
                    'upload_time': datetime.now().isoformat()
                }

                session_files[category].append(file_info)
                uploaded_count += 1

        # 保存会话数据
        session_data[session_id] = {
            'files': session_files,
            'annotations': [],
            'processing_status': 'uploaded',
            'created_at': datetime.now().isoformat()
        }

        return jsonify({
            'session_id': session_id,
            'uploaded_files': uploaded_count,
            'file_categories': {k: len(v) for k, v in session_files.items()}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/<session_id>', methods=['POST'])
def process_session(session_id):
    """处理会话中的文件"""
    try:
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404

        session = session_data[session_id]
        session['processing_status'] = 'processing'

        # 处理图像文件
        annotations = []

        for image_file in session['files']['image']:
            image_annotations = process_image_file(image_file, session)
            annotations.extend(image_annotations)

        # 如果有雷达文件，进行融合处理
        if session['files']['radar']:
            annotations = process_radar_fusion(annotations, session['files']['radar'], session)

        session['annotations'] = annotations
        session['processing_status'] = 'completed'
        session['processed_at'] = datetime.now().isoformat()

        return jsonify({
            'status': 'completed',
            'total_annotations': len(annotations),
            'annotations': annotations[:10]  # 返回前10个作为示例
        })

    except Exception as e:
        session_data[session_id]['processing_status'] = 'error'
        session_data[session_id]['error'] = str(e)
        return jsonify({'error': str(e)}), 500


def process_image_file(image_file, session):
    """处理单个图像文件，使用YOLO进行真实的目标检测"""
    try:
        # 1. 读取图像
        image_path = image_file['path']
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return []

        # 2. 初始化检测器
        # 假设配置文件在项目根目录的configs/下
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
        config_manager = ConfigManager(config_path)
        detection_pipeline = ObjectDetectionPipeline(config_manager.get_section('detection'))

        # 3. 执行目标检测
        detection_results = detection_pipeline.detect(image, detector_names=['yolo'])
        yolo_detections = detection_results.get('yolo', [])

        # 4. 格式化标注结果
        annotations = []
        for i, detection in enumerate(yolo_detections):
            annotation = {
                'id': f"{image_file['name']}_{i}",
                'image_name': image_file['name'],
                'class': detection['class_name'],
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'status': 'pending',  # 初始状态为待审核
                'created_at': datetime.now().isoformat()
            }
            annotations.append(annotation)

        return annotations

    except Exception as e:
        print(f"Error processing image {image_file['name']} with YOLO: {e}")
        return []


def process_radar_fusion(annotations, radar_files, session):
    """使用真实数据处理雷达-图像融合"""
    try:
        if not annotations or not radar_files:
            return annotations

        # 1. 初始化融合模块
        # 假设配置文件在项目根目录的configs/下
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
        config_manager = ConfigManager(config_path)
        fusion = RadarImageFusion(config_manager.get_section('fusion'))

        # 2. 加载雷达数据
        # 为简化起见，我们使用第一个雷达文件
        radar_file = radar_files[0]
        radar_processor = RadarProcessor(config_manager.get_section('radar'))
        point_cloud = radar_processor.load_point_cloud(radar_file['path'])

        # 3. 加载相机参数 (如果存在)
        camera_params = {}
        if session['files']['config']:
            config_file = session['files']['config'][0]
            # 这里需要一个函数来从配置文件加载相机内外参
            # camera_params = load_camera_params(config_file['path'])
            pass # 暂时留空

        # 4. 执行融合
        updated_annotations = fusion.fuse_radar_with_detections(
            annotations,
            point_cloud,
            camera_params
        )

        return updated_annotations

    except Exception as e:
        print(f"Error in radar fusion: {e}")
        return annotations


@app.route('/api/annotations/<session_id>')
def get_annotations(session_id):
    """获取会话的标注数据"""
    if session_id not in session_data:
        return jsonify({'error': '会话不存在'}), 404

    session = session_data[session_id]
    return jsonify({
        'annotations': session['annotations'],
        'status': session['processing_status']
    })


@app.route('/api/annotation/<session_id>/<annotation_id>', methods=['PUT'])
def update_annotation(session_id, annotation_id):
    """更新单个标注"""
    try:
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404

        session = session_data[session_id]
        annotations = session['annotations']

        # 查找并更新标注
        for annotation in annotations:
            if annotation['id'] == annotation_id:
                data = request.get_json()
                annotation.update(data)
                annotation['updated_at'] = datetime.now().isoformat()

                return jsonify({'success': True, 'annotation': annotation})

        return jsonify({'error': '标注不存在'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/review/<session_id>', methods=['POST'])
def batch_review(session_id):
    """批量审核标注"""
    try:
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404

        data = request.get_json()
        action = data.get('action')  # 'approve' or 'reject'
        annotation_ids = data.get('annotation_ids', [])

        if not annotation_ids:
            # 如果没有指定ID，则处理所有pending的标注
            annotation_ids = [
                ann['id'] for ann in session_data[session_id]['annotations']
                if ann['status'] == 'pending'
            ]

        updated_count = 0
        for annotation in session_data[session_id]['annotations']:
            if annotation['id'] in annotation_ids:
                annotation['status'] = 'approved' if action == 'approve' else 'rejected'
                annotation['reviewed_at'] = datetime.now().isoformat()
                updated_count += 1

        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'action': action
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<session_id>')
def export_results(session_id):
    """导出结果"""
    try:
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404

        session = session_data[session_id]

        # 准备导出数据
        export_data = {
            'session_id': session_id,
            'export_time': datetime.now().isoformat(),
            'statistics': {
                'total_files': sum(len(files) for files in session['files'].values()),
                'total_annotations': len(session['annotations']),
                'approved_annotations': len([a for a in session['annotations'] if a['status'] == 'approved']),
                'rejected_annotations': len([a for a in session['annotations'] if a['status'] == 'rejected']),
                'pending_annotations': len([a for a in session['annotations'] if a['status'] == 'pending'])
            },
            'files': {},
            'annotations': session['annotations']
        }

        # 添加文件信息（不包含实际文件内容）
        for category, files in session['files'].items():
            export_data['files'][category] = [
                {
                    'name': f['name'],
                    'size': f['size'],
                    'category': f['category']
                } for f in files
            ]

        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(export_data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()

        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'radar_annotations_{session_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mimetype='application/json'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<session_id>/<image_name>')
def get_image(session_id, image_name):
    """获取图像文件"""
    try:
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404

        session = session_data[session_id]

        # 查找图像文件
        image_file = None
        for files in session['files'].values():
            for f in files:
                if f['name'] == image_name:
                    image_file = f
                    break
            if image_file:
                break

        if not image_file:
            return jsonify({'error': '图像不存在'}), 404

        return send_file(image_file['path'], mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions')
def list_sessions():
    """列出所有会话"""
    sessions = []
    for session_id, session in session_data.items():
        sessions.append({
            'id': session_id,
            'created_at': session['created_at'],
            'status': session['processing_status'],
            'total_annotations': len(session['annotations']),
            'total_files': sum(len(files) for files in session['files'].values())
        })

    return jsonify({'sessions': sessions})


@app.route('/api/config')
def get_config():
    """获取系统配置"""
    if config_manager:
        return jsonify(config_manager.get_config_dict())
    else:
        # 返回默认配置
        return jsonify({
            'radar': {
                'noise_threshold': 0.1,
                'dbscan_eps': 0.5,
                'dbscan_min_samples': 5
            },
            'detection': {
                'yolo': {
                    'enabled': True,
                    'confidence_threshold': 0.5
                }
            },
            'fusion': {
                'distance_threshold': 2.0,
                'confidence_weight': 0.7
            }
        })


@app.route('/api/config', methods=['POST'])
def update_config():
    """更新系统配置"""
    try:
        new_config = request.get_json()

        if config_manager:
            # 更新配置
            for section, values in new_config.items():
                if hasattr(config_manager.config, section):
                    config_obj = getattr(config_manager.config, section)
                    for key, value in values.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)

            # 保存配置
            config_manager.save_config()

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(session_data)
    })


@app.route('/api/visualize/pointcloud/<session_id>', methods=['GET'])
def visualize_pointcloud(session_id):
    """生成并返回点云的3D可视化HTML"""
    try:
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404

        session = session_data[session_id]
        if not session['files']['radar']:
            return jsonify({'error': '会话中没有雷达文件'}), 400

        # 使用第一个雷达文件进行可视化
        radar_file = session['files']['radar'][0]

        # 1. 处理雷达数据
        # 使用RadarProcessor来正确加载点云
        # 假设配置文件在项目根目录的configs/下
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
        config_manager = ConfigManager(config_path)
        radar_processor = RadarProcessor(config_manager.get_section('radar'))
        points = radar_processor.load_point_cloud(radar_file['path'])

        # 2. 初始化可视化器
        vis_config = {'window_size': (800, 600)} # 简单配置
        visualizer = PointCloudVisualizer(vis_config)

        # 3. 生成Plotly HTML
        fig = visualizer.plot_with_plotly(points[:, :3]) # 取前三列作为x,y,z
        html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # 4. 关闭可视化器窗口以释放资源
        visualizer.close()

        return jsonify({'html': html_content})

    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({'error': '文件太大，请上传小于1000MB的文件'}), 413


@app.errorhandler(404)
def not_found(e):
    """404错误处理"""
    return jsonify({'error': '页面不存在'}), 404


@app.errorhandler(500)
def internal_error(e):
    """500错误处理"""
    return jsonify({'error': '服务器内部错误'}), 500


def cleanup_old_sessions():
    """清理旧会话"""
    current_time = datetime.now()

    for session_id, session in list(session_data.items()):
        try:
            created_time = datetime.fromisoformat(session['created_at'])
            # 删除超过24小时的会话
            if (current_time - created_time).total_seconds() > 24 * 3600:
                # 删除相关文件
                for files in session['files'].values():
                    for file_info in files:
                        try:
                            os.remove(file_info['path'])
                        except:
                            pass

                # 删除会话数据
                del session_data[session_id]
                print(f"Cleaned up old session: {session_id}")

        except Exception as e:
            print(f"Error cleaning up session {session_id}: {e}")


if __name__ == '__main__':
    # 初始化配置管理器
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
        config_manager = ConfigManager(config_path)
    except:
        print("Warning: Could not load configuration, using defaults")

    # 定期清理旧会话
    import threading
    import time

    def cleanup_worker():
        while True:
            time.sleep(3600)  # 每小时清理一次
            cleanup_old_sessions()

    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()

    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )