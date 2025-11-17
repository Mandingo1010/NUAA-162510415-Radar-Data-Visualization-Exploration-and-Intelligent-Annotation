#!/usr/bin/env python3
"""
简化版Web雷达标注系统
Simplified Web Radar Annotation System
"""

import os
import json
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
try:
    from flask import Flask, render_template, request, jsonify, send_file
    from werkzeug.utils import secure_filename
    print("Flask imported successfully")
except ImportError:
    print("Flask not available, installing...")
    os.system("python -m pip install flask")
    from flask import Flask, render_template, request, jsonify, send_file
    from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'radar_annotation_system_simple'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 全局数据存储
sessions = {}

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """处理文件上传"""
    try:
        session_id = str(uuid.uuid4())
        session_data = {
            'files': [],
            'annotations': [],
            'created_at': datetime.now().isoformat()
        }

        files = request.files.getlist('files')
        uploaded_count = 0

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_info = {
                    'name': filename,
                    'size': len(file.read()),
                    'category': 'unknown'
                }
                file.seek(0)  # 重置文件指针
                session_data['files'].append(file_info)
                uploaded_count += 1

        sessions[session_id] = session_data

        return jsonify({
            'session_id': session_id,
            'uploaded_files': uploaded_count,
            'message': 'Files uploaded successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/<session_id>', methods=['POST'])
def process_session(session_id):
    """处理会话"""
    try:
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404

        session = sessions[session_id]

        # 生成模拟标注数据
        annotations = []
        for i in range(3):  # 生成3个模拟标注
            annotation = {
                'id': f"annotation_{i}",
                'class': ['car', 'person', 'bicycle'][i % 3],
                'bbox': [50 + i * 100, 50 + i * 50, 80, 120],
                'confidence': 0.8 + i * 0.05,
                'status': 'pending'
            }
            annotations.append(annotation)

        session['annotations'] = annotations

        return jsonify({
            'status': 'completed',
            'annotations': annotations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotation/<session_id>/<annotation_id>', methods=['PUT'])
def update_annotation(session_id, annotation_id):
    """更新标注"""
    try:
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404

        data = request.get_json()
        for annotation in sessions[session_id]['annotations']:
            if annotation['id'] == annotation_id:
                annotation.update(data)
                annotation['updated_at'] = datetime.now().isoformat()
                return jsonify({'success': True})

        return jsonify({'error': 'Annotation not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<session_id>')
def export_results(session_id):
    """导出结果"""
    try:
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404

        session = sessions[session_id]

        export_data = {
            'session_id': session_id,
            'export_time': datetime.now().isoformat(),
            'files': session['files'],
            'annotations': session['annotations']
        }

        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(export_data, temp_file, indent=2)
        temp_file.close()

        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'annotations_{session_id}.json'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'message': 'Simple Web Annotation System is running',
        'active_sessions': len(sessions)
    })

if __name__ == '__main__':
    print("🚀 Starting Simplified Web Radar Annotation System...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🎯 Features: File upload, Auto-classification, Annotation review")
    print("⚠️  Press Ctrl+C to stop the server")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)