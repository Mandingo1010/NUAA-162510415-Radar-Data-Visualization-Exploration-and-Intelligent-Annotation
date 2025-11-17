// 雷达数据智能标注系统 - 前端JavaScript

class RadarAnnotationSystem {
    constructor() {
        this.files = {
            radar: [],
            image: [],
            config: [],
            unknown: []
        };
        this.annotations = [];
        this.currentAnnotation = null;
        this.currentImageName = null;
        this.currentImage = null;
        this.fabricCanvas = null;
        this.processing = false;
        this.sessionId = null; // 新增：用于存储会话ID
        this.reviewStats = {
            total: 0,
            reviewed: 0,
            approved: 0,
            rejected: 0
        };

        this.initializeEventListeners();
        this.initializeCanvas();
        this.loadFromStorage();
    }

    initializeEventListeners() {
        // 文件上传相关
        const uploadZone = document.getElementById('upload-zone');
        const folderInput = document.getElementById('folder-input');

        // 拖拽事件
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.items);
        });

        // 文件选择事件
        folderInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });

        // 处理控制
        document.getElementById('start-processing').addEventListener('click', () => {
            this.startProcessing();
        });

        document.getElementById('visualize-3d').addEventListener('click', () => {
            this.visualizePointCloud();
        });

        // 审核按钮
        document.getElementById('approve-btn').addEventListener('click', () => {
            this.approveCurrentAnnotation();
        });

        document.getElementById('reject-btn').addEventListener('click', () => {
            this.rejectCurrentAnnotation();
        });

        document.getElementById('edit-btn').addEventListener('click', () => {
            this.openEditModal();
        });

        // 批量操作
        document.getElementById('approve-all').addEventListener('click', () => {
            this.approveAllAnnotations();
        });

        document.getElementById('reject-all').addEventListener('click', () => {
            this.rejectAllAnnotations();
        });

        document.getElementById('export-results').addEventListener('click', () => {
            this.exportResults();
        });

        // 缩放控制
        document.getElementById('zoom-in').addEventListener('click', () => {
            this.zoomCanvas(1.2);
        });

        document.getElementById('zoom-out').addEventListener('click', () => {
            this.zoomCanvas(0.8);
        });

        document.getElementById('zoom-reset').addEventListener('click', () => {
            this.resetZoom();
        });

        // 编辑模态框
        document.getElementById('save-edit').addEventListener('click', () => {
            this.saveAnnotationEdit();
        });

        // 置信度滑块
        document.getElementById('edit-confidence').addEventListener('input', (e) => {
            document.getElementById('confidence-value').textContent = e.target.value;
        });

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            this.handleKeyPress(e);
        });
    }

    initializeCanvas() {
        const canvas = document.getElementById('annotation-canvas');
        this.fabricCanvas = new fabric.Canvas(canvas, {
            selection: false,
            backgroundColor: '#f8f9fa'
        });

        // 画布点击事件
        this.fabricCanvas.on('mouse:down', (e) => {
            if (e.target && e.target.annotationId) {
                this.selectAnnotation(e.target.annotationId);
            }
        });

        // 设置画布大小
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    normalizeAnnotation(annotation) {
        if (!annotation) return null;

        const normalized = { ...annotation };
        normalized.image = annotation.image || annotation.image_name || annotation.imageName || '';
        normalized.image_name = normalized.image || annotation.image_name || '';
        normalized.type = annotation.type || annotation.class || annotation.class_name || '未知类别';
        normalized.status = annotation.status || 'pending';

        const bbox = this.extractBBox(annotation);
        normalized.x = typeof annotation.x === 'number' ? annotation.x : bbox.x;
        normalized.y = typeof annotation.y === 'number' ? annotation.y : bbox.y;
        normalized.width = typeof annotation.width === 'number' ? annotation.width : bbox.width;
        normalized.height = typeof annotation.height === 'number' ? annotation.height : bbox.height;
        normalized.confidence = typeof annotation.confidence === 'number'
            ? annotation.confidence
            : parseFloat(annotation.confidence) || 0;

        if (!normalized.bbox) {
            normalized.bbox = [normalized.x, normalized.y, normalized.width, normalized.height];
        }

        return normalized;
    }

    extractBBox(annotation) {
        if (!annotation) {
            return { x: 0, y: 0, width: 0, height: 0 };
        }

        if (typeof annotation.x === 'number' && typeof annotation.width === 'number') {
            return {
                x: annotation.x,
                y: annotation.y || 0,
                width: annotation.width,
                height: annotation.height || 0
            };
        }

        const bbox = annotation.bbox || annotation.bounding_box || null;
        if (Array.isArray(bbox) && bbox.length >= 4) {
            return {
                x: Number(bbox[0]) || 0,
                y: Number(bbox[1]) || 0,
                width: Number(bbox[2]) || 0,
                height: Number(bbox[3]) || 0
            };
        }

        if (bbox && typeof bbox === 'object') {
            const width = bbox.width ?? bbox.w ?? 0;
            const height = bbox.height ?? bbox.h ?? 0;
            return {
                x: bbox.x ?? bbox.left ?? 0,
                y: bbox.y ?? bbox.top ?? 0,
                width: Number(width) || 0,
                height: Number(height) || 0
            };
        }

        return { x: 0, y: 0, width: 0, height: 0 };
    }

    syncAnnotationBBox(annotation) {
        if (!annotation) return;
        const serialized = [annotation.x, annotation.y, annotation.width, annotation.height];

        if (Array.isArray(annotation.bbox)) {
            annotation.bbox = serialized;
        } else if (annotation.bbox && typeof annotation.bbox === 'object') {
            annotation.bbox.x = annotation.x;
            annotation.bbox.y = annotation.y;
            annotation.bbox.width = annotation.width;
            annotation.bbox.height = annotation.height;
        } else {
            annotation.bbox = serialized;
        }
    }

    getAnnotationsForImage(imageName) {
        if (!imageName) return [];
        return this.annotations.filter(ann => ann.image === imageName);
    }

    resizeCanvas() {
        const container = document.getElementById('image-container');
        const width = container.clientWidth;
        const height = 600;

        this.fabricCanvas.setDimensions({ width, height });
    }

    handleFiles(items) {
        const files = [];

        // 处理文件列表
        if (items.length > 0) {
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                if (item.kind === 'file') {
                    const file = item.getAsFile();
                    if (file) {
                        files.push(file);
                    }
                } else if (item instanceof File) {
                    files.push(item);
                }
            }
        }

        // 分类文件
        this.classifyFiles(files);
        this.updateFileList();
        this.updateStatistics();
        this.showNotification('文件分类完成，正在上传到服务器...', 'info');
        this.uploadFilesToServer(); // 新增：上传文件到服务器
    }

    classifyFiles(files) {
        for (const file of files) {
            const category = this.getFileCategory(file);
            const fileInfo = {
                name: file.name,
                size: file.size,
                type: file.type,
                lastModified: file.lastModified,
                file: file
            };

            this.files[category].push(fileInfo);
        }

        this.saveToStorage();
    }

    getFileCategory(file) {
        const name = file.name.toLowerCase();

        // 雷达数据文件
        if (name.endsWith('.csv') || name.endsWith('.npy') ||
            name.endsWith('.json') || name.endsWith('.txt') ||
            name.includes('radar') || name.includes('lidar') ||
            name.includes('pointcloud')) {
            return 'radar';
        }

        // 图像文件
        if (name.endsWith('.jpg') || name.endsWith('.jpeg') ||
            name.endsWith('.png') || name.endsWith('.bmp') ||
            name.endsWith('.tiff')) {
            return 'image';
        }

        // 配置文件
        if (name.endsWith('.yaml') || name.endsWith('.yml') ||
            name.endsWith('.xml') || name.endsWith('.config') ||
            name.includes('calib') || name.includes('param')) {
            return 'config';
        }

        return 'unknown';
    }

    updateFileList() {
        const fileList = document.getElementById('file-list');
        fileList.innerHTML = '';

        // 显示各类文件
        const categories = [
            { key: 'radar', icon: 'satellite-dish', color: 'danger', title: '雷达数据' },
            { key: 'image', icon: 'image', color: 'success', title: '图像文件' },
            { key: 'config', icon: 'cog', color: 'warning', title: '配置文件' },
            { key: 'unknown', icon: 'file', color: 'secondary', title: '其他文件' }
        ];

        categories.forEach(category => {
            const files = this.files[category.key];
            if (files.length > 0) {
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'mb-3';

                const categoryHeader = document.createElement('h6');
                categoryHeader.className = `text-${category.color}`;
                categoryHeader.innerHTML = `<i class="fas fa-${category.icon} me-2"></i>${category.title} (${files.length})`;
                categoryDiv.appendChild(categoryHeader);

                files.forEach(file => {
                    const fileItem = document.createElement('div');
                    fileItem.className = `file-item file-type-${category.key}`;

                    const fileInfo = document.createElement('div');
                    fileInfo.className = 'd-flex justify-content-between align-items-center';

                    const fileName = document.createElement('div');
                    fileName.innerHTML = `
                        <small class="text-muted">${file.name}</small>
                        <div><small>${this.formatFileSize(file.size)}</small></div>
                    `;

                    const fileActions = document.createElement('div');
                    fileActions.innerHTML = `
                        <button class="btn btn-sm btn-outline-primary" onclick="app.viewFile('${category.key}', '${file.name}')">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="app.removeFile('${category.key}', '${file.name}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    `;

                    fileInfo.appendChild(fileName);
                    fileInfo.appendChild(fileActions);
                    fileItem.appendChild(fileInfo);
                    categoryDiv.appendChild(fileItem);
                });

                fileList.appendChild(categoryDiv);
            }
        });
    }

    updateStatistics() {
        document.getElementById('radar-count').textContent = this.files.radar.length;
        document.getElementById('image-count').textContent = this.files.image.length;
        document.getElementById('annotation-count').textContent = this.annotations.length;
        document.getElementById('reviewed-count').textContent = this.reviewStats.reviewed;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async startProcessing() {
        if (this.processing) {
            this.showNotification('正在处理中，请稍候...', 'warning');
            return;
        }
        if (!this.sessionId) {
            this.showNotification('请先上传文件！', 'error');
            return;
        }

        this.processing = true;
        document.getElementById('start-processing').disabled = true;
        document.querySelector('.progress-container').style.display = 'block';
        this.updateProgress(0, '开始处理...');

        try {
            document.getElementById('loading-spinner').style.display = 'block';

            const response = await fetch(`/api/process/${this.sessionId}`, { method: 'POST' });
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            this.annotations = (data.annotations || []).map(ann => this.normalizeAnnotation(ann));
            this.updateStatistics();
            this.updateProgress(100, '处理完成！');
            this.showNotification(`处理完成，共获得 ${data.total_annotations} 个标注`, 'success');

            // 显示第一个图像的标注
            if (this.annotations.length > 0) {
                this.displayImageWithAnnotations(this.annotations[0].image);
            } else {
                this.currentAnnotation = null;
                this.currentImageName = null;
                this.updateAnnotationList();
                this.updateReviewPanel();
            }

        } catch (error) {
            console.error('Processing error:', error);
            this.showNotification('处理失败：' + error.message, 'error');
            this.updateProgress(100, '处理失败');
        } finally {
            this.processing = false;
            document.getElementById('start-processing').disabled = false;
            document.getElementById('loading-spinner').style.display = 'none';
        }
    }

    async displayImageWithAnnotations(imageName, selectedAnnotationId = null) {
        if (!this.sessionId || !imageName) return;

        try {
            // 从服务器加载图像
            const response = await fetch(`/api/image/${this.sessionId}/${imageName}`);
            if (!response.ok) {
                throw new Error(`无法加载图像: ${response.statusText}`);
            }
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);

            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.currentImageName = imageName;

                const annotationsForImage = this.getAnnotationsForImage(imageName);
                if (annotationsForImage.length > 0) {
                    const target = selectedAnnotationId
                        ? annotationsForImage.find(ann => ann.id === selectedAnnotationId)
                        : annotationsForImage[0];
                    this.currentAnnotation = target || annotationsForImage[0];
                } else {
                    this.currentAnnotation = null;
                }

                this.renderCurrentImageAnnotations(this.currentAnnotation ? this.currentAnnotation.id : null);
                this.updateAnnotationList();
                this.updateReviewPanel();
            };
            img.src = imageUrl;

        } catch (error) {
            console.error('Error displaying image:', error);
            this.showNotification(`显示图像失败: ${error.message}`, 'error');
        }
    }

    displayImageOnCanvas(img) {
        // 调整图像大小以适应画布
        const canvas = this.fabricCanvas;
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;

        const scale = Math.min(canvasWidth / img.width, canvasHeight / img.height);
        const width = img.width * scale;
        const height = img.height * scale;
        const left = (canvasWidth - width) / 2;
        const top = (canvasHeight - height) / 2;

        // 清除画布
        canvas.clear();

        // 添加图像
        const fabricImage = new fabric.Image(img, {
            left: left,
            top: top,
            scaleX: scale,
            scaleY: scale,
            selectable: false,
            evented: false
        });

        canvas.add(fabricImage);
        canvas.renderAll();

        // 更新图像信息
        document.getElementById('image-info').textContent =
            `图像尺寸: ${img.width}×${img.height} | 当前显示: ${Math.round(width)}×${Math.round(height)}`;
    }

    renderCurrentImageAnnotations(highlightId = null) {
        if (!this.currentImage) {
            return;
        }

        this.displayImageOnCanvas(this.currentImage);
        const annotationsForImage = this.getAnnotationsForImage(this.currentImageName);

        annotationsForImage.forEach(annotation => {
            const isHighlighted = highlightId
                ? annotation.id === highlightId
                : (this.currentAnnotation && annotation.id === this.currentAnnotation.id);
            this.drawAnnotation(annotation, isHighlighted);
        });
    }

    drawAnnotation(annotation, isSelected = false) {
        const canvas = this.fabricCanvas;

        // 计算标注在画布上的位置
        const imgElement = canvas.item(0);
        if (!imgElement) return;

        const scale = imgElement.scaleX;
        const left = imgElement.left + annotation.x * scale;
        const top = imgElement.top + annotation.y * scale;
        const width = annotation.width * scale;
        const height = annotation.height * scale;

        const strokeColor = isSelected ? '#fd7e14' : this.getAnnotationColor(annotation.status);
        const strokeWidth = isSelected ? 3 : 2;

        // 创建矩形标注
        const rect = new fabric.Rect({
            left: left,
            top: top,
            width: width,
            height: height,
            fill: 'transparent',
            stroke: strokeColor,
            strokeWidth: strokeWidth,
            selectable: false,
            annotationId: annotation.id
        });

        // 创建标签
        const label = new fabric.Text(
            `${annotation.type} (${(annotation.confidence * 100).toFixed(1)}%)`,
            {
                left: left,
                top: top - 20,
                fontSize: 12,
                backgroundColor: strokeColor,
                fill: 'white',
                selectable: false,
                annotationId: annotation.id
            }
        );

        canvas.add(rect);
        canvas.add(label);
        canvas.renderAll();
    }

    getAnnotationColor(status) {
        switch (status) {
            case 'approved': return '#28a745';
            case 'rejected': return '#dc3545';
            default: return '#007bff';
        }
    }

    updateAnnotationList() {
        const list = document.getElementById('annotation-list');
        list.innerHTML = '';

        if (!this.currentImageName) {
            list.innerHTML = '<p class="text-muted text-center mb-0">暂无标注可显示</p>';
            return;
        }

        const currentAnnotations = this.getAnnotationsForImage(this.currentImageName);
        if (currentAnnotations.length === 0) {
            list.innerHTML = '<p class="text-muted text-center mb-0">该图像暂无标注</p>';
            return;
        }

        const selectedId = this.currentAnnotation ? this.currentAnnotation.id : null;

        currentAnnotations.forEach(annotation => {
            const item = document.createElement('div');
            item.className = `annotation-item ${annotation.id === selectedId ? 'border-primary border-2' : ''}`;
            item.style.cursor = 'pointer';
            item.onclick = () => this.selectAnnotation(annotation.id);

            item.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <strong>${annotation.type}</strong>
                        <div class="small text-muted">
                            位置: (${Math.round(annotation.x)}, ${Math.round(annotation.y)})
                            尺寸: ${Math.round(annotation.width)}×${Math.round(annotation.height)}
                        </div>
                        <div class="small text-muted">
                            置信度: ${(annotation.confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                    <div>
                        <span class="badge bg-${this.getStatusColor(annotation.status)}">
                            ${this.getStatusText(annotation.status)}
                        </span>
                    </div>
                </div>
            `;

            list.appendChild(item);
        });
    }

    updateReviewPanel() {
        const panel = document.getElementById('selected-annotation');
        const details = document.getElementById('annotation-details');

        if (!this.currentAnnotation) {
            panel.style.display = 'none';
            return;
        }

        panel.style.display = 'block';
        details.innerHTML = `
            <div class="mb-2">
                <strong>类型:</strong> ${this.currentAnnotation.type}
            </div>
            <div class="mb-2">
                <strong>位置:</strong> (${Math.round(this.currentAnnotation.x)}, ${Math.round(this.currentAnnotation.y)})
            </div>
            <div class="mb-2">
                <strong>尺寸:</strong> ${Math.round(this.currentAnnotation.width)}×${Math.round(this.currentAnnotation.height)}
            </div>
            <div class="mb-2">
                <strong>置信度:</strong> ${(this.currentAnnotation.confidence * 100).toFixed(1)}%
            </div>
            <div class="mb-2">
                <strong>状态:</strong>
                <span class="badge bg-${this.getStatusColor(this.currentAnnotation.status)}">
                    ${this.getStatusText(this.currentAnnotation.status)}
                </span>
            </div>
        `;

        // 更新审核按钮状态
        const approveBtn = document.getElementById('approve-btn');
        const rejectBtn = document.getElementById('reject-btn');

        approveBtn.disabled = this.currentAnnotation.status === 'approved';
        rejectBtn.disabled = this.currentAnnotation.status === 'rejected';

        // 更新进度
        this.updateReviewProgress();
    }

    selectAnnotation(annotationId) {
        const annotation = this.annotations.find(ann => ann.id === annotationId);
        if (!annotation) return;

        if (this.currentImageName !== annotation.image) {
            this.displayImageWithAnnotations(annotation.image, annotation.id);
            return;
        }

        this.currentAnnotation = annotation;
        this.renderCurrentImageAnnotations(annotation.id);
        this.updateAnnotationList();
        this.updateReviewPanel();
    }

    async reviewAnnotation(annotationId, action) {
        if (!this.sessionId) {
            this.showNotification('会话无效，请重新上传文件', 'error');
            return;
        }

        try {
            const response = await fetch(`/api/review/${this.sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: action,
                    annotation_ids: [annotationId]
                })
            });
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // 更新前端状态
            const annotation = this.annotations.find(ann => ann.id === annotationId);
            if (annotation) {
                annotation.status = action === 'approve' ? 'approved' : 'rejected';
                this.reviewStats.reviewed++;
                if(action === 'approve') this.reviewStats.approved++; else this.reviewStats.rejected++;

                this.updateAnnotationList();
                this.renderCurrentImageAnnotations(annotation.id);
                this.updateReviewPanel();
                this.updateStatistics();
                this.showNotification(`标注已${action === 'approve' ? '通过' : '拒绝'}`, 'success');
                this.moveToNextAnnotation();
            }
        } catch (error) {
            this.showNotification(`审核失败: ${error.message}`, 'error');
        }
    }

    approveCurrentAnnotation() {
        if (!this.currentAnnotation) return;
        this.reviewAnnotation(this.currentAnnotation.id, 'approve');
    }

    rejectCurrentAnnotation() {
        if (!this.currentAnnotation) return;
        this.reviewAnnotation(this.currentAnnotation.id, 'reject');
    }

    moveToNextAnnotation() {
        if (!this.currentAnnotation) return;

        const currentAnnotations = this.getAnnotationsForImage(this.currentAnnotation.image);

        const currentIndex = currentAnnotations.findIndex(
            ann => ann.id === this.currentAnnotation.id
        );

        // 查找下一个未审核的标注
        for (let i = currentIndex + 1; i < currentAnnotations.length; i++) {
            if (currentAnnotations[i].status === 'pending') {
                this.selectAnnotation(currentAnnotations[i].id);
                return;
            }
        }

        // 如果没有下一个未审核的，查找下一个图像
        const currentImageIndex = this.files.image.findIndex(
            img => img.name === this.currentAnnotation.image
        );
        if (currentImageIndex === -1) return;

        for (let i = currentImageIndex + 1; i < this.files.image.length; i++) {
            const nextImageAnnotations = this.getAnnotationsForImage(this.files.image[i].name);

            const pendingAnnotation = nextImageAnnotations.find(ann => ann.status === 'pending');
            if (pendingAnnotation) {
                this.selectAnnotation(pendingAnnotation.id);
                return;
            }
        }

        this.showNotification('所有标注已审核完成！', 'success');
    }

    approveAllAnnotations() {
        if (confirm('确定要通过所有标注吗？')) {
            this.annotations.forEach(ann => {
                if (ann.status === 'pending') {
                    ann.status = 'approved';
                    this.reviewStats.approved++;
                    this.reviewStats.reviewed++;
                }
            });

            this.renderCurrentImageAnnotations(this.currentAnnotation ? this.currentAnnotation.id : null);
            this.updateAnnotationList();
            this.updateReviewPanel();
            this.updateStatistics();
            this.saveToStorage();

            this.showNotification('所有标注已通过审核', 'success');
        }
    }

    rejectAllAnnotations() {
        if (confirm('确定要拒绝所有标注吗？')) {
            this.annotations.forEach(ann => {
                if (ann.status === 'pending') {
                    ann.status = 'rejected';
                    this.reviewStats.rejected++;
                    this.reviewStats.reviewed++;
                }
            });

            this.renderCurrentImageAnnotations(this.currentAnnotation ? this.currentAnnotation.id : null);
            this.updateAnnotationList();
            this.updateReviewPanel();
            this.updateStatistics();
            this.saveToStorage();

            this.showNotification('所有标注已拒绝', 'info');
        }
    }

    openEditModal() {
        if (!this.currentAnnotation) return;

        // 填充编辑表单
        document.getElementById('edit-class').value = this.currentAnnotation.type;
        document.getElementById('edit-x').value = Math.round(this.currentAnnotation.x);
        document.getElementById('edit-y').value = Math.round(this.currentAnnotation.y);
        document.getElementById('edit-width').value = Math.round(this.currentAnnotation.width);
        document.getElementById('edit-height').value = Math.round(this.currentAnnotation.height);
        document.getElementById('edit-confidence').value = this.currentAnnotation.confidence;
        document.getElementById('confidence-value').textContent = this.currentAnnotation.confidence;

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('editAnnotationModal'));
        modal.show();
    }

    saveAnnotationEdit() {
        if (!this.currentAnnotation) return;

        // 更新标注数据
        this.currentAnnotation.type = document.getElementById('edit-class').value;
        this.currentAnnotation.x = parseFloat(document.getElementById('edit-x').value);
        this.currentAnnotation.y = parseFloat(document.getElementById('edit-y').value);
        this.currentAnnotation.width = parseFloat(document.getElementById('edit-width').value);
        this.currentAnnotation.height = parseFloat(document.getElementById('edit-height').value);
        this.currentAnnotation.confidence = parseFloat(document.getElementById('edit-confidence').value);
        this.syncAnnotationBBox(this.currentAnnotation);

        // 重新显示
        this.renderCurrentImageAnnotations(this.currentAnnotation.id);
        this.updateAnnotationList();
        this.updateReviewPanel();
        this.saveToStorage();

        // 关闭模态框
        bootstrap.Modal.getInstance(document.getElementById('editAnnotationModal')).hide();

        this.showNotification('标注已更新', 'success');
    }

    exportResults() {
        const results = {
            timestamp: new Date().toISOString(),
            statistics: this.reviewStats,
            files: {
                radar: this.files.radar.map(f => ({ name: f.name, size: f.size })),
                image: this.files.image.map(f => ({ name: f.name, size: f.size })),
                config: this.files.config.map(f => ({ name: f.name, size: f.size }))
            },
            annotations: this.annotations.map(ann => ({
                id: ann.id,
                image: ann.image,
                type: ann.type,
                x: ann.x,
                y: ann.y,
                width: ann.width,
                height: ann.height,
                confidence: ann.confidence,
                status: ann.status
            }))
        };

        // 下载JSON文件
        const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `radar_annotations_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showNotification('结果已导出', 'success');
    }

    zoomCanvas(factor) {
        const canvas = this.fabricCanvas;
        const zoom = canvas.getZoom() * factor;
        canvas.setZoom(zoom);
        canvas.renderAll();
    }

    resetZoom() {
        const canvas = this.fabricCanvas;
        canvas.setZoom(1);
        canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
        canvas.renderAll();
    }

    updateProgress(percent, status) {
        document.getElementById('processing-progress').style.width = percent + '%';
        document.getElementById('processing-status').textContent = status;
    }

    updateReviewProgress() {
        const targetImage = this.currentImageName || (this.currentAnnotation ? this.currentAnnotation.image : null);
        const progressBadge = document.getElementById('current-progress');
        const progressBar = document.getElementById('review-progress');

        if (!targetImage) {
            progressBadge.textContent = '0/0';
            progressBar.style.width = '0%';
            return;
        }

        const currentAnnotations = this.getAnnotationsForImage(targetImage);

        const reviewed = currentAnnotations.filter(ann => ann.status !== 'pending').length;
        const total = currentAnnotations.length;

        progressBadge.textContent = `${reviewed}/${total}`;
        progressBar.style.width = total > 0 ? (reviewed / total * 100) + '%' : '0%';
    }

    handleKeyPress(e) {
        if (!this.currentAnnotation) return;

        switch(e.key.toLowerCase()) {
            case 'a':
                this.approveCurrentAnnotation();
                break;
            case 'd':
                this.rejectCurrentAnnotation();
                break;
            case 'arrowleft':
                this.navigateToPreviousAnnotation();
                break;
            case 'arrowright':
                this.navigateToNextAnnotation();
                break;
        }
    }

    navigateToPreviousAnnotation() {
        if (!this.currentAnnotation) return;
        const currentAnnotations = this.getAnnotationsForImage(this.currentAnnotation.image);

        const currentIndex = currentAnnotations.findIndex(
            ann => ann.id === this.currentAnnotation.id
        );

        if (currentIndex > 0) {
            this.selectAnnotation(currentAnnotations[currentIndex - 1].id);
        }
    }

    navigateToNextAnnotation() {
        if (!this.currentAnnotation) return;
        const currentAnnotations = this.getAnnotationsForImage(this.currentAnnotation.image);

        const currentIndex = currentAnnotations.findIndex(
            ann => ann.id === this.currentAnnotation.id
        );

        if (currentIndex < currentAnnotations.length - 1) {
            this.selectAnnotation(currentAnnotations[currentIndex + 1].id);
        }
    }

    getStatusColor(status) {
        switch (status) {
            case 'approved': return 'success';
            case 'rejected': return 'danger';
            default: return 'primary';
        }
    }

    getStatusText(status) {
        switch (status) {
            case 'approved': return '已通过';
            case 'rejected': return '已拒绝';
            default: return '待审核';
        }
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        container.appendChild(alert);

        // 自动移除通知
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }

    saveToStorage() {
        const data = {
            files: this.files,
            annotations: this.annotations,
            reviewStats: this.reviewStats
        };
        localStorage.setItem('radarAnnotationData', JSON.stringify(data));
    }

    loadFromStorage() {
        const saved = localStorage.getItem('radarAnnotationData');
        if (saved) {
            try {
                const data = JSON.parse(saved);
                this.files = data.files || this.files;
                this.annotations = (data.annotations || []).map(ann => this.normalizeAnnotation(ann));
                this.reviewStats = data.reviewStats || this.reviewStats;

                this.updateFileList();
                this.updateStatistics();
            } catch (e) {
                console.error('Failed to load saved data:', e);
            }
        }
    }

    removeFile(category, fileName) {
        const index = this.files[category].findIndex(f => f.name === fileName);
        if (index >= 0) {
            this.files[category].splice(index, 1);
            this.updateFileList();
            this.updateStatistics();
            this.saveToStorage();
            this.showNotification('文件已删除', 'info');
        }
    }

    viewFile(category, fileName) {
        const file = this.files[category].find(f => f.name === fileName);
        if (file) {
            this.showNotification(`查看文件: ${fileName}`, 'info');
            // 这里可以实现文件预览功能
        }
    }

    async uploadFilesToServer() {
        const formData = new FormData();
        let fileCount = 0;
        for (const category in this.files) {
            this.files[category].forEach(fileInfo => {
                formData.append('files', fileInfo.file, fileInfo.name);
                fileCount++;
            });
        }

        if (fileCount === 0) return;

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }
            this.sessionId = data.session_id;
            this.showNotification(`文件上传成功，会话ID: ${this.sessionId}`, 'success');

            // 如果有雷达文件，启用3D可视化按钮
            if (this.files.radar.length > 0) {
                document.getElementById('visualize-3d').disabled = false;
            }

        } catch (error) {
            this.showNotification(`文件上传失败: ${error.message}`, 'error');
            this.sessionId = null;
            document.getElementById('visualize-3d').disabled = true;
        }
    }

    async visualizePointCloud() {
        if (!this.sessionId) {
            this.showNotification('请先上传文件并开始处理会话', 'warning');
            return;
        }

        const container = document.getElementById('visualization-container');
        container.innerHTML = '<div class="text-center p-5"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">加载中...</span></div><p class="mt-2">正在生成三维点云图...</p></div>';

        const modal = new bootstrap.Modal(document.getElementById('visualizationModal'));
        modal.show();

        try {
            const response = await fetch(`/api/visualize/pointcloud/${this.sessionId}`);
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            container.innerHTML = data.html;
        } catch (error) {
            container.innerHTML = `<div class="alert alert-danger">无法加载三维可视化: ${error.message}</div>`;
        }
    }
}

// 初始化应用
const app = new RadarAnnotationSystem();