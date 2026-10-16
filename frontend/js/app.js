/**
 * Sport Vision — Main Application
 * WebSocket communication, UI orchestration, and event handling
 */

(function () {
    'use strict';

    // ============ DOM Elements ============
    const $ = (id) => document.getElementById(id);

    const heroSection = $('hero-section');
    const analysisSection = $('analysis-section');
    const videoCanvas = $('video-canvas');
    const videoOverlay = $('video-overlay');
    const videoOverlayMessage = $('video-overlay-message');
    const progressFill = $('progress-fill');
    const statusDot = $('status-dot');
    const statusText = $('status-text');
    const actionIcon = $('action-icon');
    const actionName = $('action-name');
    const confidenceFill = $('confidence-fill');
    const confidenceText = $('confidence-text');
    const actionCard = $('action-card');
    const timelineContainer = $('timeline-container');
    const uploadInput = $('video-upload');
    const cameraBtn = $('camera-btn');
    const cameraPreview = $('camera-preview');
    const demoBtn = $('demo-btn');
    const demoSelector = $('demo-selector');
    const demoGrid = $('demo-grid');
    const engineGrid = $('engine-grid');
    const engineCount = $('engine-count');
    const jobPanel = $('job-panel');
    const jobForm = $('job-form');
    const jobVideo = $('job-video');
    const jobTemplate = $('job-template');
    const jobTemplateRow = $('job-template-row');
    const jobDescription = $('job-engine-description');
    const jobState = $('job-state');
    const jobClose = $('job-close');
    const jobSubmit = $('job-submit');
    const btnStop = $('btn-stop');
    const btnBack = $('btn-back');
    const appNotice = $('app-notice');

    const videoCtx = videoCanvas.getContext('2d');

    // ============ State ============
    let ws = null;
    let isAnalyzing = false;
    let dashboard = null;
    let currentSport = 'all';
    let frameImage = new Image();
    let lastActionColor = '#00f0ff';
    let selectedEngine = null;
    let jobPollTimer = null;
    let noticeTimer = null;
    let cameraStream = null;
    let cameraCaptureCanvas = null;
    let cameraCaptureContext = null;
    let cameraFrameInFlight = false;
    let cameraCaptureTimer = null;

    // ============ Initialize ============
    function init() {
        dashboard = new Dashboard();
        bindEvents();
        setStatus('ready', '就绪');
        loadDemos();
        loadIntegrations();
    }

    // ============ Event Binding ============
    function bindEvents() {
        // Sport toggle
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.nav-btn').forEach(b => {
                    b.classList.remove('active');
                    b.setAttribute('aria-pressed', 'false');
                });
                btn.classList.add('active');
                btn.setAttribute('aria-pressed', 'true');
                currentSport = btn.dataset.sport;
                renderIntegrations();
            });
        });

        // Upload
        uploadInput.addEventListener('change', handleUpload);
        cameraBtn.addEventListener('click', startCameraAnalysis);

        // Demo button
        demoBtn.addEventListener('click', () => {
            demoSelector.hidden = !demoSelector.hidden;
            demoBtn.setAttribute('aria-expanded', String(!demoSelector.hidden));
        });

        // Stop / Back
        btnStop.addEventListener('click', stopAnalysis);
        btnBack.addEventListener('click', goBack);

        engineGrid.addEventListener('click', handleEngineAction);
        jobForm.addEventListener('submit', submitJob);
        jobClose.addEventListener('click', closeJobPanel);
        jobState.addEventListener('click', handleJobStateAction);
    }

    // ============ Unified Engine Hub ============
    let integrations = [];

    async function loadIntegrations() {
        try {
            const response = await fetch('/api/integrations');
            if (!response.ok) throw new Error('无法读取引擎目录');
            const data = await response.json();
            integrations = data.integrations || [];
            engineCount.textContent = `${data.available}/${data.total} 已接入`;
            engineGrid.setAttribute('aria-busy', 'false');
            renderIntegrations();
        } catch (err) {
            console.warn('Failed to load integrations:', err);
            engineCount.textContent = '目录不可用';
            engineGrid.innerHTML = '<p class="engine-empty">暂时无法读取分析引擎目录。</p>';
            engineGrid.setAttribute('aria-busy', 'false');
            announce('无法读取本机分析引擎目录，请确认服务正在运行。', 'error');
        }
    }

    function renderIntegrations() {
        if (!engineGrid) return;
        const filtered = integrations.filter(engine =>
            currentSport === 'all' || engine.sport === 'all' || engine.sport === currentSport
        );

        if (filtered.length === 0) {
            engineGrid.innerHTML = '<p class="engine-empty">该项目暂未配置专项引擎。</p>';
            return;
        }

        engineGrid.innerHTML = filtered.map(engine => `
            <article class="engine-card ${engine.status !== 'available' ? 'engine-card--inactive' : ''}">
                <div class="engine-card-top">
                    <span class="engine-icon">${engine.icon}</span>
                    <span class="engine-status">${engine.status === 'available' ? '已接入' : '待补全'}</span>
                </div>
                <h3>${escapeHtml(engine.name)}</h3>
                <p>${escapeHtml(engine.summary)}</p>
                <div class="engine-capabilities">
                    ${engine.capabilities.slice(0, 3).map(capability => `<span>${escapeHtml(capability)}</span>`).join('')}
                </div>
                ${engine.job_supported
                    ? `<button class="engine-run-btn" data-engine-id="${engine.id}">上传视频并分析</button>`
                    : `<span class="engine-mode">${escapeHtml(engine.notes || '专项服务 / 数据资源')}</span>`}
                <footer>${escapeHtml(engine.category)} · ${escapeHtml(engine.runtime)}</footer>
            </article>
        `).join('');
    }

    function escapeHtml(value) {
        const node = document.createElement('span');
        node.textContent = value;
        return node.innerHTML;
    }

    function handleEngineAction(event) {
        const button = event.target.closest('[data-engine-id]');
        if (!button) return;
        selectedEngine = integrations.find(engine => engine.id === button.dataset.engineId);
        if (!selectedEngine) return;
        jobPanel.hidden = false;
        jobDescription.textContent = `${selectedEngine.name}：${selectedEngine.summary}`;
        jobTemplateRow.hidden = !selectedEngine.requires_template;
        jobTemplate.required = Boolean(selectedEngine.requires_template);
        jobSubmit.disabled = true;
        jobState.textContent = '正在进行运行前检查…';
        jobPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
        checkEngineReadiness(selectedEngine.id);
    }

    function closeJobPanel() {
        jobPanel.hidden = true;
        jobForm.reset();
        selectedEngine = null;
        if (jobPollTimer) window.clearTimeout(jobPollTimer);
        jobPollTimer = null;
    }

    async function checkEngineReadiness(engineId) {
        try {
            const response = await fetch(`/api/integrations/${engineId}/preflight`);
            const readiness = await response.json();
            if (!selectedEngine || selectedEngine.id !== engineId) return;
            if (!response.ok) throw new Error(readiness.detail || '检查失败');
            selectedEngine = { ...selectedEngine, requires_template: readiness.requires_template };
            jobTemplateRow.hidden = !readiness.requires_template;
            jobTemplate.required = readiness.requires_template;
            if (readiness.ready) {
                jobSubmit.disabled = false;
                jobState.textContent = readiness.execution_mode === 'specialist'
                    ? '专项引擎已就绪：可上传文件并开始分析。'
                    : '专项模型未就绪，将使用本地规则分析：输出姿态、动作/运动事件、轨迹指标和标注视频。';
                return;
            }
            const problems = [];
            if (!readiness.source_ready) problems.push('项目入口缺失');
            if (!readiness.python_ready) problems.push('专属 Python 环境不可用');
            if (readiness.missing_assets.length) problems.push(`缺少模型：${readiness.missing_assets.join('、')}`);
            if (readiness.missing_packages.length) problems.push(`缺少依赖：${readiness.missing_packages.join('、')}`);
            jobState.textContent = `该引擎尚不能启动：${problems.join('；') || '未配置为托管任务'}。`;
        } catch (error) {
            jobState.textContent = `运行前检查失败：${error.message}`;
        }
    }

    async function submitJob(event) {
        event.preventDefault();
        if (!selectedEngine || !jobVideo.files[0]) return;
        if (selectedEngine.requires_template && !jobTemplate.files[0]) {
            jobState.textContent = '请先选择球场模板图像。';
            return;
        }

        try {
            jobState.textContent = '正在上传分析文件…';
            const video = await uploadForJob(jobVideo.files[0], 'video');
            const template = selectedEngine.requires_template
                ? await uploadForJob(jobTemplate.files[0], 'template')
                : null;
            const response = await fetch('/api/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ engine_id: selectedEngine.id, video_id: video.id, template_id: template && template.id }),
            });
            const job = await response.json();
            if (!response.ok) throw new Error(formatApiError(job.detail, '任务创建失败'));
            jobState.textContent = `任务 ${job.id} 已加入队列。`;
            pollJob(job.id);
        } catch (error) {
            jobState.textContent = `无法启动任务：${error.message}`;
        }
    }

    async function uploadForJob(file, kind) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('kind', kind);
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || '上传失败');
        return data;
    }

    async function pollJob(jobId) {
        try {
            const response = await fetch(`/api/jobs/${jobId}`);
            const job = await response.json();
            if (!response.ok) throw new Error(formatApiError(job.detail, '读取任务状态失败'));
            const lastLog = job.logs.length ? `<br><code>${escapeHtml(job.logs[job.logs.length - 1])}</code>` : '';
            jobState.innerHTML = `任务 ${job.id}：${escapeHtml(job.status)}${job.error ? ` — ${escapeHtml(job.error)}` : ''}${lastLog}`;
            if (job.status === 'queued' || job.status === 'running') {
                jobState.insertAdjacentHTML('beforeend', ` <button class="job-cancel" data-cancel-job="${job.id}">取消任务</button>`);
                jobPollTimer = window.setTimeout(() => pollJob(jobId), 1500);
            } else if (job.status === 'complete') {
                const files = await (await fetch(`/api/jobs/${job.id}/files`)).json();
                const summary = await loadJobSummary(job.id, files.files);
                renderJobResults(job, files.files, summary);
            }
        } catch (error) {
            jobState.textContent = `任务状态读取失败：${error.message}`;
        }
    }

    async function handleJobStateAction(event) {
        const button = event.target.closest('[data-cancel-job]');
        if (!button) return;
        button.disabled = true;
        try {
            const response = await fetch(`/api/jobs/${button.dataset.cancelJob}`, { method: 'DELETE' });
            const job = await response.json();
            if (!response.ok) throw new Error(formatApiError(job.detail, '取消失败'));
            if (jobPollTimer) window.clearTimeout(jobPollTimer);
            jobPollTimer = null;
            jobState.textContent = `任务 ${job.id} 已取消。`;
        } catch (error) {
            jobState.textContent = `取消失败：${error.message}`;
        }
    }

    function formatApiError(detail, fallback) {
        if (typeof detail === 'string') return detail;
        if (detail && typeof detail === 'object') {
            const parts = [detail.message];
            if (detail.missing_assets?.length) parts.push(`缺少模型：${detail.missing_assets.join('、')}`);
            if (detail.missing_packages?.length) parts.push(`缺少依赖：${detail.missing_packages.join('、')}`);
            return parts.filter(Boolean).join('；') || fallback;
        }
        return fallback;
    }

    async function loadJobSummary(jobId, files) {
        const summary = files.find(file => file.name === 'summary.json');
        if (!summary) return null;
        try {
            const response = await fetch(`/api/jobs/${jobId}/files/summary.json`);
            return response.ok ? await response.json() : null;
        } catch (error) {
            console.warn('Unable to load job summary:', error);
            return null;
        }
    }

    function renderJobResults(job, files, summary) {
        const links = files.map(file => (
            `<a href="/api/jobs/${job.id}/files/${encodeURIComponent(file.name)}" target="_blank" rel="noopener">${escapeHtml(file.name)}</a>`
        )).join('');
        if (!summary) {
            jobState.innerHTML = `任务 ${job.id}：完成<div class="job-result-links">${links}</div>`;
            return;
        }
        const quality = summary.analysis_quality || {};
        const events = Object.entries(summary.event_breakdown || {})
            .map(([name, count]) => `<span>${escapeHtml(name)} × ${Number(count)}</span>`).join('') || '<span>未检出规则事件</span>';
        const coverage = Number(summary.pose_coverage || 0);
        jobState.innerHTML = `
            <div class="job-result-title">任务 ${escapeHtml(job.id)}：完成 · ${escapeHtml(summary.execution_mode || job.execution_mode || 'unknown')}</div>
            <div class="job-result-metrics">
                <div><small>姿态覆盖</small><strong>${Math.round(coverage * 100)}%</strong></div>
                <div><small>数据质量</small><strong>${escapeHtml(String(quality.data_quality_score ?? '—'))}/100</strong></div>
                <div><small>规则事件</small><strong>${escapeHtml(String(summary.event_count ?? 0))}</strong></div>
            </div>
            <p class="job-result-note">${escapeHtml(quality.note || summary.method || '结果已生成。')}</p>
            <div class="job-event-tags">${events}</div>
            <div class="job-result-links">${links}</div>
        `;
    }

    // ============ Status ============
    function setStatus(type, text) {
        statusDot.className = 'status-dot';
        if (type === 'processing') {
            statusDot.classList.add('processing');
        } else if (type === 'active') {
            statusDot.classList.add('active');
        }
        statusText.textContent = text;
    }

    function announce(message, type = 'info') {
        if (!appNotice) return;
        appNotice.textContent = message;
        appNotice.dataset.type = type;
        appNotice.hidden = false;
        if (noticeTimer) window.clearTimeout(noticeTimer);
        noticeTimer = window.setTimeout(() => {
            appNotice.hidden = true;
        }, 6000);
    }

    function showVideoOverlay(message) {
        if (videoOverlayMessage) videoOverlayMessage.textContent = message;
        videoOverlay.classList.remove('hidden');
    }

    // ============ Demo Videos ============
    async function loadDemos() {
        try {
            const resp = await fetch('/api/demos');
            const data = await resp.json();
            if (!resp.ok) throw new Error('无法读取 Demo 列表');
            demoGrid.setAttribute('aria-busy', 'false');

            if (data.demos.length === 0) {
                demoGrid.innerHTML = `
                    <div class="demo-card" style="grid-column: 1/-1; text-align: center; color: var(--text-dim);">
                        <p>暂无 Demo 视频</p>
                        <p style="font-size: 0.7rem; margin-top: 8px;">
                            将 .mp4 文件放入 <code>demo_videos/</code> 目录
                        </p>
                    </div>
                `;
                return;
            }

            demoGrid.innerHTML = data.demos.map(d => `
                <div class="demo-card" data-demo-id="${d.id}" data-filename="${d.filename}">
                    <div class="demo-card-name">🎬 ${d.name}</div>
                    <div class="demo-card-size">${d.size_mb} MB</div>
                </div>
            `).join('');

            // Bind click
            demoGrid.querySelectorAll('.demo-card').forEach(card => {
                card.addEventListener('click', () => {
                    const demoId = card.dataset.demoId;
                    if (demoId) startAnalysis('demo', demoId);
                });
            });
        } catch (err) {
            console.warn('Failed to load demos:', err);
            demoGrid.setAttribute('aria-busy', 'false');
            demoGrid.innerHTML = '<p class="engine-empty">无法读取 Demo 视频列表。</p>';
        }
    }

    // ============ Upload ============
    async function handleUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        setStatus('processing', '上传中...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await resp.json().catch(() => ({}));

            if (!resp.ok || data.error || !data.id) {
                const message = data.error || formatApiError(data.detail, '上传失败，请检查文件格式和服务状态。');
                announce(message, 'error');
                setStatus('ready', '就绪');
                return;
            }

            startAnalysis('upload', data.id);
        } catch (err) {
            announce('上传失败: ' + err.message, 'error');
            setStatus('ready', '就绪');
        }
    }

    // ============ Browser Camera ============
    async function startCameraAnalysis() {
        if (!navigator.mediaDevices?.getUserMedia) {
            announce('当前浏览器不支持摄像头接口，请使用最新版 Chrome、Edge 或 Firefox。', 'error');
            return;
        }
        stopCameraCapture();
        cameraBtn.disabled = true;
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
            });
            cameraPreview.srcObject = cameraStream;
            await cameraPreview.play();
            startAnalysis('camera');
        } catch (error) {
            stopCameraCapture();
            const reason = error?.name === 'NotAllowedError'
                ? '未获得摄像头权限。请在浏览器地址栏允许摄像头后重试。'
                : '无法打开摄像头。请确认设备未被其他应用占用。';
            announce(reason, 'error');
            setStatus('ready', '摄像头不可用');
        } finally {
            cameraBtn.disabled = false;
        }
    }

    function startCameraCapture() {
        if (!cameraStream || !isAnalyzing || !cameraPreview.videoWidth) {
            if (cameraStream && isAnalyzing) {
                cameraCaptureTimer = window.setTimeout(startCameraCapture, 80);
            }
            return;
        }
        if (!cameraCaptureCanvas) {
            cameraCaptureCanvas = document.createElement('canvas');
            cameraCaptureContext = cameraCaptureCanvas.getContext('2d', { alpha: false });
        }
        captureCameraFrame();
    }

    function captureCameraFrame() {
        if (!isAnalyzing || !cameraStream || cameraFrameInFlight || !ws || ws.readyState !== WebSocket.OPEN) return;
        const sourceWidth = cameraPreview.videoWidth;
        const sourceHeight = cameraPreview.videoHeight;
        if (!sourceWidth || !sourceHeight) {
            cameraCaptureTimer = window.setTimeout(captureCameraFrame, 80);
            return;
        }
        const width = Math.min(640, sourceWidth);
        const height = Math.max(1, Math.round(sourceHeight * width / sourceWidth));
        if (cameraCaptureCanvas.width !== width || cameraCaptureCanvas.height !== height) {
            cameraCaptureCanvas.width = width;
            cameraCaptureCanvas.height = height;
        }
        cameraCaptureContext.drawImage(cameraPreview, 0, 0, width, height);
        const image = cameraCaptureCanvas.toDataURL('image/jpeg', 0.72).split(',', 2)[1];
        if (!image) {
            announce('无法读取摄像头画面。', 'error');
            stopAnalysis();
            return;
        }
        cameraFrameInFlight = true;
        ws.send(JSON.stringify({
            type: 'camera_frame',
            image,
            timestamp_ms: Math.round(performance.now()),
        }));
    }

    function scheduleNextCameraFrame() {
        cameraFrameInFlight = false;
        if (cameraStream && isAnalyzing) {
            cameraCaptureTimer = window.setTimeout(captureCameraFrame, 0);
        }
    }

    function stopCameraCapture() {
        if (cameraCaptureTimer) window.clearTimeout(cameraCaptureTimer);
        cameraCaptureTimer = null;
        cameraFrameInFlight = false;
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        if (cameraPreview) cameraPreview.srcObject = null;
    }

    // ============ WebSocket Analysis ============
    function startAnalysis(source, sourceId) {
        if (source !== 'camera') stopCameraCapture();
        if (ws) {
            try { ws.close(); } catch (_) { /* Browser already owns teardown. */ }
            ws = null;
        }
        // Show analysis UI
        heroSection.hidden = true;
        analysisSection.hidden = false;
        showVideoOverlay('正在连接分析引擎...');

        // Reset
        dashboard.reset();
        timelineContainer.innerHTML = '';
        progressFill.style.width = '0%';
        isAnalyzing = true;

        setStatus('processing', '连接分析引擎...');

        // Connect WebSocket
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws/analyze`);

        ws.onopen = () => {
            setStatus('processing', '分析中...');

            const msg = { type: 'start', source, sport: currentSport };
            if (sourceId) msg.id = sourceId;

            ws.send(JSON.stringify(msg));
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleWSMessage(msg);
            } catch (error) {
                console.warn('Invalid WebSocket message:', error);
                isAnalyzing = false;
                setStatus('ready', '分析响应无效');
                showVideoOverlay('分析响应无效，请重新开始。');
                announce('收到无效的分析响应。', 'error');
            }
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            isAnalyzing = false;
            setStatus('ready', '连接错误');
            showVideoOverlay('无法连接分析服务，请确认服务正在运行。');
            announce('分析连接失败。', 'error');
        };

        ws.onclose = () => {
            if (isAnalyzing) {
                setStatus('ready', '连接已关闭');
                isAnalyzing = false;
                showVideoOverlay('分析连接已关闭。');
            }
            stopCameraCapture();
        };
    }

    function handleWSMessage(msg) {
        switch (msg.type) {
            case 'started':
                videoOverlay.classList.add('hidden');
                setStatus('processing', msg.source === 'camera' ? '摄像头识别中...' : '分析中...');
                if (msg.source === 'camera') startCameraCapture();
                break;

            case 'frame':
                // A frame can already be buffered when the user presses Stop.
                // Do not repaint the completed/returned interface with it.
                if (isAnalyzing) renderFrame(msg.data);
                if (msg.data?.source === 'camera') scheduleNextCameraFrame();
                break;

            case 'complete':
                setStatus('active', '分析完成');
                isAnalyzing = false;
                stopCameraCapture();
                announce('视频分析已完成。');
                break;

            case 'stopped':
                setStatus('ready', '已停止');
                isAnalyzing = false;
                stopCameraCapture();
                showVideoOverlay('分析已停止。');
                break;

            case 'error':
                setStatus('ready', `错误: ${msg.message}`);
                isAnalyzing = false;
                stopCameraCapture();
                showVideoOverlay(msg.message || '分析失败，请重新开始。');
                announce(msg.message || '分析失败，请重新开始。', 'error');
                break;
        }
    }

    // ============ Frame Rendering ============
    function renderFrame(data) {
        if (!data.frame_base64) return;

        // Update progress
        progressFill.style.width = Number.isFinite(data.progress) ? `${data.progress * 100}%` : '100%';

        // Draw frame on canvas
        frameImage.onload = () => {
            videoCanvas.width = frameImage.naturalWidth;
            videoCanvas.height = frameImage.naturalHeight;
            videoCtx.drawImage(frameImage, 0, 0);
        };
        frameImage.src = 'data:image/jpeg;base64,' + data.frame_base64;

        // Update action display
        if (data.action) {
            updateActionDisplay(data.action);
        }

        // Update dashboard
        if (data.pose) {
            dashboard.updateAngles(data.pose.joint_angles);
            dashboard.updateBiomechanics(data.pose.biomechanics);
        }

        if (data.action) {
            dashboard.updateStats(data.action.action_counts);
        }

        // Update heatmap (every 10 frames)
        if (data.frame_number % 10 === 0 && data.heatmap_data) {
            dashboard.updateHeatmap(data.heatmap_data, data.width, data.height);
        }
    }

    function updateActionDisplay(actionData) {
        const info = actionData.action_info;

        // Update icon and name
        actionIcon.textContent = info.icon;
        actionName.textContent = info.name;

        // Update confidence
        const pct = Math.round((actionData.rule_match_score ?? actionData.confidence ?? 0) * 100);
        confidenceFill.style.width = `${pct}%`;
        confidenceText.textContent = `${pct}%`;

        // Update action card border color
        if (info.color !== lastActionColor) {
            actionCard.style.borderColor = info.color;
            actionName.style.color = info.color;
            lastActionColor = info.color;
        }

        // Add to timeline if new action
        if (actionData.is_new_action) {
            addTimelineItem(actionData);
        }
    }

    function addTimelineItem(actionData) {
        const info = actionData.action_info;
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.style.borderColor = info.color;
        item.style.color = info.color;
        item.style.background = info.color + '15';
        item.textContent = `${info.icon} ${info.name}`;
        timelineContainer.appendChild(item);

        // Auto-scroll to right
        timelineContainer.scrollLeft = timelineContainer.scrollWidth;

        // Flash the action card
        actionCard.classList.remove('flash');
        void actionCard.offsetWidth; // force reflow
        actionCard.classList.add('flash');
    }

    // ============ Controls ============
    function stopAnalysis() {
        stopCameraCapture();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'stop' }));
        }
        isAnalyzing = false;
        setStatus('ready', '已停止');
        showVideoOverlay('正在停止分析…');
    }

    function goBack() {
        stopAnalysis();
        if (ws) {
            ws.close();
            ws = null;
        }
        analysisSection.hidden = true;
        heroSection.hidden = false;
        setStatus('ready', '就绪');
    }

    // ============ Start ============
    document.addEventListener('DOMContentLoaded', init);
})();
