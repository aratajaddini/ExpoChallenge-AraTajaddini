(function () {
    'use strict';

    /* ─── CONFIG ──────────────────────────────────────────────── */
    const STORAGE = sessionStorage;          // tab-scoped: dies when the tab closes
    const STORAGE_KEY = 'tracesort_api_key';
    const MAX_UPLOAD_BYTES = 150 * 1024 * 1024;
    const VIDEO_EXT = /\.(mp4|mov|mkv|avi|webm)$/i;

    const CONFIG = {
        API_BASE: window.__API_BASE__ || 'http://127.0.0.1:8000',
        API_KEY: '',
        IMAGE_TIMEOUT_MS: 30000,
        VIDEO_TIMEOUT_MS: 300000,
    };

    // ── key helpers (sessionStorage-based) ──────────────────────
    function loadStoredKey() {
        try {
            return STORAGE.getItem(STORAGE_KEY) || '';
        } catch {
            return '';
        }
    }

    function storeKey(key) {
        try {
            STORAGE.setItem(STORAGE_KEY, key);
        } catch { /* non-fatal */ }
    }

    function authHeaders(extra = {}) {
        return { 'X-API-Key': loadStoredKey(), ...extra };
    }

    /** Validate the stored key against the backend. Returns identity or null. */
    async function verifyStoredKey() {
        if (!loadStoredKey()) {
            setKeyNote('Enter your API key to begin.', 'err');
            return null;
        }
        try {
            const res = await fetch(`${CONFIG.API_BASE}/auth/verify`, {
                headers: authHeaders(),
                signal: AbortSignal.timeout(8000),
            });
            if (res.ok) {
                const data = await res.json();
                const identity = data.identity || data.label || 'authenticated';
                setKeyNote('✓ Key valid — ' + identity, 'ok');
                if (apiKeyInput) apiKeyInput.value = loadStoredKey();
                return identity;
            } else if (res.status === 401) {
                CONFIG.API_KEY = '';
                storeKey('');
                setKeyNote('Stored key invalid — enter it again.', 'err');
                if (apiKeyInput) apiKeyInput.value = '';
                return null;
            } else {
                setKeyNote('Server error during key verification.', 'err');
                return null;
            }
        } catch {
            setKeyNote('Could not reach server to verify key.', 'err');
            return null;
        }
    }

    // Apply any stored key immediately, then verify it
    CONFIG.API_KEY = loadStoredKey();
    verifyStoredKey();

    /* ─── DOM refs for API key input ──────────────────────────── */
    const apiKeyInput = document.getElementById('apiKeyInput');
    const setKeyBtn   = document.getElementById('setKeyBtn');
    const apiKeyNote  = document.getElementById('apiKeyNote');

    // state
    let lastPredictionId = null;
    let selectedFile = null;
    let selectedKind = 'image';
    let previewUrl = null;
    let modelClasses = [];

    /* ─── helpers ──────────────────────────────────────────────── */
    function setKeyNote(msg, state) {
        if (!apiKeyNote) return;
        apiKeyNote.textContent = msg;
        if (state) apiKeyNote.dataset.state = state;
        else delete apiKeyNote.dataset.state;
    }

    /** Pull the key from the input into CONFIG and persist it. */
    function applyKeyFromInput() {
        if (!apiKeyInput) return false;
        const k = apiKeyInput.value.trim();
        if (!k) {
            setKeyNote('Key cannot be empty.', 'err');
            apiKeyInput.focus();
            return false;
        }
        storeKey(k);
        CONFIG.API_KEY = k;
        verifyStoredKey();   // immediately check the new key
        return true;
    }

    /** Called before every authenticated request. */
    function ensureApiKey() {
        if (loadStoredKey()) return true;
        if (apiKeyInput && apiKeyInput.value.trim()) return applyKeyFromInput();

        setKeyNote('Enter your API key first.', 'err');
        if (apiKeyInput) apiKeyInput.focus();
        showError('API key required. Set it above, then try again.');
        return false;
    }

    // Bind Set Key button
    if (setKeyBtn) {
        setKeyBtn.addEventListener('click', applyKeyFromInput);
    }
    if (apiKeyInput) {
        apiKeyInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                applyKeyFromInput();
            }
        });
    }

    /** fetch with auth header + abort timeout. */
    async function apiFetch(path, options = {}, timeoutMs) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs || CONFIG.IMAGE_TIMEOUT_MS);
        try {
            const headers = authHeaders(options.headers || {});
            // For FormData we must not set Content-Type; browser sets it with boundary.
            if (options.body instanceof FormData) {
                delete headers['Content-Type'];
            }
            return await fetch(CONFIG.API_BASE + path, {
                ...options,
                headers,
                signal: controller.signal,
            });
        } finally {
            clearTimeout(timer);
        }
    }

    /** Fetch label list from the backend. Non-fatal on failure. */
    async function loadClasses() {
        try {
            const res = await apiFetch('/classes');
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();
            modelClasses = Array.isArray(data.classes) ? data.classes : [];
            if (modelClasses.length) {
                buildFeedbackOptions(modelClasses, null);
            }
        } catch (err) {
            console.warn('Could not load classes:', err.message);
            modelClasses = [];
        }
    }

    /* ─── pretty / sort helpers ────────────────────────────────── */
    function prettyLabel(name) {
        return String(name)
            .replace(/[_-]+/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }

    function sortedScores(scores) {
        return Object.entries(scores || {}).sort((a, b) => b[1] - a[1]);
    }

    function buildFeedbackOptions(names, topClass) {
        const select = document.getElementById('feedbackClass');
        if (!select) return;
        select.innerHTML = '';
        names.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = prettyLabel(name);
            opt.selected = name === topClass;
            select.appendChild(opt);
        });
    }

    /* ─── DOM elements (may be missing on some pages) ──────────── */
    // Splash
    const enterBtn = document.getElementById('enterBtn');
    const splash = document.getElementById('splash');
    const mainContent = document.getElementById('main-content');

    // Nav
    const toggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');

    // Demo
    const launchBtn = document.getElementById('launchDemoBtn');
    const classifier = document.getElementById('classifier');
    const fileInput = document.getElementById('fileInput');
    const uploadZone = document.getElementById('uploadZone');
    const placeholder = document.getElementById('uploadPlaceholder');
    const previewImg = document.getElementById('previewImg');
    const previewVideo = document.getElementById('previewVideo');
    const classifyBtn = document.getElementById('classifyBtn');
    const resetBtn = document.getElementById('resetBtn');

    const resultEmpty = document.getElementById('resultEmpty');
    const resultLoading = document.getElementById('resultLoading');
    const loadingText = document.getElementById('loadingText');
    const resultError = document.getElementById('resultError');
    const resultContent = document.getElementById('resultContent');

    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const historyEmpty = document.getElementById('historyEmpty');
    const historyList = document.getElementById('historyList');

    const feedbackBtn = document.getElementById('feedbackBtn');

    // ============================================================
    // 1. SPLASH PAGE
    // ============================================================
    if (enterBtn && splash && mainContent) {
        enterBtn.addEventListener('click', function () {
            splash.classList.add('exit');
            setTimeout(function () {
                splash.style.display = 'none';
                mainContent.classList.add('visible');
                requestAnimationFrame(function () {
                    mainContent.classList.add('animate');
                });
                const heroVideo = document.querySelector('.hero-video');
                if (heroVideo) heroVideo.play().catch(function () {});
            }, 1000);
        });

        enterBtn.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    }

    // ============================================================
    // 2. HAMBURGER MENU
    // ============================================================
    if (toggle && navLinks) {
        toggle.addEventListener('click', function () {
            const expanded = this.getAttribute('aria-expanded') !== 'true';
            this.setAttribute('aria-expanded', String(expanded));
            navLinks.classList.toggle('open');
        });

        navLinks.querySelectorAll('a').forEach(function (link) {
            link.addEventListener('click', function () {
                toggle.setAttribute('aria-expanded', 'false');
                navLinks.classList.remove('open');
            });
        });

        document.addEventListener('click', function (e) {
            if (!e.target.closest('.site-nav')) {
                toggle.setAttribute('aria-expanded', 'false');
                navLinks.classList.remove('open');
            }
        });
    }

    // ============================================================
    // 3. SMOOTH SCROLL
    // ============================================================
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            const targetEl = document.querySelector(targetId);
            if (!targetEl) return;
            e.preventDefault();
            const navHeight = 64;
            const targetPos = targetEl.getBoundingClientRect().top + window.pageYOffset - navHeight;
            window.scrollTo({ top: targetPos, behavior: 'smooth' });
        });
    });

    // ============================================================
    // 4. LAUNCH DEMO – show/hide classifier and handle API key
    // ============================================================
    if (launchBtn && classifier) {
        launchBtn.addEventListener('click', async () => {
            const wasHidden = classifier.hasAttribute('hidden');
            classifier.hidden = !wasHidden;
            launchBtn.setAttribute('aria-expanded', String(!wasHidden));
            launchBtn.textContent = wasHidden ? 'Hide Demo' : 'Launch Demo';

            if (!wasHidden) return;

            // Verify stored key and load classes if valid
            const identity = await verifyStoredKey();
            if (identity) {
                await loadClasses();
                loadHistory();
            } else {
                if (apiKeyInput) apiKeyInput.focus();
            }

            classifier.scrollIntoView({ behavior: 'smooth', block: 'start' });
            if (!loadStoredKey() && apiKeyInput) apiKeyInput.focus();
        });
    }

    // ============================================================
    // 5. FILE UPLOAD & PREVIEW
    // ============================================================
    const hasClassifier = fileInput && uploadZone && classifyBtn && resetBtn &&
                          previewImg && resultEmpty && resultContent;

    function detectKind(file) {
        if (file.type.startsWith('image/')) return 'image';
        if (file.type.startsWith('video/') || VIDEO_EXT.test(file.name)) return 'video';
        return null;
    }

    function releasePreviewUrl() {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
    }

    function handleFile(file) {
        const kind = detectKind(file);
        if (!kind) {
            showError('Unsupported file. Use an image (PNG/JPG/WEBP/BMP) or a video (MP4/MOV/MKV/AVI/WEBM).');
            return;
        }
        if (file.size > MAX_UPLOAD_BYTES) {
            showError('File too large: ' + (file.size / 1048576).toFixed(1) + ' MB (max 150 MB).');
            return;
        }

        releasePreviewUrl();
        selectedFile = file;
        selectedKind = kind;
        previewUrl = URL.createObjectURL(file);

        if (kind === 'image') {
            previewImg.src = previewUrl;
            previewImg.hidden = false;
            if (previewVideo) {
                previewVideo.hidden = true;
                previewVideo.removeAttribute('src');
                previewVideo.load();
            }
        } else if (previewVideo) {
            previewVideo.src = previewUrl;
            previewVideo.hidden = false;
            previewImg.hidden = true;
            previewImg.removeAttribute('src');
        } else {
            showError('Video preview element is missing from the page.');
            return;
        }

        if (placeholder) placeholder.hidden = true;
        classifyBtn.disabled = false;
        resetBtn.disabled = false;
        resetResultView();
    }

    function resetAll() {
        releasePreviewUrl();
        selectedFile = null;
        selectedKind = 'image';
        fileInput.value = '';
        previewImg.removeAttribute('src');
        previewImg.hidden = true;
        if (previewVideo) {
            previewVideo.removeAttribute('src');
            previewVideo.hidden = true;
            previewVideo.load();
        }
        if (placeholder) placeholder.hidden = false;
        classifyBtn.disabled = true;
        resetBtn.disabled = true;
        resetResultView();
    }

    if (hasClassifier) {
        fileInput.addEventListener('change', function () {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        });

        ['dragenter', 'dragover'].forEach(function (evt) {
            uploadZone.addEventListener(evt, function (e) {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(function (evt) {
            uploadZone.addEventListener(evt, function (e) {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
            });
        });
        uploadZone.addEventListener('drop', function (e) {
            const file = e.dataTransfer && e.dataTransfer.files[0];
            if (file) handleFile(file);
        });

        resetBtn.addEventListener('click', resetAll);
        classifyBtn.addEventListener('click', classify);
    }

    if (refreshHistoryBtn) refreshHistoryBtn.addEventListener('click', loadHistory);
    if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', clearHistory);

    window.addEventListener('beforeunload', releasePreviewUrl);

    // ============================================================
    // 6. CLASSIFY
    // ============================================================
    async function classify() {
        if (!selectedFile) return;
        if (!ensureApiKey()) return;

        const isVideo = selectedKind === 'video';
        showLoading(isVideo);
        classifyBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res = await apiFetch('/predict', {
                method: 'POST',
                body: formData,
            }, isVideo ? CONFIG.VIDEO_TIMEOUT_MS : CONFIG.IMAGE_TIMEOUT_MS);

            if (res.status === 401) {
                setKeyNote('Key invalid or expired. Get a new one.', 'err');
                showError('Invalid API key. Please enter a new shift key.');
                classifyBtn.disabled = false;
                return;
            }

            if (!res.ok) {
                await handleHttpError(res, isVideo);
                return;
            }
            const data = await res.json();
            renderResult(data);
            loadHistory();
        } catch (err) {
            if (err.name === 'AbortError') {
                showError('Request timed out. ' + (isVideo
                    ? 'Video inference is slow without a GPU — try a shorter clip.'
                    : 'The server did not respond in time.'));
            } else {
                showError('Could not reach the server. Make sure the backend is running on ' + CONFIG.API_BASE + '.');
            }
        } finally {
            classifyBtn.disabled = false;
        }
    }

    // ============================================================
    // HTTP error mapping
    // ============================================================
    async function handleHttpError(res, isVideo) {
        let detail = '';
        try {
            const body = await res.json();
            if (typeof body.detail === 'string') detail = ' ' + body.detail;
        } catch (e) { /* non-JSON body */ }

        switch (res.status) {
            case 400:
                showError('Bad request (400).' + detail);
                break;
            case 401:
                CONFIG.API_KEY = '';
                try { STORAGE.removeItem(STORAGE_KEY); } catch {}
                if (apiKeyInput) apiKeyInput.value = '';
                setKeyNote('Invalid key — enter it again.', 'err');
                showError('Invalid API key (401).');
                classifyBtn.disabled = false;
                break;
            case 413:
                showError('File too large for the server (413). Max 150 MB.');
                break;
            case 415:
                showError(isVideo
                    ? 'Video is not supported by the backend yet (415). Upload an image instead.'
                    : 'Unsupported file type (415).' + detail);
                break;
            case 422:
                showError('The server could not process this file (422).' + detail);
                break;
            case 500:
                showError('Server error (500). Check the backend logs.' + detail);
                break;
            default:
                showError('Server error: ' + res.status + '.' + detail);
        }
    }

    // ============================================================
    // Rendering (dynamic class list from server)
    // ============================================================
    function renderResult(data) {
        hideAll();
        resultContent.hidden = false;

        const scores = data.scores || {};
        modelClasses.forEach(name => {
            if (!(name in scores)) scores[name] = 0;
        });
        const rows = sortedScores(scores);

        const topClass = data.class_name ?? data.top_class ?? (rows[0] ? rows[0][0] : 'unknown');
        const topScore = data.confidence ?? (rows[0] ? rows[0][1] : 0);

        resultContent.innerHTML = `
            <div class="top-class">
                <span class="top-label">Detected</span>
                <span class="top-value">${prettyLabel(topClass)}</span>
            </div>
            <div class="meta-row">
                <div class="meta-item">
                    <span class="meta-label">Confidence</span>
                    <span class="meta-value">${(topScore * 100).toFixed(1)}%</span>
                </div>
                ${data.inference_ms != null ? `
                <div class="meta-item">
                    <span class="meta-label">Inference</span>
                    <span class="meta-value">${Math.round(data.inference_ms)} ms</span>
                </div>` : ''}
            </div>
            <div class="scores">
                ${rows.map(([name, score]) => {
                    const pct = (score * 100).toFixed(1);
                    const isTop = name === topClass ? ' is-top' : '';
                    return `
                    <div class="score-row${isTop}">
                        <div class="score-head">
                            <span>${prettyLabel(name)}</span>
                            <span>${pct}%</span>
                        </div>
                        <div class="score-track" role="progressbar" aria-label="${prettyLabel(name)} confidence" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
                            <div class="score-fill" style="width:${pct}%"></div>
                        </div>
                    </div>`;
                }).join('')}
            </div>`;

        lastPredictionId = data.id ?? null;
        const wrap = document.getElementById('feedbackWrap');
        const note = document.getElementById('feedbackNote');
        if (note) {
            note.textContent = '';
            note.className = 'feedback-note';
        }
        if (wrap) {
            if (lastPredictionId !== null && rows.length) {
                const classNames = modelClasses.length ? modelClasses : rows.map(([name]) => name);
                buildFeedbackOptions(classNames, topClass);
                const fbBtn = document.getElementById('feedbackBtn');
                if (fbBtn) fbBtn.disabled = false;
                wrap.hidden = false;
            } else {
                wrap.hidden = true;
            }
        }
    }

    // ============================================================
    // Feedback
    // ============================================================
    async function sendFeedback() {
        const btn = document.getElementById('feedbackBtn');
        const note = document.getElementById('feedbackNote');
        const correct = document.getElementById('feedbackClass').value;
        if (!btn || !note || lastPredictionId === null || !correct) return;

        btn.disabled = true;
        note.textContent = 'Sending…';
        note.className = 'feedback-note';

        try {
            const res = await apiFetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prediction_id: lastPredictionId,
                    correct_class: correct,
                }),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            note.textContent = 'Thanks — feedback saved.';
            note.classList.add('is-ok');
        } catch (err) {
            note.textContent = `Could not save feedback: ${err.message}`;
            note.classList.add('is-error');
            btn.disabled = false;
        }
    }

    if (feedbackBtn) feedbackBtn.addEventListener('click', sendFeedback);

    // ============================================================
    // History
    // ============================================================
    async function loadHistory() {
        if (!historyList || !historyEmpty) return;
        if (!loadStoredKey()) {
            historyEmpty.textContent = 'Enter an API key to load history.';
            historyEmpty.hidden = false;
            historyList.hidden = true;
            return;
        }

        try {
            const res = await apiFetch('/history', { method: 'GET' });
            if (!res.ok) {
                if (res.status === 401) {
                    storeKey('');
                    try { STORAGE.removeItem(STORAGE_KEY); } catch {}
                    if (apiKeyInput) apiKeyInput.value = '';
                    setKeyNote('Invalid key — enter it again.', 'err');
                }
                historyEmpty.textContent = res.status === 401
                    ? 'Unauthorized (401). The API key was rejected.'
                    : 'Could not load history (' + res.status + ').';
                historyEmpty.hidden = false;
                historyList.hidden = true;
                return;
            }

            const data = await res.json();
            const items = Array.isArray(data) ? data : (data.history || data.items || []);

            if (!items.length) {
                historyEmpty.textContent = 'No history yet.';
                historyEmpty.hidden = false;
                historyList.hidden = true;
                return;
            }

            historyList.innerHTML = '';
            items.forEach(function (item) {
                const li = document.createElement('li');
                li.className = 'history-item';

                const cls = document.createElement('span');
                cls.className = 'h-class';
                cls.textContent = item.predicted_class || item.top_class || '—';

                const meta = document.createElement('span');
                meta.className = 'h-meta';
                meta.textContent = item.source === 'video'
                    ? `video · ${item.frames_used ?? 0}/${item.frames_analyzed ?? 0} frames used`
                    : 'single image';

                li.append(cls, meta);
                historyList.appendChild(li);
            });

            historyEmpty.hidden = true;
            historyList.hidden = false;
        } catch (err) {
            historyEmpty.textContent = 'Could not reach the server for history.';
            historyEmpty.hidden = false;
            historyList.hidden = true;
        }
    }

    async function clearHistory() {
        if (!confirm('Clear all history? This cannot be undone.')) return;
        if (!ensureApiKey()) return;

        clearHistoryBtn.disabled = true;
        const originalText = clearHistoryBtn.textContent;
        clearHistoryBtn.textContent = 'Clearing…';

        try {
            const res = await apiFetch('/history', { method: 'DELETE' });
            if (!res.ok) {
                if (res.status === 401) {
                    storeKey('');
                    try { STORAGE.removeItem(STORAGE_KEY); } catch {}
                    if (apiKeyInput) apiKeyInput.value = '';
                    setKeyNote('Invalid key — enter it again.', 'err');
                }
                throw new Error('Failed to clear history (' + res.status + ').');
            }

            historyList.innerHTML = '';
            historyList.hidden = true;
            historyEmpty.textContent = 'History cleared.';
            historyEmpty.hidden = false;
        } catch (err) {
            historyEmpty.textContent = err.message || 'Could not clear history.';
            historyEmpty.hidden = false;
        } finally {
            clearHistoryBtn.disabled = false;
            clearHistoryBtn.textContent = originalText;
        }
    }

    // ============================================================
    // View-state helpers
    // ============================================================
    function showLoading(isVideo) {
        hideAll();
        resultLoading.hidden = false;
        if (loadingText) {
            loadingText.textContent = isVideo
                ? 'Analyzing video, this may take a minute…'
                : 'Analyzing…';
        }
    }

    function showError(msg) {
        hideAll();
        if (resultError) {
            resultError.textContent = msg;
            resultError.hidden = false;
        }
    }

    function resetResultView() {
        hideAll();
        resultEmpty.hidden = false;
        const timeline = document.getElementById('timeline');
        if (timeline) { timeline.innerHTML = ''; timeline.hidden = true; }
    }

    function hideAll() {
        if (resultEmpty) resultEmpty.hidden = true;
        if (resultLoading) resultLoading.hidden = true;
        if (resultError) resultError.hidden = true;
        if (resultContent) resultContent.hidden = true;
    }

    // ── Bootstrap: verify key again and load classes if valid ──
    (async () => {
        const identity = await verifyStoredKey();
        if (identity) {
            await loadClasses();
        }
    })();
})();