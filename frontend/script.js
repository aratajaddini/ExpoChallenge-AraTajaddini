(function () {
    'use strict';

    /* ─── CONFIG ──────────────────────────────────────────────── */
    const STORAGE = sessionStorage;          // tab-scoped: dies when the tab closes
    const STORAGE_KEY = 'tracesort_api_key';
    const MAX_UPLOAD_BYTES = 150 * 1024 * 1024;
    const VIDEO_EXT = /\.(mp4|mov|mkv|avi|webm)$/i;

    const CONFIG = {
        API_BASE: window.__API_BASE__ || '',   // ✅ empty = same origin
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
            // ✅ Security reminder: visible to any script on the page
            console.warn('API key stored in sessionStorage. This is safe for demo, but not for production.');
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
    // 2. SPLASH FALLBACK – auto‑hide after 10 seconds if user didn't click
    // ============================================================
    setTimeout(() => {
        const splashEl = document.getElementById('splash');
        const mainEl = document.getElementById('main-content');
        if (splashEl && !splashEl.classList.contains('exit')) {
            splashEl.classList.add('exit');
            setTimeout(() => {
                splashEl.style.display = 'none';
                if (mainEl) mainEl.classList.add('visible');
            }, 1000);
        }
    }, 10000); // 10 seconds

    // ============================================================
    // 3. HAMBURGER MENU
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
    // 4. SMOOTH SCROLL
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
    // 5. LAUNCH DEMO – show/hide classifier and handle API key
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
    // 6. FILE UPLOAD & PREVIEW
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
    // 7. CLASSIFY
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
    // 8. HTTP error mapping
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
    // 9. RENDERING – DOM-based, safe (no innerHTML with dynamic data)
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

        // Clear previous content
        resultContent.innerHTML = '';

        // --- Top class ---
        const topDiv = document.createElement('div');
        topDiv.className = 'top-class';
        const label = document.createElement('span');
        label.className = 'top-label';
        label.textContent = 'Detected';
        const value = document.createElement('span');
        value.className = 'top-value';
        value.textContent = prettyLabel(topClass);
        topDiv.append(label, value);
        resultContent.appendChild(topDiv);

        // --- Meta row ---
        const metaRow = document.createElement('div');
        metaRow.className = 'meta-row';

        // Confidence
        const confItem = document.createElement('div');
        confItem.className = 'meta-item';
        const confLabel = document.createElement('span');
        confLabel.className = 'meta-label';
        confLabel.textContent = 'Confidence';
        const confVal = document.createElement('span');
        confVal.className = 'meta-value';
        confVal.textContent = (topScore * 100).toFixed(1) + '%';
        confItem.append(confLabel, confVal);
        metaRow.appendChild(confItem);

        // Inference time if available
        if (data.inference_ms != null) {
            const timeItem = document.createElement('div');
            timeItem.className = 'meta-item';
            const timeLabel = document.createElement('span');
            timeLabel.className = 'meta-label';
            timeLabel.textContent = 'Inference';
            const timeVal = document.createElement('span');
            timeVal.className = 'meta-value';
            timeVal.textContent = Math.round(data.inference_ms) + ' ms';
            timeItem.append(timeLabel, timeVal);
            metaRow.appendChild(timeItem);
        }
        resultContent.appendChild(metaRow);

        // --- Scores ---
        const scoresDiv = document.createElement('div');
        scoresDiv.className = 'scores';
        rows.forEach(([name, score]) => {
            const pct = (score * 100).toFixed(1);
            const isTop = name === topClass;
            const rowDiv = document.createElement('div');
            rowDiv.className = 'score-row' + (isTop ? ' is-top' : '');
            const head = document.createElement('div');
            head.className = 'score-head';
            const nameSpan = document.createElement('span');
            nameSpan.textContent = prettyLabel(name);
            const pctSpan = document.createElement('span');
            pctSpan.textContent = pct + '%';
            head.append(nameSpan, pctSpan);
            const track = document.createElement('div');
            track.className = 'score-track';
            track.setAttribute('role', 'progressbar');
            track.setAttribute('aria-label', prettyLabel(name) + ' confidence');
            track.setAttribute('aria-valuenow', pct);
            track.setAttribute('aria-valuemin', '0');
            track.setAttribute('aria-valuemax', '100');
            const fill = document.createElement('div');
            fill.className = 'score-fill';
            fill.style.width = pct + '%';
            track.appendChild(fill);
            rowDiv.append(head, track);
            scoresDiv.appendChild(rowDiv);
        });
        resultContent.appendChild(scoresDiv);

        // --- Feedback ---
        // Re-create the feedback container with the same ID and structure
        const fbWrap = document.createElement('div');
        fbWrap.id = 'feedbackWrap';
        fbWrap.className = 'feedback';
        fbWrap.hidden = true;

        const fbMeta = document.createElement('span');
        fbMeta.className = 'meta-label';
        fbMeta.textContent = 'Wrong prediction? Correct it';
        fbWrap.appendChild(fbMeta);

        const fbRow = document.createElement('div');
        fbRow.className = 'feedback-row';

        const fbLabel = document.createElement('label');
        fbLabel.className = 'sr-only';
        fbLabel.setAttribute('for', 'feedbackClass');
        fbLabel.textContent = 'Correct class';
        fbRow.appendChild(fbLabel);

        const fbSelect = document.createElement('select');
        fbSelect.className = 'feedback-select';
        fbSelect.id = 'feedbackClass';
        fbRow.appendChild(fbSelect);

        const fbBtn = document.createElement('button');
        fbBtn.type = 'button';
        fbBtn.className = 'btn-primary btn-secondary-dark btn-small';
        fbBtn.id = 'feedbackBtn';
        fbBtn.textContent = 'Send feedback';
        fbRow.appendChild(fbBtn);

        fbWrap.appendChild(fbRow);

        const fbNote = document.createElement('p');
        fbNote.className = 'feedback-note';
        fbNote.id = 'feedbackNote';
        fbNote.setAttribute('role', 'status');
        fbNote.setAttribute('aria-live', 'polite');
        fbWrap.appendChild(fbNote);

        resultContent.appendChild(fbWrap);

        // Store prediction ID and update feedback state
        lastPredictionId = data.id ?? null;
        if (lastPredictionId !== null && rows.length) {
            const classNames = modelClasses.length ? modelClasses : rows.map(([name]) => name);
            // Build options using the new select
            const select = document.getElementById('feedbackClass');
            if (select) {
                select.innerHTML = '';
                classNames.forEach(name => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = prettyLabel(name);
                    opt.selected = name === topClass;
                    select.appendChild(opt);
                });
            }
            const fbBtnNew = document.getElementById('feedbackBtn');
            if (fbBtnNew) fbBtnNew.disabled = false;
            fbWrap.hidden = false;
        } else {
            fbWrap.hidden = true;
        }

        // Re-bind feedback button event (since we recreated it)
        const newFeedbackBtn = document.getElementById('feedbackBtn');
        if (newFeedbackBtn) {
            // Remove any existing listeners (we only have one, but safe)
            // We'll just assign a new click handler
            newFeedbackBtn.onclick = sendFeedback;
        }
    }

    // ============================================================
    // 10. Feedback
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

    // ============================================================
    // 11. History
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

// ──────────────────────────────────────────────────────────────
//  CHATBOT WIDGET – appended per user request
// ──────────────────────────────────────────────────────────────
(function initChatbot() {
  const toggle = document.getElementById("chatToggle");
  const panel = document.getElementById("chatPanel");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const log = document.getElementById("chatLog");

  if (!toggle || !panel || !form) return;

  toggle.addEventListener("click", () => {
    const open = panel.hasAttribute("hidden");
    panel.toggleAttribute("hidden", !open);
    toggle.setAttribute("aria-expanded", String(open));
    if (open) input.focus();
  });

  function append(text, who) {
    const line = document.createElement("div");
    line.className = `chat-msg chat-msg--${who}`;
    line.textContent = text;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    append(message, "user");
    input.value = "";

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      append(data.reply, "bot");
    } catch (err) {
      append("Sorry, I couldn't reach the analytics service.", "bot");
    }
  });
})();