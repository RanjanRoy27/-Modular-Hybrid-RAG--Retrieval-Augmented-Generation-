/* === RAG Assistant v2 — Frontend with Sessions & File Management === */

const API_BASE = window.location.origin;

// ── DOM refs ────────────────────────────────────────────────────────────────
const messagesList = document.getElementById('messages-list');
const welcomeState = document.getElementById('welcome-state');
const welcomeSubtitle = document.getElementById('welcome-subtitle');
const suggestionChips = document.getElementById('suggestion-chips');
const questionInput = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const ingestBtn = document.getElementById('ingest-btn');
const ingestIcon = document.getElementById('ingest-icon');
const ingestLabel = document.getElementById('ingest-label');
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadLabel = document.getElementById('upload-label');
const filesList = document.getElementById('files-list');
const sessionsList = document.getElementById('sessions-list');
const sessionTitle = document.getElementById('session-title');
const sessionSourcePills = document.getElementById('session-source-pills');
const newChatBtn = document.getElementById('new-chat-btn');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const toast = document.getElementById('toast');
const chips = document.querySelectorAll('.chip');

// ── App State ────────────────────────────────────────────────────────────────
let activeSessionId = null;
let sessions = [];       // [{id, name, files, messages, created_at}]
let pendingFiles = [];       // files selected but not yet uploaded
let isStreaming = false;
let systemInitialized = false;

// ── Toast ────────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
    toast.textContent = msg;
    toast.className = `toast show ${type}`;
    setTimeout(() => { toast.className = 'toast'; }, 3500);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function fileIcon(name) {
    const ext = name.split('.').pop().toLowerCase();
    return ext === 'pdf' ? '📄' : ext === 'docx' ? '📝' : '📃';
}

function relativeTime(iso) {
    const d = new Date(iso);
    const diff = Date.now() - d.getTime();
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return d.toLocaleDateString();
}

function getSession(id) {
    return sessions.find(s => s.id === id);
}

// ── Status Check ─────────────────────────────────────────────────────────────
async function checkStatus() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        systemInitialized = data.initialized;
        if (data.initialized) {
            statusDot.className = 'status-dot-small online';
            statusText.textContent = 'Knowledge base ready';
        } else {
            statusDot.className = 'status-dot-small offline';
            statusText.textContent = 'Not indexed – add docs';
        }
    } catch {
        statusDot.className = 'status-dot-small offline';
        statusText.textContent = 'API offline';
        systemInitialized = false;
    }
}

// ── File List ────────────────────────────────────────────────────────────────
async function loadFiles() {
    try {
        const res = await fetch(`${API_BASE}/rag/files`);
        const data = await res.json();
        renderFilesList(data.files || []);
    } catch { /* silent */ }
}

function renderFilesList(files) {
    filesList.innerHTML = '';
    if (!files || files.length === 0) {
        filesList.innerHTML = '<div class="files-empty">No documents indexed yet.</div>';
        return;
    }
    files.forEach(f => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <span class="file-icon">${fileIcon(f.name)}</span>
            <span class="file-name" title="${f.name}">${f.name}</span>
            <button class="file-delete-btn" data-name="${f.name}" title="Remove file" aria-label="Delete ${f.name}">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </button>`;
        item.querySelector('.file-delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteFile(f.name);
        });
        filesList.appendChild(item);
    });
}

async function deleteFile(filename) {
    try {
        const res = await fetch(`${API_BASE}/rag/files/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Delete failed');
        showToast(`Removed ${filename}`, 'info');
        loadFiles();
    } catch (e) {
        showToast(`Could not delete: ${e.message}`, 'error');
    }
}

// ── Sessions ─────────────────────────────────────────────────────────────────
async function loadSessions() {
    try {
        const res = await fetch(`${API_BASE}/rag/sessions`);
        const data = await res.json();
        sessions = data.sessions || [];
        activeSessionId = data.active_session_id;
        renderSessionsList();
        if (activeSessionId) {
            const active = getSession(activeSessionId);
            if (active) loadSessionIntoUI(active);
        } else {
            showWelcome();
        }
    } catch { /* silent */ }
}

function renderSessionsList() {
    sessionsList.innerHTML = '';
    if (sessions.length === 0) {
        sessionsList.innerHTML = '<div class="sessions-empty">No chats yet. Upload documents and ask a question!</div>';
        return;
    }
    sessions.forEach(s => {
        const item = document.createElement('div');
        item.className = `session-item${s.id === activeSessionId ? ' active' : ''}`;
        item.dataset.id = s.id;

        // File pills (show first 2)
        const filePills = (s.files || []).slice(0, 2).map(f =>
            `<span class="session-file-pill">${fileIcon(f)} ${f.replace(/\.[^.]+$/, '')}</span>`
        ).join('');
        const extra = s.files && s.files.length > 2 ? `<span style="font-size:10px;color:var(--text-muted)">+${s.files.length - 2}</span>` : '';
        const msgCount = s.messages.length > 0 ? `${Math.floor(s.messages.length / 2)} Q&A` : 'Empty';

        item.innerHTML = `
            <div class="session-icon">💬</div>
            <div class="session-info">
                <div class="session-name" title="${s.name}">${s.name}</div>
                <div class="session-meta">
                    <span>${relativeTime(s.created_at)}</span>
                    <span>·</span>
                    <span>${msgCount}</span>
                </div>
                <div class="session-meta" style="margin-top:3px;">${filePills}${extra}</div>
            </div>
            <button class="session-delete-btn" data-id="${s.id}" title="Delete session" aria-label="Delete session">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </button>`;

        item.addEventListener('click', (e) => {
            if (e.target.closest('.session-delete-btn')) return;
            switchSession(s.id);
        });
        item.querySelector('.session-delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteSession(s.id);
        });
        sessionsList.appendChild(item);
    });
}

function loadSessionIntoUI(session) {
    // Update header
    sessionTitle.textContent = session.name;
    renderSourcePills(session.files || []);

    // Render messages
    messagesList.innerHTML = '';
    if (session.messages.length === 0) {
        showWelcome();
    } else {
        hideWelcome();
        // Re-render messages
        const history = session.messages;
        for (let i = 0; i < history.length; i++) {
            const msg = history[i];
            if (msg.role === 'human') {
                appendUserBubble(msg.content);
            } else {
                appendAIBubble(msg.content, msg.meta || {});
            }
        }
    }
    scrollToBottom();
}

function renderSourcePills(files) {
    sessionSourcePills.innerHTML = '';
    if (!files || files.length === 0) return;
    files.slice(0, 4).forEach(f => {
        const span = document.createElement('span');
        span.className = 'source-tag';
        span.innerHTML = `<span class="source-tag-icon">${fileIcon(f)}</span>${f}`;
        sessionSourcePills.appendChild(span);
    });
    if (files.length > 4) {
        const more = document.createElement('span');
        more.style.cssText = 'font-size:10.5px;color:var(--text-muted)';
        more.textContent = `+${files.length - 4} more`;
        sessionSourcePills.appendChild(more);
    }
}

async function switchSession(id) {
    if (id === activeSessionId) return;
    activeSessionId = id;
    // Mark active on server
    await fetch(`${API_BASE}/rag/sessions/active/${id}`, { method: 'PUT' }).catch(() => { });
    // Update UI
    renderSessionsList();
    const s = getSession(id);
    if (s) loadSessionIntoUI(s);
}

async function deleteSession(id) {
    try {
        await fetch(`${API_BASE}/rag/sessions/${id}`, { method: 'DELETE' });
        sessions = sessions.filter(s => s.id !== id);
        if (activeSessionId === id) {
            activeSessionId = sessions[0]?.id || null;
        }
        renderSessionsList();
        if (activeSessionId) {
            const active = getSession(activeSessionId);
            if (active) loadSessionIntoUI(active);
        } else {
            sessionTitle.textContent = 'New Chat';
            sessionSourcePills.innerHTML = '';
            messagesList.innerHTML = '';
            showWelcome();
        }
        showToast('Chat deleted', 'info');
    } catch {
        showToast('Could not delete chat', 'error');
    }
}

async function persistMessages() {
    if (!activeSessionId) return;
    const session = getSession(activeSessionId);
    if (!session) return;
    try {
        await fetch(`${API_BASE}/rag/sessions/${activeSessionId}/messages`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: session.messages })
        });
    } catch { /* silent, will retry next time */ }
}

// ── Welcome State ─────────────────────────────────────────────────────────────
function showWelcome() {
    welcomeState.style.display = '';
    welcomeState.style.opacity = '1';
    if (systemInitialized) {
        welcomeSubtitle.innerHTML = 'Ask a question below, or choose a chat from the sidebar.';
        suggestionChips.style.display = 'flex';
    } else {
        welcomeSubtitle.innerHTML = 'Upload documents and click <strong>Index &amp; Start New Chat</strong> to begin.';
        suggestionChips.style.display = 'none';
    }
}

function hideWelcome() {
    if (welcomeState.style.display !== 'none') {
        welcomeState.style.opacity = '0';
        welcomeState.style.transition = 'opacity 0.2s ease';
        setTimeout(() => { welcomeState.style.display = 'none'; }, 200);
    }
}

// ── Message Rendering ─────────────────────────────────────────────────────────
function scrollToBottom() {
    const c = document.getElementById('messages-container');
    c.scrollTo({ top: c.scrollHeight, behavior: 'smooth' });
}

function appendUserBubble(text) {
    const w = document.createElement('div');
    w.className = 'message user-message';
    w.innerHTML = `<div class="bubble-wrapper"><div class="bubble">${escapeHtml(text)}</div></div>
                   <div class="avatar">U</div>`;
    messagesList.appendChild(w);
    scrollToBottom();
}

function appendAIBubble(text, meta = {}) {
    const w = document.createElement('div');
    w.className = 'message ai-message';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    const bw = document.createElement('div');
    bw.className = 'bubble-wrapper';
    bw.appendChild(bubble);

    // Sources from stream response
    if (meta.sources && meta.sources.length) {
        const metaRow = document.createElement('div');
        metaRow.className = 'meta-row';
        meta.sources.forEach(src => {
            const pill = document.createElement('span');
            pill.className = 'source-pill';
            pill.innerHTML = `${fileIcon(src)} ${src}`;
            metaRow.appendChild(pill);
        });
        bw.appendChild(metaRow);
    }

    if (meta.latency_ms || meta.context_length) {
        const metaRow = document.createElement('div');
        metaRow.className = 'meta-row';
        if (meta.latency_ms) {
            const s = document.createElement('span');
            s.textContent = `⚡ ${(meta.latency_ms / 1000).toFixed(2)}s`;
            metaRow.appendChild(s);
        }
        if (meta.context_length) {
            const s = document.createElement('span');
            s.textContent = `${meta.context_length} chars retrieved`;
            metaRow.appendChild(s);
        }
        bw.appendChild(metaRow);
    }

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '✦';
    w.appendChild(avatar);
    w.appendChild(bw);
    messagesList.appendChild(w);
    scrollToBottom();
    return bubble;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Thinking Indicator ────────────────────────────────────────────────────────
function showThinking() {
    const w = document.createElement('div');
    w.className = 'message ai-message';
    w.id = 'thinking-msg';
    w.innerHTML = `<div class="avatar">✦</div>
        <div class="bubble-wrapper">
            <div class="thinking-indicator">
                <div class="thinking-dot"></div><div class="thinking-dot"></div><div class="thinking-dot"></div>
            </div>
        </div>`;
    messagesList.appendChild(w);
    scrollToBottom();
}

function removeThinking() {
    document.getElementById('thinking-msg')?.remove();
}

// ── Streaming Answer ──────────────────────────────────────────────────────────
async function streamAnswer(question) {
    const session = getSession(activeSessionId);
    const history = session ? session.messages : [];

    showThinking();
    sendBtn.disabled = true;
    isStreaming = true;

    let fullAnswer = '';
    let collectedMeta = {};
    let collectedSources = [];
    let bubble = null;
    let cursor = null;

    try {
        const res = await fetch(`${API_BASE}/rag/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                history,
                session_id: activeSessionId
            })
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        removeThinking();

        // Create streaming bubble
        const w = document.createElement('div');
        w.className = 'message ai-message';
        const bw = document.createElement('div');
        bw.className = 'bubble-wrapper';
        bubble = document.createElement('div');
        bubble.className = 'bubble';
        cursor = document.createElement('span');
        cursor.className = 'stream-cursor';
        bubble.appendChild(cursor);
        bw.appendChild(bubble);
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = '✦';
        w.appendChild(avatar);
        w.appendChild(bw);
        messagesList.appendChild(w);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const raw = line.slice(6).trim();
                if (!raw || raw === '[DONE]') continue;
                try {
                    const parsed = JSON.parse(raw);
                    if (parsed.type === 'token') {
                        fullAnswer += parsed.content;
                        bubble.textContent = fullAnswer;
                        bubble.appendChild(cursor);
                        scrollToBottom();
                    } else if (parsed.type === 'sources') {
                        collectedSources = parsed.sources;
                    } else if (parsed.type === 'metadata') {
                        collectedMeta = parsed;
                    } else if (parsed.type === 'error') {
                        throw new Error(parsed.content);
                    }
                } catch (parseErr) {
                    if (parseErr.message && !parseErr.message.includes('JSON')) throw parseErr;
                }
            }
        }

        cursor?.remove();

        // Add source pills and metadata
        if (collectedSources.length) {
            const mr = document.createElement('div');
            mr.className = 'meta-row';
            collectedSources.forEach(src => {
                const pill = document.createElement('span');
                pill.className = 'source-pill';
                pill.innerHTML = `${fileIcon(src)} ${src}`;
                mr.appendChild(pill);
            });
            bw.appendChild(mr);
        }
        if (collectedMeta.latency_ms || collectedMeta.context_length) {
            const mr = document.createElement('div');
            mr.className = 'meta-row';
            if (collectedMeta.latency_ms) {
                const s = document.createElement('span');
                s.textContent = `⚡ ${(collectedMeta.latency_ms / 1000).toFixed(2)}s`;
                mr.appendChild(s);
            }
            if (collectedMeta.context_length) {
                const s = document.createElement('span');
                s.textContent = `${collectedMeta.context_length} chars retrieved`;
                mr.appendChild(s);
            }
            bw.appendChild(mr);
        }

        // Persist to session
        if (fullAnswer && activeSessionId) {
            const sess = getSession(activeSessionId);
            if (sess) {
                sess.messages.push({ role: 'human', content: question });
                sess.messages.push({
                    role: 'ai', content: fullAnswer,
                    meta: { sources: collectedSources, ...collectedMeta }
                });
                renderSessionsList();  // update Q&A count
                await persistMessages();
            }
        }

    } catch (err) {
        removeThinking();
        cursor?.remove();
        appendAIBubble(`Sorry, I ran into an error: ${err.message}`);
        showToast(`Error: ${err.message}`, 'error');
    } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

// ── Submit ────────────────────────────────────────────────────────────────────
async function handleSubmit() {
    const question = questionInput.value.trim();
    if (!question || isStreaming) return;
    if (!systemInitialized) {
        showToast('Please index your documents first.', 'error');
        return;
    }

    questionInput.value = '';
    questionInput.style.height = 'auto';
    hideWelcome();
    appendUserBubble(question);

    // If no active session, create one
    if (!activeSessionId) {
        try {
            const res = await fetch(`${API_BASE}/rag/sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: question.slice(0, 40) })
            });
            const newSess = await res.json();
            sessions.unshift(newSess);
            activeSessionId = newSess.id;
            sessionTitle.textContent = newSess.name;
            renderSourcePills(newSess.files || []);
            renderSessionsList();
        } catch (e) {
            showToast('Could not create session', 'error');
        }
    }

    await streamAnswer(question);
}

// ── File Upload ───────────────────────────────────────────────────────────────
async function handleFileSelect(files) {
    if (!files || files.length === 0) return;
    pendingFiles = Array.from(files);
    const names = pendingFiles.map(f => f.name).join(', ');
    uploadLabel.textContent = `${pendingFiles.length} file(s) selected`;
    showToast(`${pendingFiles.length} file(s) ready. Click "Index & Start New Chat".`, 'info');
}

async function uploadPendingFiles() {
    if (pendingFiles.length === 0) return true;
    const fd = new FormData();
    pendingFiles.forEach(f => fd.append('files', f));
    try {
        const res = await fetch(`${API_BASE}/rag/upload`, { method: 'POST', body: fd });
        const data = await res.json();
        if (data.errors && data.errors.length) {
            showToast(`Some files failed: ${data.errors.join(', ')}`, 'error');
        }
        pendingFiles = [];
        uploadLabel.textContent = 'Drop or click to upload';
        return data.saved && data.saved.length > 0;
    } catch (e) {
        showToast(`Upload failed: ${e.message}`, 'error');
        return false;
    }
}

// ── Ingest ────────────────────────────────────────────────────────────────────
ingestBtn.addEventListener('click', async () => {
    ingestBtn.disabled = true;
    ingestIcon.outerHTML = '<div class="spinner" id="ingest-icon"></div>';
    ingestLabel.textContent = 'Uploading & Indexing...';

    try {
        // 1. Upload pending files
        if (pendingFiles.length > 0) {
            await uploadPendingFiles();
        }

        // 2. Trigger ingest
        const res = await fetch(`${API_BASE}/rag/ingest`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Ingest failed');

        showToast(`✓ ${data.files_loaded} files, ${data.chunks_created} chunks indexed`, 'success');

        // 3. Reload sessions and files
        await Promise.all([loadFiles(), loadSessions()]);
        await checkStatus();

        // 4. Switch to the new session that was auto-created
        if (data.session_id) {
            await switchSession(data.session_id);
        }

        showWelcome();

    } catch (e) {
        showToast(`Failed: ${e.message}`, 'error');
    } finally {
        ingestBtn.disabled = false;
        document.querySelector('.spinner')?.remove();
        ingestBtn.insertAdjacentHTML('afterbegin',
            `<svg id="ingest-icon" width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M23 4V10H17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M1 20V14H7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M3.51 9A9 9 0 0 1 20.49 15M20.49 15L23 10M20.49 15L17 20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`);
        ingestLabel.textContent = 'Index & Start New Chat';
    }
});

// ── New Chat Button ───────────────────────────────────────────────────────────
newChatBtn.addEventListener('click', async () => {
    if (!systemInitialized) {
        showToast('Index your documents first.', 'info');
        return;
    }
    try {
        const res = await fetch(`${API_BASE}/rag/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const newSess = await res.json();
        sessions.unshift(newSess);
        activeSessionId = newSess.id;
        renderSessionsList();
        loadSessionIntoUI(newSess);
        showWelcome();
        showToast('New chat started', 'info');
    } catch {
        showToast('Could not create new chat', 'error');
    }
});

// ── Session Title Rename ──────────────────────────────────────────────────────
sessionTitle.addEventListener('dblclick', () => {
    if (!activeSessionId) return;
    const current = sessionTitle.textContent;
    const input = document.createElement('input');
    input.value = current;
    input.style.cssText = 'background:var(--bg-glass);border:1px solid var(--accent-purple);border-radius:4px;color:inherit;font:inherit;font-size:14px;padding:1px 6px;width:220px;outline:none;';
    sessionTitle.replaceWith(input);
    input.focus();
    input.select();

    const save = async () => {
        const newName = input.value.trim() || current;
        input.replaceWith(sessionTitle);
        sessionTitle.textContent = newName;
        const sess = getSession(activeSessionId);
        if (sess) sess.name = newName;
        renderSessionsList();
        await fetch(`${API_BASE}/rag/sessions/${activeSessionId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName })
        }).catch(() => { });
    };

    input.addEventListener('blur', save);
    input.addEventListener('keydown', e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') { input.replaceWith(sessionTitle); } });
});

// ── Input Events ──────────────────────────────────────────────────────────────
sendBtn.addEventListener('click', handleSubmit);
questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
});
questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
});

fileInput.addEventListener('change', () => handleFileSelect(fileInput.files));
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    handleFileSelect(e.dataTransfer.files);
});

chips.forEach(chip => {
    chip.addEventListener('click', () => {
        questionInput.value = chip.dataset.question;
        handleSubmit();
    });
});

sidebarToggle.addEventListener('click', () => sidebar.classList.toggle('open'));
document.addEventListener('click', e => {
    if (window.innerWidth <= 768 && !sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
        sidebar.classList.remove('open');
    }
});

// ── Boot ──────────────────────────────────────────────────────────────────────
(async function init() {
    await checkStatus();
    await Promise.all([loadFiles(), loadSessions()]);
    if (!activeSessionId) showWelcome();
})();
