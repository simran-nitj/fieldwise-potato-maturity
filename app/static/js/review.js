document.addEventListener('DOMContentLoaded', () => {
  const app = document.getElementById('review-app');
  if (!app) return;

  const el = (id) => document.getElementById(id);
  const canvas = el('annotation-canvas');
  const ctx = canvas.getContext('2d');
  const frame = el('canvas-frame');
  const colors = { 0: '#65a45b', 1: '#dda33e', 2: '#bd6655' };
  const classNames = { 0: 'Vegetative', 1: 'Tuber Bulking', 2: 'Maturity' };
  const state = { filter: 'pending', query: '', items: [], index: -1, item: null, image: null, view: 'annotated', zoom: 1, editing: false, draft: [], selected: -1, pointer: null };

  const encodePath = (value) => encodeURIComponent(value);
  const queryUrl = (base, rel) => `${base}?image_path=${encodePath(rel)}`;
  const setMessage = (message, kind = '') => {
    const node = el('review-message');
    node.textContent = message || '';
    node.className = `review-message ${kind}`;
  };
  const updateCounts = (counts) => {
    Object.entries(counts || {}).forEach(([key, value]) => {
      document.querySelectorAll(`[data-count="${key}"]`).forEach((node) => { node.textContent = key === 'progress_percent' ? Number(value).toFixed(1) : value; });
    });
    if (counts) el('review-progress-fill').style.width = `${counts.progress_percent}%`;
  };
  const statusLabel = (status) => ({ pending: 'Pending', approved: 'Approved', rejected: 'Rejected', edited: 'Edited', approved_empty: 'Approved empty' })[status] || status;
  const apiJson = async (url, options = {}) => {
    const response = await fetch(url, { headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }, ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  };

  const fetchQueue = async (preferred = null, fallbackIndex = 0) => {
    setMessage('Loading review queue…');
    const params = new URLSearchParams({ status: state.filter, q: state.query });
    const data = await apiJson(`/review/api/items?${params}`);
    state.items = data.items;
    updateCounts(data.counts);
    el('queue-count').textContent = `${state.items.length} image${state.items.length === 1 ? '' : 's'}`;
    if (!state.items.length) {
      state.index = -1;
      state.item = null;
      state.image = null;
      el('queue-current-label').textContent = 'No images match this filter.';
      renderEmpty('No images in this queue', state.query ? 'Try a different filename search.' : 'Choose another review status or clear the search.');
      setMessage('');
      return;
    }
    let index = preferred ? state.items.findIndex((item) => item.image_path === preferred) : -1;
    if (index < 0) index = Math.min(Math.max(0, fallbackIndex), state.items.length - 1);
    await loadItem(index);
  };

  const renderEmpty = (title, detail) => {
    canvas.width = 800;
    canvas.height = 480;
    canvas.style.width = '100%';
    canvas.style.height = 'auto';
    ctx.fillStyle = '#f5f7f3';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#59705e';
    ctx.textAlign = 'center';
    ctx.font = '600 24px sans-serif';
    ctx.fillText(title, 400, 220);
    ctx.font = '16px sans-serif';
    ctx.fillStyle = '#87938a';
    ctx.fillText(detail, 400, 254);
    el('canvas-empty').hidden = true;
    el('item-filename').textContent = title;
    el('item-position').textContent = '—';
    el('item-box-count').textContent = '';
    el('item-status').textContent = '';
    el('prediction-list').replaceChildren();
    el('prediction-summary').textContent = '—';
    el('no-predictions-note').hidden = true;
    el('approve-button').disabled = true;
    el('reject-button').disabled = true;
    el('edit-button').disabled = true;
    el('approve-empty-button').hidden = true;
    el('box-editor').hidden = true;
  };

  const loadItem = async (index) => {
    state.index = index;
    const entry = state.items[index];
    if (!entry) return;
    setMessage('Loading image and labels…');
    try {
      const data = await apiJson(queryUrl('/review/api/item', entry.image_path));
      const image = new Image();
      image.onload = () => {
        state.item = data;
        state.image = image;
        state.editing = false;
        state.zoom = 1;
        state.view = 'annotated';
        state.selected = -1;
        state.draft = structuredClone(data.boxes || []);
        document.querySelectorAll('[data-view]').forEach((button) => button.classList.toggle('selected', button.dataset.view === state.view));
        renderItem();
        setMessage('');
      };
      image.onerror = () => { setMessage('The original image could not be loaded. Check the pre-annotation source files.', 'error'); };
      image.src = data.original_url;
    } catch (error) {
      setMessage(error.message, 'error');
    }
  };

  const bboxPixels = (box) => {
    const width = state.image.naturalWidth;
    const height = state.image.naturalHeight;
    return { x: (box.cx - box.width / 2) * width, y: (box.cy - box.height / 2) * height, w: box.width * width, h: box.height * height };
  };

  const visibleBoxes = () => state.editing ? state.draft : (state.view === 'annotated' ? state.item?.predicted_boxes || [] : []);

  const draw = () => {
    if (!state.image) return;
    const image = state.image;
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const fitWidth = Math.max(1, Math.min(width, frame.clientWidth - 20));
    const cssWidth = Math.max(1, fitWidth * state.zoom);
    const cssHeight = Math.max(1, cssWidth * height / width);
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);

    const boxes = visibleBoxes();
    const line = Math.max(2, width / cssWidth * 2.2);
    const fontSize = Math.max(12, width / cssWidth * 13);
    boxes.forEach((box, index) => {
      const b = bboxPixels(box);
      const color = colors[box.class_id] || '#5e7864';
      ctx.save();
      ctx.strokeStyle = index === state.selected && state.editing ? '#27a7dc' : color;
      ctx.lineWidth = index === state.selected && state.editing ? line * 1.5 : line;
      ctx.setLineDash(box.manual ? [line * 3, line * 2] : []);
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.setLineDash([]);
      const conf = box.confidence == null ? (box.manual ? 'manual' : 'reviewed') : `${(box.confidence * 100).toFixed(1)}% model score`;
      const label = `${classNames[box.class_id]} · ${conf}`;
      ctx.font = `600 ${fontSize}px sans-serif`;
      const labelWidth = ctx.measureText(label).width + fontSize;
      const labelHeight = fontSize * 1.65;
      const labelY = Math.max(labelHeight, b.y);
      ctx.fillStyle = color;
      ctx.fillRect(b.x, labelY - labelHeight, labelWidth, labelHeight);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, b.x + fontSize / 2, labelY - fontSize * .35);
      if (state.editing && index === state.selected) {
        const handle = line * 3;
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = '#148dc1';
        ctx.lineWidth = line;
        ctx.fillRect(b.x + b.w - handle / 2, b.y + b.h - handle / 2, handle, handle);
        ctx.strokeRect(b.x + b.w - handle / 2, b.y + b.h - handle / 2, handle, handle);
      }
      ctx.restore();
    });
    el('zoom-label').textContent = `${Math.round(state.zoom * 100)}%`;
  };

  const renderPredictions = () => {
    const list = el('prediction-list');
    list.replaceChildren();
    const boxes = state.item?.predicted_boxes || [];
    el('prediction-summary').textContent = `${boxes.length} object${boxes.length === 1 ? '' : 's'} · ${state.item?.metadata?.confidence_threshold == null ? 'threshold unavailable' : `confidence threshold ${state.item.metadata.confidence_threshold}`}`;
    el('no-predictions-note').hidden = boxes.length !== 0;
    el('approve-empty-button').hidden = boxes.length !== 0;
    boxes.forEach((box, index) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'prediction-row';
      const marker = document.createElement('span');
      marker.className = 'prediction-color';
      marker.style.background = colors[box.class_id];
      const name = document.createElement('strong');
      name.textContent = box.class_name;
      const confidence = document.createElement('small');
      confidence.textContent = box.confidence == null ? 'score n/a' : `${(box.confidence * 100).toFixed(1)}% confidence`;
      row.append(marker, name, confidence);
      row.addEventListener('click', () => {
        state.view = 'annotated';
        state.selected = index;
        document.querySelectorAll('[data-view]').forEach((button) => button.classList.toggle('selected', button.dataset.view === state.view));
        draw();
      });
      list.append(row);
    });
  };

  const renderBoxEditor = () => {
    const container = el('box-list');
    container.replaceChildren();
    state.draft.forEach((box, index) => {
      const row = document.createElement('div');
      row.className = `box-row ${index === state.selected ? 'selected' : ''}`;
      const title = document.createElement('button');
      title.type = 'button';
      title.className = 'box-row-select';
      title.textContent = `Box ${index + 1}${box.manual ? ' · manual' : ''}`;
      title.addEventListener('click', () => { state.selected = index; renderBoxEditor(); draw(); });
      const select = document.createElement('select');
      select.setAttribute('aria-label', `Class for box ${index + 1}`);
      Object.entries(classNames).forEach(([id, name]) => {
        const option = document.createElement('option');
        option.value = id; option.textContent = name; option.selected = Number(id) === box.class_id; select.append(option);
      });
      select.addEventListener('change', () => { box.class_id = Number(select.value); box.class_name = classNames[box.class_id]; box.manual = true; state.selected = index; renderBoxEditor(); draw(); });
      const detail = document.createElement('small');
      detail.textContent = box.confidence == null ? 'corrected/manual label' : `original score ${(box.confidence * 100).toFixed(1)}%`;
      row.append(title, select, detail);
      container.append(row);
    });
    el('delete-box').disabled = state.selected < 0 || state.selected >= state.draft.length;
  };

  const renderItem = () => {
    if (!state.item || !state.image) return;
    const item = state.item;
    el('item-filename').textContent = item.filename;
    el('item-position').textContent = `${state.index + 1} of ${state.items.length}`;
    el('item-box-count').textContent = `${item.detection_count} predicted object${item.detection_count === 1 ? '' : 's'}`;
    const badge = el('item-status');
    badge.textContent = statusLabel(item.review_status);
    badge.className = `review-status-badge status-${item.review_status}`;
    el('image-dimensions').textContent = `${state.image.naturalWidth} × ${state.image.naturalHeight}`;
    el('canvas-view-caption').textContent = state.editing ? 'Editing corrected boxes · model confidence is shown only as the original suggestion score.' : state.view === 'annotated' ? 'Predicted boxes shown · confidence is the detector score, not accuracy.' : 'Original image only · no box overlay.';
    el('previous-button').disabled = state.index <= 0;
    el('next-button').disabled = state.index >= state.items.length - 1;
    el('edit-button').disabled = false;
    el('approve-button').disabled = item.review_status === 'edited';
    el('approve-button').title = item.review_status === 'edited' ? 'Corrected labels are saved as Edited.' : 'Approve original predictions (A)';
    el('reject-button').disabled = false;
    el('approve-empty-button').hidden = item.detection_count !== 0;
    el('box-editor').hidden = !state.editing;
    renderPredictions();
    if (state.editing) renderBoxEditor();
    draw();
  };

  const enterEditMode = () => {
    if (!state.item) return;
    if (state.editing) {
      state.editing = false;
      state.selected = -1;
      state.draft = structuredClone(state.item.boxes || []);
      el('box-editor').hidden = true;
      renderItem();
      setMessage('Unsaved edits were discarded.', 'info');
      return;
    }
    state.editing = true;
    state.view = 'annotated';
    state.draft = structuredClone(state.item.boxes || []);
    state.selected = -1;
    el('box-editor').hidden = false;
    document.querySelectorAll('[data-view]').forEach((button) => button.classList.toggle('selected', button.dataset.view === state.view));
    renderItem();
    setMessage('Edit mode: draw a box, drag a box to move it, drag the lower-right handle to resize, or change its class.', 'info');
  };

  const reviewer = () => {
    const name = el('reviewer-name').value.trim();
    if (!name) throw new Error('Enter a reviewer name or ID before saving a decision.');
    localStorage.setItem('fieldwise-reviewer', name);
    return name;
  };

  const saveAction = async (action, boxes = null) => {
    if (!state.item) return;
    const previous = state.item.image_path;
    const oldIndex = state.index;
    const payload = { action, reviewer: reviewer() };
    if (boxes) payload.boxes = boxes.map(({ class_id, cx, cy, width, height }) => ({ class_id, cx, cy, width, height }));
    setMessage(action === 'edit' ? 'Saving corrected annotations…' : 'Saving review decision…');
    try {
      const response = await apiJson(queryUrl('/review/api/action', previous), { method: 'POST', body: JSON.stringify(payload) });
      updateCounts(response.counts);
      state.editing = false;
      setMessage(`${statusLabel(response.item.review_status)} · saved by ${response.item.reviewer} at ${response.item.reviewed_at}.`, 'success');
      await fetchQueue(previous, oldIndex);
    } catch (error) {
      setMessage(error.message, 'error');
    }
  };

  const pointFromEvent = (event) => {
    const rect = canvas.getBoundingClientRect();
    return { x: Math.max(0, Math.min(state.image.naturalWidth, (event.clientX - rect.left) * state.image.naturalWidth / rect.width)), y: Math.max(0, Math.min(state.image.naturalHeight, (event.clientY - rect.top) * state.image.naturalHeight / rect.height)) };
  };
  const toNormalizedBox = (x1, y1, x2, y2, classId, extra = {}) => {
    const left = Math.max(0, Math.min(x1, x2));
    const top = Math.max(0, Math.min(y1, y2));
    const right = Math.min(state.image.naturalWidth, Math.max(x1, x2));
    const bottom = Math.min(state.image.naturalHeight, Math.max(y1, y2));
    return { class_id: Number(classId), class_name: classNames[Number(classId)], cx: (left + right) / 2 / state.image.naturalWidth, cy: (top + bottom) / 2 / state.image.naturalHeight, width: (right - left) / state.image.naturalWidth, height: (bottom - top) / state.image.naturalHeight, confidence: null, manual: true, ...extra };
  };
  const boxAt = (point) => {
    for (let i = state.draft.length - 1; i >= 0; i -= 1) {
      const b = bboxPixels(state.draft[i]);
      const slack = 10 * state.image.naturalWidth / canvas.clientWidth;
      if (point.x >= b.x - slack && point.x <= b.x + b.w + slack && point.y >= b.y - slack && point.y <= b.y + b.h + slack) return i;
    }
    return -1;
  };

  canvas.addEventListener('pointerdown', (event) => {
    if (!state.editing || !state.image) return;
    const point = pointFromEvent(event);
    const selectedBox = state.draft[state.selected];
    if (selectedBox) {
      const b = bboxPixels(selectedBox);
      const handleSlack = 14 * state.image.naturalWidth / canvas.clientWidth;
      if (Math.abs(point.x - (b.x + b.w)) <= handleSlack && Math.abs(point.y - (b.y + b.h)) <= handleSlack) {
        state.pointer = { kind: 'resize', start: point, original: structuredClone(selectedBox) };
        canvas.setPointerCapture(event.pointerId);
        return;
      }
    }
    const hit = boxAt(point);
    if (hit >= 0) {
      state.selected = hit;
      state.pointer = { kind: 'move', start: point, original: structuredClone(state.draft[hit]) };
    } else {
      const classId = Number(el('new-box-class').value);
      const box = toNormalizedBox(point.x, point.y, point.x, point.y, classId);
      state.draft.push(box);
      state.selected = state.draft.length - 1;
      state.pointer = { kind: 'draw', start: point, original: structuredClone(box) };
    }
    canvas.setPointerCapture(event.pointerId);
    renderBoxEditor(); draw();
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!state.pointer || !state.image) return;
    const point = pointFromEvent(event);
    const { kind, start, original } = state.pointer;
    const i = state.selected;
    if (kind === 'draw') state.draft[i] = toNormalizedBox(start.x, start.y, point.x, point.y, original.class_id);
    if (kind === 'resize') {
      const originalPixels = bboxPixels(original);
      state.draft[i] = toNormalizedBox(originalPixels.x, originalPixels.y, point.x, point.y, original.class_id, { confidence: original.confidence, manual: true });
    }
    if (kind === 'move') {
      const dx = (point.x - start.x) / state.image.naturalWidth;
      const dy = (point.y - start.y) / state.image.naturalHeight;
      const width = original.width, height = original.height;
      state.draft[i] = { ...original, cx: Math.max(width / 2, Math.min(1 - width / 2, original.cx + dx)), cy: Math.max(height / 2, Math.min(1 - height / 2, original.cy + dy)), manual: true };
    }
    draw();
  });
  const endPointer = () => {
    if (!state.pointer) return;
    const { kind } = state.pointer;
    const box = state.draft[state.selected];
    state.pointer = null;
    if (kind === 'draw' && box && (box.width * state.image.naturalWidth < 4 || box.height * state.image.naturalHeight < 4)) {
      state.draft.splice(state.selected, 1);
      state.selected = -1;
    }
    renderBoxEditor(); draw();
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  document.querySelectorAll('.review-filter').forEach((button) => button.addEventListener('click', async () => {
    state.filter = button.dataset.status;
    document.querySelectorAll('.review-filter').forEach((node) => node.classList.toggle('active', node === button));
    await fetchQueue();
  }));
  el('search-button').addEventListener('click', () => { state.query = el('review-search').value.trim(); fetchQueue(); });
  el('review-search').addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); state.query = el('review-search').value.trim(); fetchQueue(); } });
  el('next-button').addEventListener('click', () => { if (state.index < state.items.length - 1) loadItem(state.index + 1); });
  el('previous-button').addEventListener('click', () => { if (state.index > 0) loadItem(state.index - 1); });
  el('edit-button').addEventListener('click', enterEditMode);
  el('cancel-edits').addEventListener('click', enterEditMode);
  el('save-edits').addEventListener('click', () => saveAction('edit', state.draft));
  el('delete-box').addEventListener('click', () => { if (state.selected >= 0) { state.draft.splice(state.selected, 1); state.selected = -1; renderBoxEditor(); draw(); } });
  el('approve-button').addEventListener('click', () => saveAction('approve'));
  el('reject-button').addEventListener('click', () => saveAction('reject'));
  el('approve-empty-button').addEventListener('click', () => saveAction('approve_empty'));
  document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => {
    if (state.editing) return;
    state.view = button.dataset.view;
    document.querySelectorAll('[data-view]').forEach((node) => node.classList.toggle('selected', node === button));
    draw();
    el('canvas-view-caption').textContent = state.view === 'annotated' ? 'Predicted boxes shown · confidence is the detector score, not accuracy.' : 'Original image only · no box overlay.';
  }));
  const setZoom = (value) => { state.zoom = Math.max(.5, Math.min(4, value)); draw(); };
  el('zoom-in').addEventListener('click', () => setZoom(state.zoom + .25));
  el('zoom-out').addEventListener('click', () => setZoom(state.zoom - .25));
  el('zoom-reset').addEventListener('click', () => setZoom(1));
  new ResizeObserver(() => draw()).observe(frame);

  const storedReviewer = localStorage.getItem('fieldwise-reviewer');
  if (storedReviewer) el('reviewer-name').value = storedReviewer;
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && (target.matches('input, textarea, select') || target.isContentEditable)) return;
    const key = event.key.toLowerCase();
    if (event.key === 'Delete' || event.key === 'Backspace') {
      if (state.editing && state.selected >= 0) { state.draft.splice(state.selected, 1); state.selected = -1; renderBoxEditor(); draw(); event.preventDefault(); }
      return;
    }
    if (key === 'n') el('next-button').click();
    if (key === 'p') el('previous-button').click();
    if (key === 'e') enterEditMode();
    if (key === 'a' && !el('approve-button').disabled) el('approve-button').click();
    if (key === 'r' && !el('reject-button').disabled) el('reject-button').click();
    if (key === 's' && state.editing) el('save-edits').click();
  });

  fetchQueue(app.dataset.initialItem || null).catch((error) => { setMessage(error.message, 'error'); renderEmpty('Review could not start', error.message); });
});
