/**
 * static/js/app.js
 * Wildlife Track AI — Frontend Logic
 *
 * Handles:
 *  - Drag & drop / file upload
 *  - Image preview with face-recognition bounding box overlay
 *  - /api/predict fetch + result rendering
 *  - /api/train polling & progress bar
 *  - Chart.js bar chart + radar chart for model comparison
 *  - Toast notifications
 */

/* ── Species Emoji Map ─────────────────────────────────────────────────────── */
const SPECIES_EMOJI = {
  Tiger:    '🐯', Lion:      '🦁', Elephant: '🐘', Zebra:   '🦓',
  Giraffe:  '🦒', Wolf:      '🐺', Bear:     '🐻', Deer:    '🦌',
  Leopard:  '🐆', Eagle:     '🦅', Unknown:  '❓',
};

/* ── Model colour palette ──────────────────────────────────────────────────── */
const MODEL_COLORS = {
  'CNN (MobileNetV2)':    'rgba(0, 255, 136, 0.85)',
  'Random Forest':        'rgba(124, 58, 237, 0.85)',
  'SVM (RBF)':            'rgba(245, 158, 11, 0.85)',
  'KNN':                  'rgba(239, 68, 68, 0.85)',
  'Gradient Boosting':    'rgba(59, 130, 246, 0.85)',
  'Logistic Regression':  'rgba(236, 72, 153, 0.85)',
};

/* ── State ─────────────────────────────────────────────────────────────────── */
let currentFile        = null;
let barChart           = null;
let radarChart         = null;
let trainingPollId     = null;

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const uploadZone    = document.getElementById('upload-zone');
const fileInput     = document.getElementById('file-input');
const previewWrap   = document.getElementById('preview-wrap');
const previewImg    = document.getElementById('preview-img');
const faceCanvas    = document.getElementById('face-canvas');
const btnAnalyse    = document.getElementById('btn-analyse');
const btnTrain      = document.getElementById('btn-train');
const progressWrap  = document.getElementById('progress-wrap');
const progressFill  = document.getElementById('progress-fill');
const progressLabel = document.getElementById('progress-label');
const resultsPanel  = document.getElementById('results-panel');
const statusDot     = document.getElementById('status-dot');
const statusLabel   = document.getElementById('status-label');

/* ══════════════════════════════════════════════════════════════════════════════
   ANIMATED BACKGROUND
══════════════════════════════════════════════════════════════════════════════ */
function initBackground() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx    = canvas.getContext('2d');

  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function randRange(a, b) { return a + Math.random() * (b - a); }

  function initParticles() {
    particles = [];
    const count = Math.floor((W * H) / 18000);
    for (let i = 0; i < count; i++) {
      particles.push({
        x:    Math.random() * W,
        y:    Math.random() * H,
        r:    randRange(0.5, 2.2),
        vx:   randRange(-0.15, 0.15),
        vy:   randRange(-0.25, -0.05),
        alpha:randRange(0.2, 0.7),
        hue:  randRange(140, 180),  // green-ish
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Subtle radial gradient overlay
    const grd = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W * 0.65);
    grd.addColorStop(0, 'rgba(0, 40, 25, 0.15)');
    grd.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, W, H);

    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 80%, 60%, ${p.alpha})`;
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;
      if (p.y < -5) { p.y = H + 5; p.x = Math.random() * W; }
      if (p.x < -5) p.x = W + 5;
      if (p.x > W + 5) p.x = -5;
    });

    requestAnimationFrame(draw);
  }

  resize();
  initParticles();
  draw();
  window.addEventListener('resize', () => { resize(); initParticles(); });
}

/* ══════════════════════════════════════════════════════════════════════════════
   UPLOAD HANDLING
══════════════════════════════════════════════════════════════════════════════ */
function initUpload() {
  // Click on zone → open file dialog
  uploadZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  // Drag & drop
  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please upload an image file (JPG, PNG, WebP)', 'error');
    return;
  }

  currentFile = file;
  btnAnalyse.disabled = false;

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewWrap.classList.add('visible');
    clearFaceCanvas();
  };
  reader.readAsDataURL(file);
}

/* ══════════════════════════════════════════════════════════════════════════════
   FACE RECOGNITION CANVAS OVERLAY
══════════════════════════════════════════════════════════════════════════════ */
function clearFaceCanvas() {
  const ctx = faceCanvas.getContext('2d');
  ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
}

function drawFaceBox(box, label, isKnown) {
  previewImg.onload = () => drawBox(box, label, isKnown);
  if (previewImg.complete) drawBox(box, label, isKnown);
}

function drawBox(box, label, isKnown) {
  const imgW  = previewImg.naturalWidth  || 224;
  const imgH  = previewImg.naturalHeight || 224;
  const dispW = previewImg.clientWidth;
  const dispH = previewImg.clientHeight;

  faceCanvas.width  = dispW;
  faceCanvas.height = dispH;

  const scaleX = dispW / imgW;
  const scaleY = dispH / imgH;

  const [bx, by, bw, bh] = box;
  const rx = bx * scaleX, ry = by * scaleY;
  const rw = bw * scaleX, rh = bh * scaleY;

  const ctx = faceCanvas.getContext('2d');
  ctx.clearRect(0, 0, dispW, dispH);

  // Glowing border
  const color = isKnown ? '#00ff88' : '#f59e0b';
  ctx.save();
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 12;
  ctx.lineWidth   = 2.5;
  ctx.strokeRect(rx, ry, rw, rh);
  ctx.restore();

  // Corner accents
  const cLen = Math.min(rw, rh) * 0.15;
  ctx.strokeStyle = color;
  ctx.lineWidth   = 3.5;
  ctx.lineCap     = 'square';
  [
    [[rx, ry + cLen], [rx, ry], [rx + cLen, ry]],
    [[rx + rw - cLen, ry], [rx + rw, ry], [rx + rw, ry + cLen]],
    [[rx, ry + rh - cLen], [rx, ry + rh], [rx + cLen, ry + rh]],
    [[rx + rw - cLen, ry + rh], [rx + rw, ry + rh], [rx + rw, ry + rh - cLen]],
  ].forEach(pts => {
    ctx.beginPath();
    ctx.moveTo(...pts[0]);
    ctx.lineTo(...pts[1]);
    ctx.lineTo(...pts[2]);
    ctx.stroke();
  });

  // Label
  const labelText = label || 'Unknown';
  ctx.font = 'bold 12px Inter, sans-serif';
  const tw = ctx.measureText(labelText).width + 16;
  ctx.fillStyle = 'rgba(0,0,0,0.75)';
  ctx.beginPath();
  ctx.roundRect(rx, ry - 26, tw, 22, 4);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.fillText(labelText, rx + 8, ry - 9);
}

/* ══════════════════════════════════════════════════════════════════════════════
   ANALYSE / PREDICT
══════════════════════════════════════════════════════════════════════════════ */
async function runAnalysis() {
  if (!currentFile) return;

  btnAnalyse.disabled = true;
  btnAnalyse.classList.add('loading');

  try {
    const formData = new FormData();
    formData.append('image', currentFile);

    const resp = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showToast(data.error || 'Prediction failed', 'error');
      return;
    }

    renderResults(data);
    showToast(`Detected: ${data.consensus_species}`, 'success');

  } catch (e) {
    showToast('Network error — is the server running?', 'error');
  } finally {
    btnAnalyse.disabled = false;
    btnAnalyse.classList.remove('loading');
  }
}

/* ══════════════════════════════════════════════════════════════════════════════
   RENDER RESULTS
══════════════════════════════════════════════════════════════════════════════ */
function renderResults(data) {
  resultsPanel.classList.remove('hidden');
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const preds   = data.model_predictions || {};
  const species = data.consensus_species || 'Unknown';
  const votes   = data.consensus_votes   || {};
  const face    = data.face_recognition  || {};

  /* ── Species Hero ──────────────────────────────────────────────────────── */
  const totalVotes = Object.values(votes).reduce((a, b) => a + b, 0);
  const cnnPred    = preds['CNN (MobileNetV2)'] || {};
  const cnnConf    = cnnPred.confidence || 0;

  document.getElementById('result-species-icon').textContent  = SPECIES_EMOJI[species] || '🐾';
  document.getElementById('result-species-name').textContent  = species;
  document.getElementById('result-species-conf').textContent  = `CNN confidence: ${cnnConf}%`;
  document.getElementById('result-votes').textContent         = `${votes[species] || 0}/${totalVotes}`;

  /* ── Face Recognition ──────────────────────────────────────────────────── */
  const faceIdEl   = document.getElementById('face-id');
  const faceConfEl = document.getElementById('face-conf');

  if (face.is_known) {
    faceIdEl.className = 'face-id-known';
    faceIdEl.textContent = `🔍 ${face.individual_id}`;
    faceConfEl.textContent = `Match confidence: ${face.confidence}%`;
  } else {
    faceIdEl.className = 'face-id-unknown';
    faceIdEl.textContent = 'Unknown Individual';
    faceConfEl.textContent = face.confidence
      ? `Best match: ${face.confidence}%`
      : 'No registered match found';
  }

  // Draw bounding box
  if (face.box && face.box.some(v => v > 0)) {
    drawFaceBox(face.box, face.individual_id || 'Unknown', face.is_known);
  }

  /* ── Per-Model Prediction Bars ──────────────────────────────────────────── */
  const predsContainer = document.getElementById('model-preds');
  predsContainer.innerHTML = '';

  const modelOrder = [
    'CNN (MobileNetV2)', 'Random Forest', 'SVM (RBF)',
    'KNN', 'Gradient Boosting', 'Logistic Regression',
  ];

  modelOrder.forEach(name => {
    const pred = preds[name];
    if (!pred) return;

    const color = MODEL_COLORS[name] || 'rgba(200,200,200,0.8)';
    const conf  = pred.confidence || 0;

    const row = document.createElement('div');
    row.className = 'model-pred-row';
    row.innerHTML = `
      <span class="model-pred-name">${name}</span>
      <div class="model-pred-bar-bg">
        <div class="model-pred-bar-fill" data-width="${conf}"
             style="background:${color};width:0%"></div>
      </div>
      <span class="model-pred-species">${pred.species || '—'}</span>
      <span class="model-pred-conf">${conf}%</span>
    `;
    predsContainer.appendChild(row);
  });

  // Animate bars after a tiny delay
  requestAnimationFrame(() => {
    document.querySelectorAll('.model-pred-bar-fill[data-width]').forEach(el => {
      const w = parseFloat(el.dataset.width);
      setTimeout(() => { el.style.width = Math.min(w, 100) + '%'; }, 50);
    });
  });

  /* ── Preview badge ──────────────────────────────────────────────────────── */
  const badge = document.getElementById('preview-badge');
  if (badge) {
    badge.textContent = species;
    badge.style.background = 'rgba(0,255,136,0.2)';
    badge.style.color = '#00ff88';
    badge.style.border = '1px solid rgba(0,255,136,0.3)';
  }

  /* ── Load comparison chart ──────────────────────────────────────────────── */
  fetchComparison();
}

/* ══════════════════════════════════════════════════════════════════════════════
   MODEL COMPARISON CHART
══════════════════════════════════════════════════════════════════════════════ */
async function fetchComparison() {
  try {
    const resp = await fetch('/api/comparison');
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.chart_data) {
      renderBarChart(data.chart_data.bar_chart);
      renderMetricsTable(data.chart_data.table, data.chart_data.winner);
    }
  } catch (e) {
    // Metrics not available yet — silently ignore
  }
}

function renderBarChart(barData) {
  const ctx = document.getElementById('bar-chart');
  if (!ctx || !barData) return;

  if (barChart) barChart.destroy();

  const chartDefaults = {
    font:   { family: 'Inter, sans-serif' },
    color:  '#8b9cb3',
  };

  barChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: barData.labels,
      datasets: [
        {
          label:           'Accuracy (%)',
          data:            barData.accuracy,
          backgroundColor: barData.colors,
          borderColor:     barData.colors.map(c => c.replace('0.85)', '1)')),
          borderWidth:     1,
          borderRadius:    6,
        },
        {
          label:           'F1 Score (%)',
          data:            barData.f1,
          backgroundColor: barData.colors.map(c => c.replace('0.85)', '0.35)')),
          borderColor:     barData.colors.map(c => c.replace('0.85)', '0.6)')),
          borderWidth:     1,
          borderRadius:    6,
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#8b9cb3', font: { family: 'Inter' } },
        },
        tooltip: {
          backgroundColor: 'rgba(6,13,16,0.92)',
          borderColor:     'rgba(255,255,255,0.08)',
          borderWidth:     1,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%`,
          },
        },
      },
      scales: {
        x: {
          ticks:   { color: '#8b9cb3', font: { family: 'Inter', size: 11 } },
          grid:    { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          beginAtZero: true,
          max:         100,
          ticks:   { color: '#8b9cb3', callback: v => v + '%', font: { family: 'Inter', size: 11 } },
          grid:    { color: 'rgba(255,255,255,0.06)' },
        },
      },
    },
  });
}

function renderMetricsTable(tableData, winner) {
  const tbody = document.getElementById('metrics-tbody');
  const winnerEl = document.getElementById('winner-model');
  if (!tbody) return;

  if (winnerEl) winnerEl.textContent = `🏆 Best Model: ${winner}`;

  tbody.innerHTML = '';
  tableData.forEach((row, i) => {
    const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : 'other';
    const rankEmoji = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : i + 1;
    const isBest    = (r) => r === Math.max(...tableData.map(d => d[Object.keys(r)[0]]));

    tbody.innerHTML += `
      <tr>
        <td><span class="rank-badge ${rankClass}">${rankEmoji}</span></td>
        <td>
          <div class="model-dot-label">
            <span class="model-dot" style="background:${row.color}"></span>
            ${row.model}
          </div>
        </td>
        <td><span class="metric-value ${i===0?'best':''}">${row.accuracy}%</span></td>
        <td><span class="metric-value">${row.precision}%</span></td>
        <td><span class="metric-value">${row.recall}%</span></td>
        <td><span class="metric-value">${row.f1}%</span></td>
      </tr>
    `;
  });
}

/* ══════════════════════════════════════════════════════════════════════════════
   TRAINING
══════════════════════════════════════════════════════════════════════════════ */
async function startTraining() {
  btnTrain.disabled = true;
  progressWrap.classList.add('visible');
  progressFill.style.width = '0%';
  progressLabel.textContent = 'Starting training …';

  try {
    const resp = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: 12, samples_per_class: 100, generate_demo: true }),
    });
    const data = await resp.json();

    if (data.status === 'started') {
      showToast('Training started! Generating synthetic data + training models…', 'info');
      pollTrainingStatus();
    } else {
      showToast(data.message || 'Could not start training', 'warning');
      btnTrain.disabled = false;
    }
  } catch (e) {
    showToast('Could not reach server', 'error');
    btnTrain.disabled = false;
  }
}

function pollTrainingStatus() {
  if (trainingPollId) clearInterval(trainingPollId);

  trainingPollId = setInterval(async () => {
    try {
      const resp   = await fetch('/api/status');
      const data   = await resp.json();
      const ts     = data.training_status || {};
      const prog   = ts.progress || 0;
      const msg    = ts.message  || 'Working …';

      progressFill.style.width = prog + '%';
      progressLabel.textContent = msg;

      updateStatusIndicator(data);

      if (ts.step === 'complete' || (!data.training_active && data.trained)) {
        clearInterval(trainingPollId);
        trainingPollId = null;
        btnTrain.disabled = false;
        showToast('Training complete! You can now analyse wildlife images.', 'success');
        progressLabel.textContent = '✓ Training complete!';
        fetchComparison();
      }

      if (ts.step === 'error') {
        clearInterval(trainingPollId);
        trainingPollId = null;
        btnTrain.disabled = false;
        showToast('Training failed: ' + msg, 'error');
      }
    } catch (e) { /* ignore */ }
  }, 2000);
}

function updateStatusIndicator(data) {
  if (!statusDot || !statusLabel) return;
  statusDot.className = 'status-dot';
  if (data.training_active) {
    statusDot.classList.add('training');
    statusLabel.textContent = 'Training…';
  } else if (data.trained) {
    statusDot.classList.add('trained');
    statusLabel.textContent = 'Models Ready';
  } else {
    statusLabel.textContent = 'Not Trained';
  }
}

/* ══════════════════════════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
══════════════════════════════════════════════════════════════════════════════ */
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toast-out 0.3s ease forwards';
    setTimeout(() => toast.remove(), 320);
  }, duration);
}

/* ══════════════════════════════════════════════════════════════════════════════
   INITIAL STATUS CHECK
══════════════════════════════════════════════════════════════════════════════ */
async function checkInitialStatus() {
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();
    updateStatusIndicator(data);

    if (data.trained) {
      fetchComparison();
    }

    if (data.training_active) {
      btnTrain.disabled = true;
      progressWrap.classList.add('visible');
      pollTrainingStatus();
    }
  } catch (e) { /* server may not be ready */ }
}

/* ══════════════════════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initBackground();
  initUpload();

  btnAnalyse.addEventListener('click', runAnalysis);
  btnTrain.addEventListener('click', startTraining);

  checkInitialStatus();

  // Refresh comparison chart periodically when training might be running
  setInterval(async () => {
    if (trainingPollId) return;  // already polling
    const resp = await fetch('/api/status').catch(() => null);
    if (!resp) return;
    const data = await resp.json();
    if (data.training_active) {
      pollTrainingStatus();
    }
  }, 10000);
});
