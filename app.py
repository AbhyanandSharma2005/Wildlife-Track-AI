"""
app.py — Wildlife Track AI  |  Flask REST API
"""
import os
import sys
import json
import logging
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('WildlifeTrackAI')

UPLOAD_DIR = Path('uploads')
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

_training_lock = threading.Lock()
_training_active = False


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── Status ─────────────────────────────────────────────────────────────────────
@app.route('/api/status', methods=['GET'])
def api_status():
    """Return training status + model availability."""
    from model.predict import get_training_status
    status = get_training_status()
    status['training_active'] = _training_active
    return jsonify(status)


# ── Predict ────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Accept a wildlife image file (multipart/form-data key: 'image').
    Returns JSON with predictions from all models + face recognition.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided. Use form-data key "image".'}), 400

    file = request.files['image']
    if not file.filename or not _allowed(file.filename):
        return jsonify({
            'error': f'Unsupported file type. Use: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB guard
            return jsonify({'error': 'File too large. Max 20 MB.'}), 413

        from model.predict import predict_species
        result = predict_species(image_bytes)

        if result.get('status') == 'not_trained':
            return jsonify(result), 503

        return jsonify(result)

    except Exception as exc:
        logger.exception('Prediction failed')
        return jsonify({'error': str(exc)}), 500


# ── Train ──────────────────────────────────────────────────────────────────────
@app.route('/api/train', methods=['POST'])
def api_train():
    """Start model training in a background thread."""
    global _training_active

    with _training_lock:
        if _training_active:
            return jsonify({
                'status':  'already_running',
                'message': 'Training is already in progress. Poll /api/status for updates.',
            })
        _training_active = True

    body = request.get_json(silent=True) or {}
    epochs            = int(body.get('epochs', 12))
    samples_per_class = int(body.get('samples_per_class', 100))
    generate_demo     = bool(body.get('generate_demo', True))

    def _train():
        global _training_active
        try:
            from model.train import run_training
            from model.predict import invalidate_model_cache
            run_training(
                epochs=epochs,
                generate_demo=generate_demo,
                samples_per_class=samples_per_class,
            )
            invalidate_model_cache()
            logger.info('Training complete.')
        except Exception as exc:
            logger.error(f'Training failed: {exc}', exc_info=True)
        finally:
            _training_active = False

    t = threading.Thread(target=_train, daemon=True, name='TrainingThread')
    t.start()

    return jsonify({
        'status':  'started',
        'message': (
            f'Training started — {epochs} epochs, '
            f'{samples_per_class} images/class. '
            'Poll /api/status for progress.'
        ),
    })


# ── Comparison data ────────────────────────────────────────────────────────────
@app.route('/api/comparison', methods=['GET'])
def api_comparison():
    """Return chart-ready model comparison data."""
    try:
        from model.model_comparison import get_full_report
        report = get_full_report()
        if not report.get('metrics'):
            return jsonify({'error': 'No metrics yet. Train models first.'}), 404
        return jsonify(report)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── Register individual animal ─────────────────────────────────────────────────
@app.route('/api/register-animal', methods=['POST'])
def api_register_animal():
    """Register a named individual animal for face re-ID."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    animal_id = request.form.get('animal_id', '').strip()
    species   = request.form.get('species', 'Unknown').strip()

    if not animal_id:
        return jsonify({'error': 'animal_id is required.'}), 400

    try:
        import cv2
        import numpy as np
        raw = np.frombuffer(request.files['image'].read(), np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({'error': 'Cannot decode image.'}), 400

        from model.face_recognition_module import WildlifeFaceRecognizer
        result = WildlifeFaceRecognizer().register_animal(bgr, animal_id, species)
        return jsonify(result)
    except Exception as exc:
        logger.exception('Registration failed')
        return jsonify({'error': str(exc)}), 500


# ── Known animals list ─────────────────────────────────────────────────────────
@app.route('/api/known-animals', methods=['GET'])
def api_known_animals():
    try:
        from model.face_recognition_module import WildlifeFaceRecognizer
        animals = WildlifeFaceRecognizer().list_known_animals()
        return jsonify({'animals': animals, 'count': len(animals)})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── Health ─────────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'Wildlife Track AI'})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    logger.info(f'Starting Wildlife Track AI on port {port}')
    app.run(host='0.0.0.0', port=port, debug=debug)
