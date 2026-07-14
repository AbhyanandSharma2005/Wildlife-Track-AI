"""
model/predict.py
Unified inference module — runs a single image through all trained models
and returns predictions + face-recognition results.
"""
import io
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

SAVED_MODELS_DIR = Path('saved_models')

SPECIES = [
    'Tiger', 'Lion', 'Elephant', 'Zebra', 'Giraffe',
    'Wolf',  'Bear', 'Deer',     'Leopard', 'Eagle',
]

# Model file stems (must match train.py)
_CLASSICAL_STEMS = [
    ('Random Forest',      'Random_Forest'),
    ('SVM (RBF)',          'SVM_RBF'),
    ('KNN',                'KNN'),
    ('Gradient Boosting',  'Gradient_Boosting'),
    ('Logistic Regression','Logistic_Regression'),
]

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_cnn_model        = None
_embedding_model  = None
_classical_models = {}   # display_name → estimator
_scaler           = None
_models_loaded    = False


def _load_models() -> bool:
    """Load all saved models into module globals. Returns True on success."""
    global _cnn_model, _embedding_model, _classical_models, _scaler, _models_loaded
    if _models_loaded:
        return True

    cnn_path = SAVED_MODELS_DIR / 'cnn_model.keras'
    if not cnn_path.exists():
        return False

    import tensorflow as tf
    _cnn_model       = tf.keras.models.load_model(str(cnn_path))
    _embedding_model = tf.keras.models.load_model(
        str(SAVED_MODELS_DIR / 'embedding_model.keras')
    )

    scaler_path = SAVED_MODELS_DIR / 'scaler.pkl'
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            _scaler = pickle.load(f)

    _classical_models = {}
    for display_name, safe_name in _CLASSICAL_STEMS:
        p = SAVED_MODELS_DIR / f'{safe_name}.pkl'
        if p.exists():
            with open(p, 'rb') as f:
                _classical_models[display_name] = pickle.load(f)

    _models_loaded = True
    return True


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes → float32 RGB array, shape (224, 224, 3), values 0-255.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    return np.array(img, dtype=np.float32)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_species(image_bytes: bytes) -> dict:
    """
    Run uploaded image through all trained models.

    Returns:
        {
          "status": "success",
          "consensus_species": str,
          "consensus_votes": {species: vote_count},
          "model_predictions": {
              model_name: {
                  "species": str,
                  "confidence": float,
                  "all_probs": {species: float}
              }
          },
          "face_recognition": {
              "status": str,
              "individual_id": str,
              "confidence": float,
              "is_known": bool,
              "box": [x, y, w, h]
          }
        }
    """
    if not _load_models():
        return {
            'status': 'not_trained',
            'error':  'Models are not trained yet. Please click "Train Models" first.',
        }

    arr = _preprocess(image_bytes)              # (224, 224, 3) float32 0-255
    batch = arr[np.newaxis, ...]               # (1, 224, 224, 3)

    predictions = {}

    # ── CNN ──────────────────────────────────────────────────────────────────
    cnn_probs  = _cnn_model.predict(batch, verbose=0)[0]          # (10,)
    cnn_idx    = int(np.argmax(cnn_probs))
    predictions['CNN (MobileNetV2)'] = {
        'species':    SPECIES[cnn_idx],
        'confidence': round(float(cnn_probs[cnn_idx]) * 100, 1),
        'all_probs':  {s: round(float(p) * 100, 1)
                       for s, p in zip(SPECIES, cnn_probs)},
    }

    # ── Embedding → classical models ─────────────────────────────────────────
    emb   = _embedding_model.predict(batch, verbose=0)            # (1, 256)
    emb_sc = _scaler.transform(emb) if _scaler is not None else emb

    for name, clf in _classical_models.items():
        try:
            probs   = clf.predict_proba(emb_sc)[0]                # (10,)
            pred_idx = int(np.argmax(probs))
            predictions[name] = {
                'species':    SPECIES[pred_idx],
                'confidence': round(float(probs[pred_idx]) * 100, 1),
                'all_probs':  {s: round(float(p) * 100, 1)
                               for s, p in zip(SPECIES, probs)},
            }
        except Exception as e:
            predictions[name] = {'error': str(e)}

    # ── Consensus vote ────────────────────────────────────────────────────────
    votes = {}
    for res in predictions.values():
        sp = res.get('species')
        if sp:
            votes[sp] = votes.get(sp, 0) + 1
    consensus = max(votes, key=votes.get) if votes else 'Unknown'

    # ── Face / individual recognition ─────────────────────────────────────────
    face_result = _run_face_recognition(arr.astype(np.uint8), consensus)

    return {
        'status':             'success',
        'consensus_species':  consensus,
        'consensus_votes':    votes,
        'model_predictions':  predictions,
        'face_recognition':   face_result,
    }


def _run_face_recognition(image_rgb: np.ndarray,
                           species_hint: str = None) -> dict:
    try:
        from model.face_recognition_module import WildlifeFaceRecognizer
        recognizer  = WildlifeFaceRecognizer()
        image_bgr   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results     = recognizer.identify(image_bgr, species_hint=species_hint)
        if not results:
            return {'status': 'no_region', 'individual_id': 'Unknown',
                    'confidence': 0.0, 'is_known': False, 'box': [0, 0, 0, 0]}
        best = results[0]
        return {
            'status':        'success',
            'individual_id': best['individual_id'],
            'confidence':    best['confidence'],
            'is_known':      best['is_known'],
            'box':           list(best['box']),
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e),
                'individual_id': 'Unknown', 'is_known': False}


# ── Metadata helpers ──────────────────────────────────────────────────────────

def get_metrics() -> dict | None:
    """Return stored model comparison metrics (from metrics.json)."""
    p = SAVED_MODELS_DIR / 'metrics.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def get_training_status() -> dict:
    """Return training status dict (written by train.py)."""
    p = SAVED_MODELS_DIR / 'training_status.json'
    if p.exists():
        with open(p) as f:
            status = json.load(f)
    else:
        status = {'step': 'idle', 'progress': 0, 'message': 'No training started yet.'}

    cnn_ready = (SAVED_MODELS_DIR / 'cnn_model.keras').exists()
    return {
        'trained':        cnn_ready,
        'training_status': status,
        'metrics':         get_metrics(),
        'classical_count': len([
            p for _, sn in _CLASSICAL_STEMS
            if (SAVED_MODELS_DIR / f'{sn}.pkl').exists()
        ]),
    }


def invalidate_model_cache() -> None:
    """Force model reload on next predict call (call after training completes)."""
    global _models_loaded
    _models_loaded = False
