"""
model/face_recognition_module.py
Wildlife Individual Face Recognition Module.

Pipeline:
  1. Detect animal face/head region via OpenCV
  2. Extract 256-d CNN embedding from the detected region
  3. Compare with known-animal database (cosine similarity)
  4. Return individual ID + confidence + bounding box

For demo: uses the trained embedding model.
Falls back to colour-histogram pseudo-embedding if CNN not yet trained.
"""
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

SAVED_MODELS_DIR = Path('saved_models')
KNOWN_ANIMALS_DB = SAVED_MODELS_DIR / 'known_animals.json'
SIMILARITY_THRESHOLD = 0.80
IMG_SIZE = 224


class WildlifeFaceRecognizer:
    """Individual animal recognition via CNN embedding + cosine similarity."""

    def __init__(self):
        self.known_animals: dict = self._load_db()
        self._embedding_model = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_db(self) -> dict:
        if KNOWN_ANIMALS_DB.exists():
            with open(KNOWN_ANIMALS_DB, 'r') as f:
                return json.load(f)
        return {}

    def _save_db(self) -> None:
        SAVED_MODELS_DIR.mkdir(exist_ok=True)
        with open(KNOWN_ANIMALS_DB, 'w') as f:
            json.dump(self.known_animals, f, indent=2)

    # ── CNN embedding ─────────────────────────────────────────────────────────

    def _get_embedding_model(self):
        """Lazy-load the CNN embedding model."""
        if self._embedding_model is not None:
            return self._embedding_model
        model_path = SAVED_MODELS_DIR / 'embedding_model.keras'
        if model_path.exists():
            import tensorflow as tf
            self._embedding_model = tf.keras.models.load_model(str(model_path))
        return self._embedding_model

    def _extract_embedding(self, image_rgb: np.ndarray) -> list:
        """
        Extract embedding from an RGB numpy image (uint8).
        Uses CNN if available, otherwise colour-histogram fallback.
        """
        model = self._get_embedding_model()
        if model is not None:
            img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
            pred = model.predict(img[np.newaxis, ...], verbose=0)
            return pred[0].tolist()

        # Fallback: 3-channel 8-bin colour histogram (normalised)
        img_small = cv2.resize(image_rgb, (64, 64))
        hist = cv2.calcHist(
            [img_small], [0, 1, 2], None,
            [8, 8, 8], [0, 256, 0, 256, 0, 256]
        ).flatten()
        total = hist.sum() + 1e-9
        return (hist / total).tolist()

    # ── Face / head detection ─────────────────────────────────────────────────

    def _detect_face_regions(self, image_bgr: np.ndarray) -> list:
        """
        Detect probable animal face / head bounding boxes.

        Strategy:
          1. Try OpenCV DNN (MobileNet-SSD on COCO) if model files present.
          2. Fallback: heuristic upper-centre crop (animals' heads are typically
             in the upper 60 % of the frame).

        Returns list of (x, y, w, h) tuples.
        """
        h, w = image_bgr.shape[:2]

        # Attempt DNN-based detection
        prototxt  = SAVED_MODELS_DIR / 'deploy.prototxt'
        caffemodel= SAVED_MODELS_DIR / 'mobilenet_ssd.caffemodel'
        if prototxt.exists() and caffemodel.exists():
            try:
                net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image_bgr, (300, 300)),
                    0.007843, (300, 300), 127.5
                )
                net.setInput(blob)
                detections = net.forward()
                boxes = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.40:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        boxes.append((
                            max(0, x1), max(0, y1),
                            min(w, x2) - max(0, x1),
                            min(h, y2) - max(0, y1),
                        ))
                if boxes:
                    return boxes
            except Exception:
                pass  # Fall through to heuristic

        # Heuristic: upper-centre crop
        face_w = int(w * 0.65)
        face_h = int(h * 0.60)
        face_x = (w - face_w) // 2
        face_y = int(h * 0.04)
        return [(face_x, face_y, face_w, face_h)]

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-10:
            return 0.0
        return float(np.dot(a, b) / denom)

    def identify(self, image_bgr: np.ndarray,
                 species_hint: str = None) -> list:
        """
        Run recognition on an image.

        Args:
            image_bgr   : OpenCV BGR numpy array.
            species_hint: Optional species label to narrow DB search.

        Returns:
            List of dicts, one per detected region:
              box           (x, y, w, h)
              individual_id str
              confidence    float (0-100)
              is_known      bool
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = self._detect_face_regions(image_bgr)

        results = []
        for box in boxes:
            x, y, bw, bh = box
            crop = image_rgb[y: y + bh, x: x + bw]
            if crop.size == 0:
                continue

            emb = self._extract_embedding(crop)
            best_id, best_sim = 'Unknown', 0.0

            for animal_id, data in self.known_animals.items():
                if species_hint and data.get('species') != species_hint:
                    continue
                sim = self._cosine_similarity(emb, data['embedding'])
                if sim > best_sim:
                    best_sim = sim
                    best_id  = animal_id

            is_known = best_sim >= SIMILARITY_THRESHOLD
            results.append({
                'box':           box,
                'individual_id': best_id if is_known else 'Unknown',
                'confidence':    round(best_sim * 100, 1),
                'is_known':      is_known,
            })

        return results

    def register_animal(self, image_bgr: np.ndarray,
                        animal_id: str, species: str) -> dict:
        """
        Register a new individual in the known-animals database.

        Args:
            image_bgr : OpenCV BGR image of the animal.
            animal_id : Unique identifier (e.g. 'Tiger_001').
            species   : Species label.

        Returns:
            dict with status and animal_id.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        emb = self._extract_embedding(image_rgb)
        self.known_animals[animal_id] = {
            'species':       species,
            'embedding':     emb,
            'registered_at': datetime.utcnow().isoformat(),
        }
        self._save_db()
        return {'status': 'registered', 'animal_id': animal_id, 'species': species}

    def list_known_animals(self) -> list:
        return [
            {'id': k, 'species': v['species'], 'registered_at': v.get('registered_at', '')}
            for k, v in self.known_animals.items()
        ]
