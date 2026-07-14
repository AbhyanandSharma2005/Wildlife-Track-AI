"""
model/train.py
Full training pipeline for Wildlife Track AI.

Steps:
  1. (Optional) Generate synthetic demo dataset
  2. Load + split dataset
  3. Train CNN (MobileNetV2 transfer learning)
  4. Extract 256-d CNN embeddings from training set
  5. Train 5 classical ML models on those embeddings
  6. Evaluate all 6 models on the held-out test set
  7. Save models + metrics JSON

Usage:
  python model/train.py                  # uses data/raw, generates if missing
  python model/train.py --epochs 20      # override epoch count
  python model/train.py --no-generate    # skip synthetic data generation
"""
import sys
import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

SAVED_MODELS_DIR = Path('saved_models')
DATA_DIR         = Path('data/raw')


# ── Training status (for API polling) ────────────────────────────────────────
_STATUS: dict = {'step': 'idle', 'progress': 0, 'message': ''}


def _update_status(step: str, progress: int, message: str) -> None:
    global _STATUS
    _STATUS.update({'step': step, 'progress': progress, 'message': message})
    status_file = SAVED_MODELS_DIR / 'training_status.json'
    SAVED_MODELS_DIR.mkdir(exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(_STATUS, f)
    print(f'[{progress:3d}%] {message}')


# ── Step helpers ──────────────────────────────────────────────────────────────

def _generate_data(samples_per_class: int = 100) -> None:
    _update_status('generating', 5, 'Generating synthetic demo dataset …')
    from data.sample_data_generator import generate_dataset
    generate_dataset(output_dir=str(DATA_DIR), samples_per_class=samples_per_class)
    _update_status('generating', 12, 'Dataset ready.')


def _load_data(verbose: bool = True) -> tuple:
    _update_status('loading', 15, 'Loading images …')
    from data.dataset_loader import load_dataset, get_splits
    X, y = load_dataset(str(DATA_DIR), verbose=verbose)
    splits = get_splits(X, y)
    X_train, y_train = splits['train']
    X_val,   y_val   = splits['val']
    X_test,  y_test  = splits['test']
    _update_status('loading', 22,
                   f'Data split — Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}')
    return X_train, y_train, X_val, y_val, X_test, y_test


def _train_cnn(X_train, y_train, X_val, y_val, epochs: int) -> tuple:
    """Train MobileNetV2 classifier. Returns (model, embedding_model, history)."""
    import tensorflow as tf
    from model.cnn_model import build_cnn_model, compile_model, get_callbacks

    _update_status('cnn_training', 25, 'Building CNN (MobileNetV2) …')
    model, embedding_model = build_cnn_model(fine_tune_layers=0)
    model = compile_model(model, learning_rate=1e-3)

    # Build augmentation layer
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.12),
    ])

    # Create tf.data pipelines
    def make_dataset(X, y, training=False, batch=32):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            ds = ds.shuffle(len(X), seed=42)
            ds = ds.map(lambda img, lbl: (aug(img, training=True), lbl),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds   = make_dataset(X_val,   y_val,   training=False)

    _update_status('cnn_training', 28, f'Training CNN for up to {epochs} epochs …')

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            progress = 28 + int((epoch + 1) / epochs * 35)
            acc  = logs.get('accuracy', 0) * 100
            vacc = logs.get('val_accuracy', 0) * 100
            _update_status(
                'cnn_training', progress,
                f'Epoch {epoch+1}/{epochs} — '
                f'acc:{acc:.1f}% val_acc:{vacc:.1f}%'
            )

    callbacks = get_callbacks() + [ProgressCallback()]
    history   = model.fit(train_ds, validation_data=val_ds,
                          epochs=epochs, callbacks=callbacks, verbose=0)
    return model, embedding_model, history


def _extract_embeddings(embedding_model, X: np.ndarray,
                        batch_size: int = 64) -> np.ndarray:
    """Extract CNN embeddings from image array (values 0-255)."""
    return embedding_model.predict(X, batch_size=batch_size, verbose=0)


def _train_classical(X_emb_train: np.ndarray,
                     y_train: np.ndarray) -> tuple:
    """Train all classical ML models. Returns (dict of pipelines, scaler)."""
    from sklearn.preprocessing import StandardScaler
    from model.supervised_models import get_all_models

    _update_status('classical_training', 65, 'Training classical ML models …')
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_emb_train)

    models  = get_all_models()
    trained = {}
    total   = len(models)
    for i, (name, (clf, safe_name)) in enumerate(models.items()):
        _update_status('classical_training',
                       65 + int((i + 1) / total * 20),
                       f'Training {name} …')
        clf.fit(X_sc, y_train)
        trained[name] = (clf, safe_name)

    return trained, scaler


def _evaluate_all(cnn_model, embedding_model, trained: dict,
                  scaler, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate CNN + all classical models. Returns dict of metric dicts."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    from data.dataset_loader import SPECIES

    _update_status('evaluating', 86, 'Evaluating models on test set …')
    results = {}

    # -- CNN --
    cnn_probs = cnn_model.predict(X_test, batch_size=64, verbose=0)
    cnn_preds = np.argmax(cnn_probs, axis=1)
    results['CNN (MobileNetV2)'] = _metric_dict(y_test, cnn_preds)

    # -- Classical --
    X_emb_test = _extract_embeddings(embedding_model, X_test)
    X_sc_test  = scaler.transform(X_emb_test)
    for name, (clf, _) in trained.items():
        preds = clf.predict(X_sc_test)
        results[name] = _metric_dict(y_test, preds)

    # Print summary
    print('\n  === Accuracy Comparison ===')
    for m, v in results.items():
        print(f'  {m:28s}  acc={v["accuracy"]:5.1f}%  f1={v["f1"]:5.1f}%')
    return results


def _metric_dict(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    return {
        'accuracy':  round(float(accuracy_score(y_true, y_pred))  * 100, 2),
        'precision': round(float(precision_score(y_true, y_pred,
                           average='weighted', zero_division=0)) * 100, 2),
        'recall':    round(float(recall_score(y_true, y_pred,
                           average='weighted', zero_division=0)) * 100, 2),
        'f1':        round(float(f1_score(y_true, y_pred,
                           average='weighted', zero_division=0)) * 100, 2),
    }


def _save_all(cnn_model, embedding_model, trained: dict,
              scaler, metrics: dict) -> None:
    SAVED_MODELS_DIR.mkdir(exist_ok=True)
    _update_status('saving', 92, 'Saving models …')

    cnn_model.save(SAVED_MODELS_DIR / 'cnn_model.keras')
    embedding_model.save(SAVED_MODELS_DIR / 'embedding_model.keras')

    with open(SAVED_MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    for name, (clf, safe_name) in trained.items():
        with open(SAVED_MODELS_DIR / f'{safe_name}.pkl', 'wb') as f:
            pickle.dump(clf, f)

    with open(SAVED_MODELS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    _update_status('complete', 100, 'Training complete! All models saved.')


# ── Public entry point ────────────────────────────────────────────────────────

def run_training(data_dir: str = str(DATA_DIR),
                 epochs: int = 12,
                 generate_demo: bool = True,
                 samples_per_class: int = 100) -> dict:
    """
    Orchestrate the full training pipeline.
    Returns metrics dict.
    """
    SAVED_MODELS_DIR.mkdir(exist_ok=True)

    # 1. Data
    if generate_demo and not DATA_DIR.exists():
        _generate_data(samples_per_class)
    X_train, y_train, X_val, y_val, X_test, y_test = _load_data()

    # 2. CNN
    cnn_model, embedding_model, _ = _train_cnn(X_train, y_train, X_val, y_val, epochs)

    # 3. Embeddings → Classical
    _update_status('embedding', 63, 'Extracting CNN embeddings for classical models …')
    X_emb_train = _extract_embeddings(embedding_model, X_train)

    # 4. Classical models
    trained, scaler = _train_classical(X_emb_train, y_train)

    # 5. Evaluate
    metrics = _evaluate_all(cnn_model, embedding_model, trained, scaler,
                            X_test, y_test)

    # 6. Save
    _save_all(cnn_model, embedding_model, trained, scaler, metrics)

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Wildlife Track AI models')
    parser.add_argument('--epochs',        type=int,  default=12)
    parser.add_argument('--samples',       type=int,  default=100,
                        help='Synthetic images per class (default 100)')
    parser.add_argument('--no-generate',   action='store_true',
                        help='Skip synthetic data generation')
    parser.add_argument('--demo',          action='store_true',
                        help='Run a quick demo training (2 epochs, 20 samples)')
    parser.add_argument('--data-dir',      default=str(DATA_DIR))
    args = parser.parse_args()

    epochs = 2 if args.demo else args.epochs
    samples = 20 if args.demo else args.samples

    run_training(
        data_dir=args.data_dir,
        epochs=epochs,
        generate_demo=not args.no_generate,
        samples_per_class=samples,
    )
