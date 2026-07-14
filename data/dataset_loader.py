"""
data/dataset_loader.py
Loads and preprocesses wildlife images from data/raw/{Species}/ directories.
Returns train / val / test splits as float32 numpy arrays (pixel values 0–255).
"""
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

SPECIES = ['Tiger', 'Lion', 'Elephant', 'Zebra', 'Giraffe',
           'Wolf', 'Bear', 'Deer', 'Leopard', 'Eagle']
IMG_SIZE = 224


def load_dataset(data_dir: str = 'data/raw', img_size: int = IMG_SIZE,
                 verbose: bool = True) -> tuple:
    """
    Scan data_dir/{Species}/ for JPG/PNG images.
    Returns:
        X : np.ndarray  shape (N, img_size, img_size, 3), float32, values 0-255
        y : np.ndarray  shape (N,), int32  — class indices matching SPECIES list
    """
    X, y = [], []
    data_path = Path(data_dir)

    for idx, species in enumerate(SPECIES):
        species_dir = data_path / species
        if not species_dir.exists():
            if verbose:
                print(f'  [WARN] Missing directory: {species_dir}')
            continue

        img_paths = sorted(
            list(species_dir.glob('*.jpg')) +
            list(species_dir.glob('*.jpeg')) +
            list(species_dir.glob('*.png'))
        )

        loaded = 0
        for p in img_paths:
            try:
                img = Image.open(p).convert('RGB').resize((img_size, img_size))
                X.append(np.array(img, dtype=np.float32))
                y.append(idx)
                loaded += 1
            except Exception as e:
                if verbose:
                    print(f'  [WARN] Cannot load {p}: {e}')

        if verbose:
            print(f'  {species}: {loaded} images')

    if not X:
        raise FileNotFoundError(
            f'No images found under {data_path}. '
            'Run: python data/sample_data_generator.py'
        )

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def get_splits(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> dict:
    """
    Split data into train / val / test.
    Returns dict with keys 'train', 'val', 'test', each containing (X, y) tuple.
    """
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # val_size is fraction of the *remaining* train+val pool
    adj_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=adj_val, stratify=y_tv, random_state=random_state
    )
    return {
        'train': (X_train, y_train),
        'val':   (X_val,   y_val),
        'test':  (X_test,  y_test),
    }
