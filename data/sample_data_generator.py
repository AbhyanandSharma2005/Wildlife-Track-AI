"""
data/sample_data_generator.py
Generates a synthetic wildlife image dataset using PIL patterns.
Each species has a distinct color/pattern signature so CNN can learn real features.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path


# Per-species visual configuration
SPECIES_CONFIG = {
    'Tiger':   {'base': (210, 120, 50),  'pattern': 'stripes',  'detail': (15, 15, 15)},
    'Lion':    {'base': (200, 170, 95),  'pattern': 'solid',    'detail': None},
    'Elephant':{'base': (110, 112, 125), 'pattern': 'solid',    'detail': None},
    'Zebra':   {'base': (240, 240, 240), 'pattern': 'stripes',  'detail': (15, 15, 15)},
    'Giraffe': {'base': (220, 175, 80),  'pattern': 'patches',  'detail': (110, 65, 20)},
    'Wolf':    {'base': (130, 130, 148), 'pattern': 'gradient', 'detail': (60, 60, 70)},
    'Bear':    {'base': (75, 48, 28),    'pattern': 'solid',    'detail': None},
    'Deer':    {'base': (178, 130, 75),  'pattern': 'spots',    'detail': (235, 215, 175)},
    'Leopard': {'base': (200, 168, 75),  'pattern': 'spots',    'detail': (35, 18, 8)},
    'Eagle':   {'base': (95, 65, 35),    'pattern': 'gradient', 'detail': (240, 235, 220)},
}

SPECIES = list(SPECIES_CONFIG.keys())


def _fill_base(arr: np.ndarray, color: tuple, noise_std: float = 18.0) -> np.ndarray:
    """Fill array with base color + Gaussian noise for texture."""
    for c, val in enumerate(color):
        arr[:, :, c] = val
    arr = arr + np.random.normal(0, noise_std, arr.shape)
    return np.clip(arr, 0, 255)


def generate_synthetic_image(species: str, img_size: int = 224, seed: int = None) -> Image.Image:
    """Generate a single synthetic wildlife image for the given species."""
    if seed is not None:
        np.random.seed(seed)

    cfg = SPECIES_CONFIG[species]
    base_color = cfg['base']
    pattern = cfg['pattern']
    detail_color = cfg['detail']

    arr = np.zeros((img_size, img_size, 3), dtype=np.float32)
    arr = _fill_base(arr, base_color)
    img = Image.fromarray(arr.astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(img)

    if pattern == 'stripes' and detail_color:
        stripe_w = img_size // 7
        angle_shift = np.random.randint(0, img_size // 3)
        for i in range(-img_size, img_size * 2, stripe_w * 2):
            pts = [
                (i + angle_shift, 0),
                (i + stripe_w + angle_shift, 0),
                (i + stripe_w, img_size),
                (i, img_size),
            ]
            draw.polygon(pts, fill=detail_color)

    elif pattern == 'spots' and detail_color:
        num_spots = np.random.randint(18, 35)
        for _ in range(num_spots):
            cx = np.random.randint(10, img_size - 10)
            cy = np.random.randint(10, img_size - 10)
            rx = np.random.randint(6, 22)
            ry = np.random.randint(4, 16)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=detail_color)

    elif pattern == 'patches' and detail_color:
        for _ in range(np.random.randint(12, 22)):
            cx = np.random.randint(15, img_size - 15)
            cy = np.random.randint(15, img_size - 15)
            w  = np.random.randint(18, 48)
            h  = np.random.randint(14, 40)
            pts = [
                (cx, cy),
                (cx + w, cy + np.random.randint(-8, 8)),
                (cx + w + np.random.randint(-8, 8), cy + h),
                (cx + np.random.randint(-8, 8), cy + h),
            ]
            draw.polygon(pts, fill=detail_color)

    elif pattern == 'gradient' and detail_color:
        arr2 = np.array(img).astype(np.float32)
        for row in range(img_size):
            t = row / img_size
            for c in range(3):
                arr2[row, :, c] = base_color[c] * (1 - t) + detail_color[c] * t
        img = Image.fromarray(arr2.astype(np.uint8), 'RGB')

    # Soft-blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Vignette effect
    arr3 = np.array(img).astype(np.float32)
    Y, X = np.ogrid[:img_size, :img_size]
    cx, cy = img_size / 2, img_size / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1 - 0.4 * dist, 0.6, 1.0)[:, :, np.newaxis]
    arr3 = np.clip(arr3 * vignette, 0, 255).astype(np.uint8)

    return Image.fromarray(arr3, 'RGB')


def generate_dataset(output_dir: str = 'data/raw', samples_per_class: int = 100) -> None:
    """Generate the full synthetic dataset."""
    output_path = Path(output_dir)
    total = len(SPECIES) * samples_per_class

    print(f"Generating {total} synthetic images -> {output_path}")
    for species in SPECIES:
        species_dir = output_path / species
        species_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = generate_synthetic_image(species, seed=i * len(SPECIES) + SPECIES.index(species))
            img.save(species_dir / f'{species.lower()}_{i:04d}.jpg', quality=92)
        print(f"  * {species}: {samples_per_class} images")

    print(f"Dataset generation complete. Total: {total} images")


if __name__ == '__main__':
    generate_dataset(samples_per_class=100)
