"""
data/augmentation.py
On-the-fly data augmentation helpers using TensorFlow / numpy.
"""
import numpy as np


def random_flip(images: np.ndarray) -> np.ndarray:
    """Randomly flip images horizontally."""
    mask = np.random.rand(len(images)) > 0.5
    images[mask] = images[mask, :, ::-1, :]
    return images


def random_brightness(images: np.ndarray, delta: float = 30.0) -> np.ndarray:
    """Add random brightness offset."""
    offsets = np.random.uniform(-delta, delta, (len(images), 1, 1, 1))
    return np.clip(images + offsets, 0, 255).astype(np.float32)


def random_contrast(images: np.ndarray, lower: float = 0.75, upper: float = 1.25) -> np.ndarray:
    """Scale contrast randomly per image."""
    factors = np.random.uniform(lower, upper, (len(images), 1, 1, 1))
    means = images.mean(axis=(1, 2, 3), keepdims=True)
    return np.clip((images - means) * factors + means, 0, 255).astype(np.float32)


def augment_batch(images: np.ndarray) -> np.ndarray:
    """
    Apply a random subset of augmentations to a batch of images.
    images: float32 array, shape (N, H, W, 3), values 0-255.
    Returns augmented copy.
    """
    images = images.copy()
    images = random_flip(images)
    images = random_brightness(images)
    images = random_contrast(images)
    return images


def build_tf_augmentation_layer():
    """
    Returns a Keras Sequential augmentation layer suitable for use
    inside a tf.data pipeline or model.
    """
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.12),
        tf.keras.layers.RandomContrast(0.12),
    ], name='data_augmentation')
