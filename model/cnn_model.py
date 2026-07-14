"""
model/cnn_model.py
MobileNetV2-based CNN for wildlife species classification.

Architecture:
  MobileNetV2 (ImageNet weights, frozen) →
  GlobalAveragePooling2D →
  Dense(512, relu) →
  Dropout(0.4) →
  Dense(256, relu, name='embeddings') →
  Dropout(0.3) →
  Dense(NUM_CLASSES, softmax)

The 'embeddings' layer output is used as features for classical ML models.
"""
import tensorflow as tf
from tensorflow import keras

# ── Constants ─────────────────────────────────────────────────────────────────
SPECIES = [
    'Tiger', 'Lion', 'Elephant', 'Zebra', 'Giraffe',
    'Wolf',  'Bear', 'Deer',     'Leopard', 'Eagle',
]
NUM_CLASSES = len(SPECIES)
IMG_SIZE    = 224
EMBED_DIM   = 256


def build_cnn_model(fine_tune_layers: int = 0) -> tuple:
    """
    Build the full classification model and a companion embedding extractor.

    Args:
        fine_tune_layers: Number of MobileNetV2 layers to unfreeze
                          (0 = feature-extraction only, >0 = fine-tuning).

    Returns:
        model          : Full Keras model (input → softmax species probs)
        embedding_model: Keras model (input → 256-d embedding vector)
    """
    # Base: MobileNetV2 pretrained on ImageNet
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False

    # Optionally unfreeze the last N layers for fine-tuning
    if fine_tune_layers > 0:
        for layer in base.layers[-fine_tune_layers:]:
            layer.trainable = True

    # Build model graph
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')

    # MobileNetV2 expects values in [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name='gap')(x)

    x = keras.layers.Dense(512, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           name='fc1')(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.Dropout(0.40, name='drop1')(x)

    embeddings = keras.layers.Dense(EMBED_DIM, activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(1e-4),
                                    name='embeddings')(x)
    x = keras.layers.Dropout(0.30, name='drop2')(embeddings)

    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax',
                                 name='species_output')(x)

    # Full classifier
    model = keras.Model(inputs=inputs, outputs=outputs, name='WildlifeCNN')

    # Embedding extractor (shares weights with classifier)
    embedding_model = keras.Model(
        inputs=inputs, outputs=embeddings, name='WildlifeEmbedder'
    )

    return model, embedding_model


def compile_model(model: keras.Model,
                  learning_rate: float = 1e-3) -> keras.Model:
    """Compile the classifier with Adam + sparse cross-entropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_callbacks(monitor: str = 'val_accuracy') -> list:
    """Standard training callbacks: EarlyStopping + ReduceLROnPlateau."""
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
