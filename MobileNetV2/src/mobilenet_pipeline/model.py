from __future__ import annotations

import tensorflow as tf


def build_model(
    image_size: int,
    num_classes: int,
    learning_rate: float,
    dropout_rate: float,
    use_augmentation: bool,
    base_trainable: bool,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs

    # MobileNetV2 pretrained weights expect one of the canonical sizes.
    # Strategy:
    # - If the requested `image_size` is one of the canonical sizes, use it
    #   directly so we don't resize (e.g., 128 or 224).
    # - If `image_size` is smaller than the smallest canonical (e.g., 32),
    #   upscale to the smallest canonical (96). This maps 32 -> 96.
    # - Otherwise, pick the smallest canonical >= image_size, or the max if
    #   none are larger.
    CANONICAL_SIZES = [96, 128, 160, 192, 224]
    if image_size in CANONICAL_SIZES:
        base_size = image_size
    elif image_size < CANONICAL_SIZES[0]:
        base_size = CANONICAL_SIZES[0]
    else:
        # find smallest canonical >= image_size
        larger = [s for s in CANONICAL_SIZES if s >= image_size]
        base_size = larger[0] if larger else CANONICAL_SIZES[-1]

    if base_size != image_size:
        x = tf.keras.layers.Resizing(base_size, base_size, interpolation="bilinear", name=f"resize_to_{base_size}")(x)

    if use_augmentation:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                # tf.keras.layers.RandomRotation(0.08),
                # tf.keras.layers.RandomZoom(0.15),
            ],
            name="augmentation",
        )
        x = augmentation(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(base_size, base_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable
    print(f"!!!! BASE MODEL trainable: {base_model.trainable}")

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
