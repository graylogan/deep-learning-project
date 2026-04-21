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

    if use_augmentation:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.15),
            ],
            name="augmentation",
        )
        x = augmentation(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

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
