#!/usr/bin/env python3
"""Train and evaluate MobileNetV2 on plant disease dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_classes: int, img_size: int = 224) -> Model:
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True

    # Keep early backbone layers frozen and fine-tune later blocks.
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    inputs = Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history: tf.keras.callbacks.History, out_path: Path) -> None:
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["loss"], label="train_loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["accuracy"], label="train_accuracy")
    plt.plot(epochs, hist["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on plant disease dataset")
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/train"))
    parser.add_argument("--valid-dir", type=Path, default=Path("data/splits/valid"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "best_model.keras"

    train_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    valid_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_gen.flow_from_directory(
        args.train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    valid_data = valid_gen.flow_from_directory(
        args.valid_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    with (args.model_dir / "class_indices.json").open("w", encoding="utf-8") as fp:
        json.dump(train_data.class_indices, fp, indent=2)

    model = build_model(num_classes=train_data.num_classes, img_size=args.img_size)

    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(valid_data, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    save_training_plot(history, args.model_dir / "training_metrics.png")
    with (args.model_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history.history, fp, indent=2)

    print(f"Best model checkpoint path: {model_path}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()
