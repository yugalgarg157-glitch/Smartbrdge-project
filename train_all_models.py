#!/usr/bin/env python3
"""Train and compare CustomCNN, MobileNetV2, and EfficientNetB0 for malaria detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators(data_dir: Path, img_size: int, batch_size: int, validation_split: float):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=validation_split)

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_gen, val_gen


def create_custom_cnn(img_size: int) -> Model:
    model = Sequential(
        [
            Input(shape=(img_size, img_size, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


def create_transfer_model(model_name: str, img_size: int) -> Model:
    if model_name == "MobileNetV2":
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3),
        )
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3),
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    base_model.trainable = False

    inputs = Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs, name=model_name)


def compile_model(model: Model, learning_rate: float) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )


def train_model(model: Model, model_name: str, train_gen, val_gen, epochs: int, model_dir: Path):
    checkpoint_path = model_dir / f"{model_name}_best.keras"

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history, checkpoint_path


def evaluate_model(model_path: Path, val_gen):
    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(val_gen, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = val_gen.classes

    report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()), output_dict=True)
    val_acc = float(report["accuracy"])

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    return {
        "val_accuracy": val_acc,
        "roc_auc": roc_auc,
        "classification_report": report,
    }


def save_history_plot(history, model_name: str, out_path: Path) -> None:
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["accuracy"], label="Train Accuracy")
    plt.plot(epochs, hist["val_accuracy"], label="Val Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["loss"], label="Train Loss")
    plt.plot(epochs, hist["val_loss"], label="Val Loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare malaria classifiers")
    parser.add_argument("--data-dir", type=Path, default=Path("data/splits/train"))
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--best-model-name", type=str, default="best_malaria_model.keras")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen = create_generators(args.data_dir, args.img_size, args.batch_size, args.validation_split)

    models = {
        "CustomCNN": create_custom_cnn(args.img_size),
        "MobileNetV2": create_transfer_model("MobileNetV2", args.img_size),
        "EfficientNetB0": create_transfer_model("EfficientNetB0", args.img_size),
    }

    histories = {}
    results = {}
    checkpoints = {}

    for model_name, model in models.items():
        print("\n" + "=" * 50)
        print(f"Training {model_name}")
        print("=" * 50)

        compile_model(model, args.learning_rate)
        history, ckpt_path = train_model(model, model_name, train_gen, val_gen, args.epochs, args.model_dir)

        histories[model_name] = history.history
        checkpoints[model_name] = ckpt_path
        save_history_plot(history, model_name, args.model_dir / f"{model_name}_metrics.png")

        metrics = evaluate_model(ckpt_path, val_gen)
        results[model_name] = metrics

        print(f"Validation Accuracy ({model_name}): {metrics['val_accuracy']:.4f}")
        print(f"ROC AUC ({model_name}): {metrics['roc_auc']:.4f}")

    comparison_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Validation Accuracy": [results[m]["val_accuracy"] for m in results],
            "ROC AUC": [results[m]["roc_auc"] for m in results],
        }
    ).sort_values(by="Validation Accuracy", ascending=False)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]["Model"]
    best_checkpoint = checkpoints[best_model_name]
    final_best_path = args.model_dir / args.best_model_name

    tf.keras.models.load_model(best_checkpoint).save(final_best_path)
    print(f"\nBest model: {best_model_name}")
    print(f"Saved best model to: {final_best_path}")

    summary = {
        "comparison": comparison_df.to_dict(orient="records"),
        "best_model": best_model_name,
        "best_model_path": str(final_best_path),
        "class_indices": train_gen.class_indices,
        "results": results,
        "histories": histories,
    }

    with (args.model_dir / "comparison_results.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
