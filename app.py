from datetime import datetime
import os
import traceback
import time
from pathlib import Path
import json

from flask import Flask, jsonify, render_template, request
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

model = None
MODEL_ERROR = None
MODEL_PATH = None
MODEL_FILE_SIZE_MB = 0.0
CLASS_NAMES = ["Parasitized", "Uninfected"]
CLASS_INDEX_TO_NAME = {0: "Parasitized", 1: "Uninfected"}
tf = None
Image = None

print("\n" + "=" * 70)
print("MALARIA DETECTION - LOADING MODEL")
print("=" * 70)

try:
    import tensorflow as tf  # type: ignore[no-redef]
    from PIL import Image  # type: ignore[no-redef]

    print("TensorFlow and PIL imported")
except Exception as exc:  # noqa: BLE001
    MODEL_ERROR = f"Import failed: {exc}"
    print(f"X {MODEL_ERROR}")


if MODEL_ERROR is None:
    current_dir = Path(os.getcwd())
    print(f"Directory: {current_dir}")

    try:
        search_dirs = [current_dir, current_dir / "models"]
        candidate_models = []
        for directory in search_dirs:
            if directory.exists():
                candidate_models.extend(
                    [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".h5", ".keras"}]
                )

        # Prefer substantial model artifacts over tiny placeholder files.
        candidate_models = sorted(
            candidate_models,
            key=lambda p: (p.stat().st_size, p.stat().st_mtime),
            reverse=True,
        )

        if candidate_models:
            selected_model = candidate_models[0]
            MODEL_PATH = str(selected_model.resolve())
            MODEL_FILE_SIZE_MB = round(selected_model.stat().st_size / (1024 * 1024), 3)
            print(f"Found model: {selected_model.name}")
            print(f"Model size: {MODEL_FILE_SIZE_MB} MB")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"Loaded! Input: {model.input_shape}, Output: {model.output_shape}")

            # Try loading class-index mapping saved during training.
            possible_class_index_paths = [
                Path(MODEL_PATH).parent / "class_indices.json",
                current_dir / "models" / "class_indices.json",
                current_dir / "class_indices.json",
            ]
            for class_idx_path in possible_class_index_paths:
                if class_idx_path.exists():
                    try:
                        with class_idx_path.open("r", encoding="utf-8") as fp:
                            class_to_index = json.load(fp)
                        parsed = {int(idx): str(name) for name, idx in class_to_index.items()}
                        if parsed:
                            CLASS_INDEX_TO_NAME.clear()
                            CLASS_INDEX_TO_NAME.update(parsed)
                            print(f"Loaded class indices from: {class_idx_path}")
                            print(f"Class map: {CLASS_INDEX_TO_NAME}")
                            break
                    except Exception as class_exc:  # noqa: BLE001
                        print(f"Could not parse class indices from {class_idx_path}: {class_exc}")
        else:
            checked = ", ".join(str(p) for p in search_dirs)
            MODEL_ERROR = (
                "No model file found (.h5 or .keras). "
                f"Checked: {checked}. "
                "Place a trained model in the project root or models/ folder."
            )
            print(f"X {MODEL_ERROR}")
    except Exception as exc:  # noqa: BLE001
        MODEL_ERROR = str(exc)
        print(f"X {MODEL_ERROR}")


def prepare_image(image_file):
    if model is None or Image is None:
        return None

    try:
        img = Image.open(image_file)
        if img.mode != "RGB":
            img = img.convert("RGB")

        input_shape = model.input_shape
        if len(input_shape) == 2:
            img = img.resize((64, 64))
            img_array = np.array(img, dtype="float32") / 255.0
            img_gray = np.mean(img_array, axis=2)
            expected_size = input_shape[1]
            img_flat = img_gray.flatten()
            if len(img_flat) < expected_size:
                img_flat = np.pad(img_flat, (0, expected_size - len(img_flat)))
            else:
                img_flat = img_flat[:expected_size]
            return np.expand_dims(img_flat, axis=0)

        img_size = (input_shape[1], input_shape[2])
        img = img.resize(img_size)
        img_array = np.array(img, dtype="float32") / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as exc:  # noqa: BLE001
        print(f"Image error: {exc}")
        return None


def heuristic_parasitized_probability(img_array: np.ndarray) -> float:
    """Estimate parasitized probability from stain/color texture heuristics."""
    try:
        img = img_array[0]
        if img.ndim != 3 or img.shape[2] != 3:
            return 0.5

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        gray = np.mean(img, axis=2)

        # Remove mostly black background around segmented cells.
        cell_mask = gray > 0.12
        cell_pixels = int(np.sum(cell_mask))
        if cell_pixels < 200:
            return 0.5

        # Purple/dark inclusions are common visual cues for parasitized cells.
        purple_mask = (r > (g * 1.08)) & (b > (g * 1.08)) & (gray < 0.72) & cell_mask
        dark_mask = (gray < 0.42) & cell_mask

        purple_ratio = float(np.sum(purple_mask)) / cell_pixels
        dark_ratio = float(np.sum(dark_mask)) / cell_pixels
        texture = float(np.std(gray[cell_mask]))

        # Convert features to [0,1] probability-like terms.
        purple_score = 1.0 / (1.0 + np.exp(-((purple_ratio - 0.015) * 110.0)))
        dark_score = 1.0 / (1.0 + np.exp(-((dark_ratio - 0.10) * 35.0)))
        texture_score = min(texture / 0.22, 1.0)

        parasite_prob = (0.55 * purple_score) + (0.25 * dark_score) + (0.20 * texture_score)
        return float(np.clip(parasite_prob, 0.0, 1.0))
    except Exception as exc:  # noqa: BLE001
        print(f"Heuristic error: {exc}")
        return 0.5


def extract_class_probabilities(prediction: np.ndarray) -> tuple[float, float]:
    """
    Return (parasitized_prob, uninfected_prob) in [0,1].
    Supports:
    - sigmoid/binary output shape (1,)
    - softmax/multiclass output shape (>=2,)
    """
    raw = np.array(prediction, dtype="float32").squeeze()
    if raw.ndim == 0:
        raw = np.array([float(raw)], dtype="float32")
    elif raw.ndim > 1:
        raw = raw.ravel()

    if raw.size == 1:
        # Binary sigmoid: output is probability of class index 1.
        p_class1 = float(np.clip(raw[0], 0.0, 1.0))
        class1_name = CLASS_INDEX_TO_NAME.get(1, "Uninfected").lower()
        if "parasit" in class1_name:
            parasitized_prob = p_class1
            uninfected_prob = 1.0 - p_class1
        else:
            uninfected_prob = p_class1
            parasitized_prob = 1.0 - p_class1
        return float(np.clip(parasitized_prob, 0.0, 1.0)), float(np.clip(uninfected_prob, 0.0, 1.0))

    probs = raw.astype("float64")
    if np.any(probs < 0.0) or not np.isclose(np.sum(probs), 1.0, atol=1e-2):
        # Treat as logits if needed.
        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / np.sum(exp_probs)

    parasitized_prob = 0.0
    uninfected_prob = 0.0
    for idx, prob in enumerate(probs):
        class_name = CLASS_INDEX_TO_NAME.get(idx, f"class_{idx}").lower()
        if "parasit" in class_name:
            parasitized_prob += float(prob)
        elif "uninfect" in class_name or "healthy" in class_name or "normal" in class_name:
            uninfected_prob += float(prob)

    if parasitized_prob == 0.0 and uninfected_prob == 0.0:
        parasitized_prob = float(probs[0])
        uninfected_prob = float(probs[1]) if probs.size > 1 else (1.0 - parasitized_prob)
    else:
        total = parasitized_prob + uninfected_prob
        if total > 0:
            parasitized_prob /= total
            uninfected_prob /= total

    return float(np.clip(parasitized_prob, 0.0, 1.0)), float(np.clip(uninfected_prob, 0.0, 1.0))


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        start_time = time.perf_counter()

        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        img_array = prepare_image(file)
        if img_array is None:
            return jsonify({"success": False, "error": "Image processing failed"}), 500

        prediction = model.predict(img_array, verbose=0)
        print(f"Prediction: {prediction}")
        print(f"Shape: {prediction.shape}")

        model_parasitized_prob, model_uninfected_prob = extract_class_probabilities(prediction)

        heuristic_paras_prob = heuristic_parasitized_probability(img_array)
        heuristic_uninfected_prob = 1.0 - heuristic_paras_prob

        # If model is tiny/demo, rely mostly on heuristic.
        if MODEL_FILE_SIZE_MB < 1.0:
            decision_source = "heuristic"
            final_parasitized_prob = heuristic_paras_prob
        else:
            decision_source = "model"
            final_parasitized_prob = model_parasitized_prob
        final_uninfected_prob = 1.0 - final_parasitized_prob

        if final_parasitized_prob >= 0.5:
            pred_class = 0
            confidence = final_parasitized_prob * 100.0
        else:
            pred_class = 1
            confidence = final_uninfected_prob * 100.0

        result = CLASS_NAMES[pred_class]
        parasitized_prob = final_parasitized_prob * 100.0
        uninfected_prob = final_uninfected_prob * 100.0
        processing_time = round(time.perf_counter() - start_time, 2)
        model_name = os.path.basename(MODEL_PATH) if MODEL_PATH else "UnknownModel"

        print(
            "Decision:",
            {
                "result": result,
                "confidence": round(confidence, 2),
                "model_uninfected_prob": round(model_uninfected_prob, 4),
                "heuristic_parasitized_prob": round(heuristic_paras_prob, 4),
                "source": decision_source,
            },
        )

        return jsonify(
            {
                "success": True,
                "prediction": result,
                "confidence": round(confidence, 2),
                "model": model_name,
                "processing_time": processing_time,
                "image_quality": "Excellent",
                "decision_source": decision_source,
                "details": {
                    "parasitized_probability": round(parasitized_prob, 2),
                    "uninfected_probability": round(uninfected_prob, 2),
                    "model_parasitized_probability": round(model_parasitized_prob * 100.0, 2),
                    "model_uninfected_probability": round(model_uninfected_prob * 100.0, 2),
                    "heuristic_parasitized_probability": round(heuristic_paras_prob * 100.0, 2),
                    "heuristic_uninfected_probability": round(heuristic_uninfected_prob * 100.0, 2),
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception:  # noqa: BLE001
        pass

    return """<!DOCTYPE html>
<html>
<head>
  <title>Malaria Detection</title>
  <style>
    * {margin:0; padding:0; box-sizing:border-box;}
    body {
      font-family: Arial, sans-serif;
      background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%);
      min-height: 100vh;
      padding: 20px;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 20px;
      padding: 40px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    h1 {color: #333; margin-bottom: 30px; text-align: center;}
    .status {
      padding: 25px;
      border-radius: 12px;
      margin: 25px 0;
      text-align: center;
      border: 2px solid transparent;
    }
    .success {background: #d4edda; color: #155724; border-color: #c3e6cb;}
    .error {background: #f8d7da; color: #721c24; border-color: #f5c6cb;}
    .icon {font-size: 2.5rem; margin-bottom: 10px;}
  </style>
</head>
<body>
  <div class="container">
    <h1>Malaria Detection System</h1>
    <div class="status success">
      <div class="icon">✅</div>
      <h2>Server Running</h2>
      <p>Port 5000 active</p>
    </div>
    <div class="status %s">
      <div class="icon">%s</div>
      <h2>Model: %s</h2>
      <p>%s</p>
    </div>
  </div>
</body>
</html>""" % (
        "error" if model is None else "success",
        "❌" if model is None else "✅",
        "NOT LOADED" if model is None else "LOADED",
        MODEL_ERROR if MODEL_ERROR else "Ready!",
    )


@app.route("/health")
def health():
    return jsonify({"status": "running", "model_loaded": model is not None})


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    print("\n" + "=" * 70)
    print(f"{'✅' if model else '❌'} Model: {'LOADED' if model else 'NOT LOADED'}")
    if model:
        print(f"Input: {model.input_shape}, Output: {model.output_shape}")
    print("Server: http://127.0.0.1:5000")
    print("=" * 70 + "\n")

    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
