# Malaria Cell Detection (SmartBridge Project)

End-to-end deep learning project for malaria blood cell classification using microscopy images.

The system classifies cell images into:
- `Parasitized` (infected)
- `Uninfected` (healthy)

## Project Summary
- Dataset: malaria cell images from Kaggle (NIH source)
- Pipeline: data preparation, training, evaluation, and Flask deployment
- Best model result (reported): Custom CNN with ~98.99% validation accuracy and ~0.9995 ROC-AUC
- Web app: upload image, run prediction, and view confidence + diagnostic details

## Dataset
- Kaggle: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- Notes: see `docs/malaria_dataset.md`

## Tech Stack
- Python 3.11
- TensorFlow / Keras
- Flask
- NumPy, Pandas, Matplotlib, Seaborn, SciPy, scikit-learn, Pillow

## Repository Structure
- `app.py`: Flask backend for inference
- `templates/index.html`: frontend page
- `static/style.css`: frontend styles
- `train.py`: single-model training pipeline
- `train_all_models.py`: multi-model comparison (CustomCNN, MobileNetV2, EfficientNetB0)
- `scripts/prepare_dataset.py`: split and organize dataset
- `scripts/explore_dataset.py`: dataset analysis and visualization
- `requirements.txt`: pip dependencies
- `environment.yml`: conda environment

## Installation
### Option A: pip
```bash
pip install -r requirements.txt
```

### Option B: conda
```bash
conda env create -f environment.yml
conda activate smartbridge
```

## Data Preparation
Place extracted images under `data/raw`, then run:

```bash
python scripts/prepare_dataset.py --source data/raw --output data/splits --val-size 0.2
```

Optional dataset exploration:
```bash
python scripts/explore_dataset.py --train-dir data/splits/train --report-dir reports
```

## Training
Train one model:
```bash
python train.py --train-dir data/splits/train --valid-dir data/splits/valid --epochs 30 --batch-size 32
```

Train and compare multiple models:
```bash
python train_all_models.py --data-dir data/splits/train --img-size 128 --batch-size 32 --epochs 40 --best-model-name best_malaria_model.keras
```

## How to Run (Local)
`app.py` loads the newest model file from:
- project root, or
- `models/` folder

Supported model formats:
- `.h5`
- `.keras`

Recommended quick start:
```bash
cd "/Users/aanyagarg/Documents/SMARTBRIDGE PROJECT"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open in browser:
- http://127.0.0.1:5000

If you see `Model: NOT LOADED`, add a trained model file (example: `best_malaria_model_fixed.h5`) in the root or `models/`, then restart.

Health endpoint:
- `GET /health`

Prediction endpoint:
- `POST /predict` with form-data key: `file`

Example:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@/absolute/path/to/cell_image.png"
```

## Important Notes
- This project is for educational/research screening support.
- Not a substitute for professional medical diagnosis.
- Model files and large datasets are excluded from Git by default via `.gitignore`.

## Update GitHub Repository
If git is missing on macOS:
```bash
xcode-select --install
```

Set remote (one-time):
```bash
cd "/Users/aanyagarg/Documents/SMARTBRIDGE PROJECT"
git remote set-url origin "https://github.com/yugalgarg157-glitch/Smartbrdge-project.git"
```

Commit and push updates:
```bash
git add .
git commit -m "Update malaria app and run instructions"
git push -u origin main
```

Authentication prompt:
- Username: `yugalgarg157-glitch`
- Password: use GitHub Personal Access Token (PAT), not your normal GitHub password.
