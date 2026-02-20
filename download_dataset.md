# Download Malaria Cell Images Dataset

## Kaggle Source
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

## Option 1: Kaggle API (recommended)
1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```
2. Place your API key at `~/.kaggle/kaggle.json`.
3. Download and extract:
   ```bash
   kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
   unzip cell-images-for-detecting-malaria.zip -d data/raw
   ```

## Option 2: Manual download
1. Open the Kaggle dataset URL in your browser.
2. Download ZIP.
3. Extract into `data/raw`.

## Expected Folder Layout
`data/raw` should contain class folders like:
- `Parasitized`
- `Uninfected`

Then run:
```bash
python scripts/prepare_dataset.py --source data/raw --output data/splits --val-size 0.2
python scripts/explore_dataset.py --train-dir data/splits/train --report-dir reports
```
