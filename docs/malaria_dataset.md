# Malaria Cell Images Dataset

## Source
- Kaggle: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

## Overview
The Malaria Cell Images Dataset is a public collection of microscopic thin blood smear images used to train deep learning models for malaria detection. The classification objective is binary:
- `Parasitized` (infected)
- `Uninfected` (healthy)

This dataset supports automated diagnostic systems that classify blood-cell images into infected vs non-infected categories.

## Dataset Structure
### Full NIH Dataset
- Total images: `27,558`
- Download size: `337.08 MB`
- Uncompressed size: `317.62 MB`
- Image format: `PNG` (RGB)
- Class balance: approximately `50/50`

### Project Subset Used
- Total images: `998`
- Parasitized: `499` (`50.00%`)
- Uninfected: `499` (`50.00%`)
- Provided as two ZIP files:
  - `parasitized.zip`
  - `uninfected.zip`

## Dataset Characteristics
### Image Properties
- Image type: segmented individual blood-cell images from thin blood smears
- Color space: RGB (3 channels)
- Variable dimensions:
  - Average: `141 x 143`
  - Minimum: `79 x 82`
  - Maximum: `247 x 226`
- Staining method: Giemsa staining

### Class Definitions
1. `Parasitized` (Infected)
- Contains Plasmodium-infected red blood cells
- Parasites are visible inside cells
- Multiple parasite stages may appear (rings, trophozoites, schizonts)
- Class label: `0`

2. `Uninfected` (Healthy)
- Normal red blood cells with no visible parasites
- Clear cell morphology without malaria infection indicators
- Class label: `1`

## Notes For Modeling
- Binary classification problem with balanced classes.
- Variable image sizes should be standardized (for example, resized to `224 x 224`) before training.
- Typical preprocessing includes pixel normalization (`rescale=1/255`).
