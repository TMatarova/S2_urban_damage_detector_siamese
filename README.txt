# Siamese CNN for Building Damage Detection

A Siamese Convolutional Neural Network for binary building damage detection
using pre- and post-conflict Sentinel-2 satellite imagery, trained on
22 conflict-affected Ukrainian regions.

**Dataset and pretrained weights:** https://zenodo.org/records/19441105

---

## Repository Contents

| File / Folder | Description |
|---|---|
| `Model_3 (final model).py` | Final model — ResNet-101 backbone (recommended) |
| `Model_0.py` – `Model_7.py` | All 8 model variants |
| `Prepocessing_patches.py` | Patch extraction from Sentinel-2 GeoTIFFs |
| `EDA.ipynb` | Exploratory data analysis notebook |
| `Results_Comparison.py` | Cross-model evaluation and ranking |
| `FinalBalanced_splits.npz` | Final train/val/test dataset — download from Zenodo |
| `Model_3_weights.pt` | Pretrained weights — download from Zenodo |
| `outputs/` | Metrics, confusion matrices, and predictions for all models |

---

## Requirements

- Python 3.10+
- Conda environment recommended

### Python Libraries

```
pip install torch torchvision numpy scikit-learn matplotlib rasterio pandas gdal
```

| Library | Purpose |
|---|---|
| PyTorch + torchvision | Model training and ResNet backbone |
| NumPy | Array operations |
| scikit-learn | Metrics and evaluation |
| Matplotlib | Plots and confusion matrices |
| Rasterio | Reading/writing GeoTIFF satellite images |
| GDAL | Geospatial raster processing |
| pandas | Results comparison |

---

## Hardware

### Training

| Spec | Detail |
|---|---|
| **Trained on** | NVIDIA GeForce RTX 4070 (12 GB VRAM) |
| **Minimum GPU** | CUDA-enabled GPU with 6 GB+ VRAM |
| **RAM** | 16 GB+ recommended |
| **Storage** | ~2 GB for dataset and model weights |
| **OS** | Windows 10/11 or Linux |

> Training on CPU is possible but slow (~2–4 hours). GPU training completes
> in under 10 minutes on an RTX 4070.

### Inference Only (running pretrained model)

| Spec | Detail |
|---|---|
| **GPU** | Not required — CPU is sufficient |
| **RAM** | 8 GB+ |
| **Storage** | ~500 MB for model weights and patches |
| **OS** | Windows 10/11 or Linux |

---

## Training Options

### Option 1 — Use the preprocessed dataset (recommended)

The `FinalBalanced_splits.npz` file contains the ready-to-use train/val/test
splits. Simply run:

```
python "Model_3 (final model).py"
```

### Option 2 — Train on your own dataset

1. Download pre- and post-event **Sentinel-2 Level-2A** imagery (bands B02,
   B03, B04, B08) from the Copernicus Browser:
   https://browser.dataspace.copernicus.eu/

2. Obtain a building damage assessment shapefile for your region (e.g. from
   UNOSAT: https://www.unitar.org/maps/unosat-automated-map-production)

3. In **QGIS**:
   - Clip the before and after Sentinel-2 images to your region of interest
   - Rasterise the UNOSAT damage polygons to produce a binary mask GeoTIFF

4. Update the city paths in `Prepocessing_patches.py` and run it to extract
   64×64 pixel patch pairs saved as `.npz` files

5. Combine and split your `.npz` files to match the `FinalBalanced_splits.npz`
   format (keys: `Xb_train`, `Xa_train`, `y_train`, `Xb_val`, `Xa_val`,
   `y_val`, `Xb_test`, `Xa_test`, `y_test`)

6. Update `NPZ_PATH` in the model script and run

---

## Running Options

### Option 1 — Run inference on the provided test set

Load the pretrained weights and evaluate on the existing test split:

```
python "Model_3 (final model).py"
```

The script will automatically skip training if weights are already saved and
run evaluation on the test set, saving metrics to `outputs/`.

### Option 2 — Run the pretrained model on a new region

Use `Model_3_weights.pt` to run inference on new Sentinel-2 imagery without
retraining.

1. Download pre- and post-event **Sentinel-2 Level-2A** imagery (bands B02,
   B03, B04, B08) for your region from the Copernicus Browser:
   https://browser.dataspace.copernicus.eu/

2. In **QGIS**, clip and align the before and after images to the same extent

3. Run `Prepocessing_patches.py` on your new imagery to extract 64×64 pixel
   patch pairs (no mask required — use a blank raster for the mask)

4. Run inference using the pretrained weights:

```python
import torch
import numpy as np
from torch.utils.data import DataLoader

# Load your new patches
data = np.load("your_new_patches.npz")
# Use normalisation stats from the original training data
norm = np.load("normalization_stats.npz")

ds = PairsDataset(data["before"], data["after"],
                  np.zeros(len(data["before"])),  # dummy labels
                  norm["mean"], norm["std"])
loader = DataLoader(ds, batch_size=64, shuffle=False)

# Load pretrained model
model = SiameseCNN(in_channels=4)
model.load_state_dict(torch.load("Model_3_weights.pt", map_location="cpu"))
model.eval()

# Run inference
probs, _ = _probs_and_labels(model, loader, device="cpu")
predictions = (probs > 0.5).astype(int)  # 0 = No-Damage, 1 = Damage
np.save("predictions.npy", predictions)
```

5. Predictions are saved as a NumPy array (0 = No-Damage, 1 = Damage),
   one value per patch pair

> **Note:** The pretrained model was trained on Ukrainian urban regions using
> Sentinel-2 imagery. Performance may vary on regions with different land
> cover or damage patterns. Retraining on local data is recommended for
> best results.

---

## Dataset

- **22 conflict-affected Ukrainian regions** (Antonivka, Avdiivka, Azovstal,
  Bucha, Chernihiv, Hostomel, Irpin, Kharkiv, Kherson, Kramatorsk,
  Kremenchuk, Lysychansk, Makariv, Melitopol, Mykolaiv, Okhtyrka, Rubizhne,
  Shchastia, Sumy, Trostianets, Volnovakha, Vorzel)
- **7,968 patch pairs** — 3,984 damaged, 3,984 non-damaged (balanced)
- **Patch size**: 64×64 pixels (640×640 m at 10 m resolution)
- **Bands**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
- **Labels**: UNOSAT Copernicus Damage Assessment shapefiles
- **Split**: 60% train / 20% val / 20% test

---

## Results (Final Model — ResNet-101)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No-Damage | 0.895 | 0.794 | 0.842 |
| Damage | 0.815 | 0.907 | 0.859 |
| **Overall Accuracy** | | | **0.851** |


---

## Data Sources

- Sentinel-2 imagery: European Space Agency / Copernicus Programme
- Damage labels: UNOSAT / United Nations Satellite Centre (UNITAR)
