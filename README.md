# IPEO Hurricane Damage Assessment

A deep learning project for detecting hurricane damage in satellite imagery using PyTorch Lightning and ResNet models. This project uses GeoEye-1 satellite imagery from Hurricane Harvey to classify areas as **damaged** or **undamaged**.

---

## 📁 Project Structure

```
IPEO_hurricane_assesment/
├── inference.py                 # Training & inference script
├── calibration.ipynb            # Model calibration notebook
├── environment.yaml             # Conda environment file
│
├── checkpoints/                 # Trained model checkpoints (.ckpt)
├── ipeo_hurricane_for_students/
│   ├── train/                   # Training images (damage / no_damage)
│   ├── validation/              # Validation images
│   └── test/                    # Test images
├── logs/                        # CSV logs
│
└── src/
    ├── calibration.py           # Calibration utilities (Isotonic, ECE)
    ├── models/
    │   └── lightningmodel.py         # Pytorch lightning model specifications and constructors
    └── preprocessing/
        ├── data_loader.py       # Dataset class & DataLoaders
        ├── mean.pt              # Dataset normalization mean
        └── std.pt               # Dataset normalization std
```

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Model checkpoint downloadable here: https://filesender.switch.ch/filesender2/?s=download&token=a85d269d-99df-48a8-90f4-67102b64cbd9 (until 19/02/2026)
### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/IPEO_hurricane_assesment.git
cd IPEO_hurricane_assesment

# Create conda environment
conda env create -f environment.yaml
conda activate ipeo_hurricane

# Alternative: pip installation
# pip install -r requirements.txt
```


---

## 📊 Dataset

The project uses **GeoEye-1 satellite imagery** from Hurricane Harvey:

- **Classes**: 2 (damage, no_damage)
- **Format**: JPEG images

### Class Labels
| Label | Index |
|-------|-------|
| no_damage | 0 |
| damage | 1 |

---


## 📂 Key Files Description

### Source Code (`src/`)

| File | Description |
|------|-------------|
| `calibration.py` | Isotonic calibration, ECE computation, reliability diagrams |
| `models/lightningmodel.py` | PyTorch Lightning wrapper with training logic |
| `models/train.py` | Simplified training utilities |
| `preprocessing/data_loader.py` | `GeoEye1` dataset class, transforms, dataloaders |

### Notebooks

| Notebook | Description |
|----------|-------------|
| `inference.ipynb` | Interactive inference and visualization |
| `calibration.ipynb` | Model calibration analysis and reliability diagrams |

---


## 👤 Author

Cyrielle Manissadjan
Quentin Poindextre
Noé Staeheli






