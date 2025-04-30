# Hieroglyphs Handwriting Letter Recognition

Lightweight CNN for Egyptian hieroglyph handwriting classification with end-to-end data preparation, training, evaluation, and model explainability (SHAP + LIME). This project is centered around a single notebook, `code.ipynb`, and targets the Kaggle dataset "Hieroglyphs Handwriting Letter Recognition".

## Project Structure

```
heiroglyphics/
├─ code.ipynb        # Main notebook: setup → train → evaluate → explain → export
└─ README.md         # This guide
```

## Dataset

- Source: Kaggle dataset `mdismielhossenabir/hieroglyphs-handwriting-letter-recognition`.
- Each class folder contains images for a specific hieroglyph letter (e.g., `f`, `h`, `kh`).
- The notebook downloads the dataset via Kaggle API and splits it into `train/val/test` with an 80/10/10 ratio.

## Environment Options

You can run the notebook in either Google Colab (recommended for simplicity) or locally on Windows with VS Code.

### Option A: Google Colab (Recommended)

1. Open `code.ipynb` in Colab.
2. Upload your `kaggle.json` when prompted (Cell 2).
3. Run all cells in order (1 → 13). See the Notebook Walkthrough below.

### Option B: Local on Windows (VS Code)

1. Install Python 3.10+ and VS Code. Create and activate a virtual environment:

	 ```powershell
	 python -m venv .venv
	 .venv\Scripts\activate
	 ```

2. Install required packages (TensorFlow 2.14 is expected):

	 ```powershell
	 pip install kaggle tensorflow==2.14.0 shap==0.42.1 lime scikit-image matplotlib seaborn scikit-learn
	 # If SHAP errors occur, pin NumPy:
	 pip install numpy==1.24.4
	 ```

3. Configure Kaggle CLI (place your API token):

	 ```powershell
	 mkdir $env:USERPROFILE\.kaggle
	 copy C:\path\to\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
	 ```

4. Download the dataset locally and unzip:

	 ```powershell
	 kaggle datasets download -d mdismielhossenabir/hieroglyphs-handwriting-letter-recognition -p . --unzip
	 ```

5. Open `code.ipynb` in VS Code and edit paths:
	 - Replace Colab-specific paths like `/content` with a local folder (e.g., `C:/Users/<you>/hieroglyphs_data`).
	 - Update the variables in Cells 3–13:
		 - `base`, `DATA_ROOT`, `OUT_DIR`, `checkpoint_path`, model save path.
	 - Remove or disable Colab-only imports (`from google.colab import files`) and upload flow.

## Notebook Walkthrough (Cell-by-Cell)

This is a brief tour of the notebook cells to help you run them confidently.

1. Markdown – Title banner.
2. Install & Kaggle auth – Installs packages and handles `kaggle.json` (Colab upload). Locally, install via pip and place `kaggle.json` in `%USERPROFILE%\.kaggle`.
3. Dataset download listing – Verifies dataset structure under `/content` (adjust `base` locally).
4. Split to train/val/test – Creates `hieroglyphs_split/train|val|test` with deterministic shuffling.
5. Data generators – `ImageDataGenerator` with augmentation, rescale to 100×100 as per paper.
6. Model – Four convolution blocks, BatchNorm, L2 regularization, Dropout, dense head.
7. Compile & callbacks – `Adam(1e-4)`, checkpoint to `best_model.h5`, LR reduction, early stopping.
8. Train – Fit for `EPOCHS = 8` (tuneable), validates on `val` split.
9. Metrics plots & reload best – Plots loss/accuracy; reloads the best checkpoint.
10. Test evaluation – Classification report, confusion matrix, ROC/AUC (one-vs-rest).
11. SHAP – `DeepExplainer` explanations over selected test images (uses a small background set).
12. LIME – Local surrogate explanations with superpixel boundaries for predicted class.
13. Save model – Exports `hieroglyph_light_cnn.h5`.
14. Compatibility note – Pins `numpy==1.24.4` if SHAP compatibility issues arise.

## Model Details

- Input: RGB 100×100.
- Architecture: 4 × Conv2D blocks with `BatchNormalization` and `MaxPooling`.
- Regularization: L2 on conv and dense layers; Dropout 0.5 and 0.4.
- Optimizer: Adam `1e-4`.
- Loss: Categorical cross-entropy.
- Metrics: Accuracy.

## Results & Explainability

- Evaluation: Per-class metrics, confusion matrix heatmap, and macro AUC (if sufficient samples).
- SHAP: Global/feature contribution maps via `DeepExplainer`; average absolute SHAP shown per predicted class.
- LIME: Top positive regions highlighting local evidence for predictions.

## Troubleshooting

- SHAP/NumPy version conflicts:
	- If `DeepExplainer` raises errors, downgrade NumPy:
		```powershell
		pip install numpy==1.24.4
		```
	- Rerun the SHAP cell afterward.

- Paths on Windows:
	- Change `/content/...` to a local absolute path.
	- Ensure `checkpoint_path` and model save locations are writable.

- GPU/CPU:
	- TensorFlow will default to CPU if no GPU is available; training time will increase.

## Quick Commands (Windows)

```powershell
# Create venv and install deps
python -m venv .venv
.venv\Scripts\activate
pip install kaggle tensorflow==2.14.0 shap==0.42.1 lime scikit-image matplotlib seaborn scikit-learn

# Kaggle auth
mkdir $env:USERPROFILE\.kaggle
copy C:\path\to\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json

# Download dataset
kaggle datasets download -d mdismielhossenabir/hieroglyphs-handwriting-letter-recognition -p . --unzip
```

## Acknowledgements

- Dataset: `mdismielhossenabir/hieroglyphs-handwriting-letter-recognition` on Kaggle.
- SHAP and LIME libraries for model explainability.

## Credits
Hrishikesh Dalal
