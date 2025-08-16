# ML PyTorch Linear Regression
A lightweight PyTorch project for learning and experimenting with linear regression on noisy, outlier‑rich synthetic data. Includes training, model persistence, and clean visualizations.

🎯 **Purpose**  
Teach and demo end‑to‑end regression on imperfect data: generate a dataset, fit a tiny model, save/load weights, and visualize predictions and loss.

🛠 **Tech Stack**  
- Language: Python 3.13  
- ML: PyTorch  
- Viz: Matplotlib, NumPy

🚀 **Quick Start**

Install dependencies
```bash
pip install torch matplotlib numpy
```

Run locally
```bash
python main.py
```
- If a saved model exists at `models/ml_model.pth`, it’s loaded on CPU and used for inference; otherwise a new model is trained and plotted.

📁 **Core Features**
- **Synthetic dataset** with heteroscedastic noise, mild nonlinearity (X² term), and configurable outliers.
- **Tiny model**: single `nn.Linear(1, 1)` for transparent math and fast iteration.
- **Training loop** using SGD + L1/MAE, with a loss curve over epochs. 
- **Model persistence**: utilities to save and load weights; auto‑selects CUDA/MPS/CPU. 
- **Visualization**: scatter train/test and a smooth predictions line for clarity.

🧭 **Project Structure**
```
.
├─ main.py
├─ liniarRegressionModel.py
├─ trainingLoop.py
├─ workingData.py
├─ predictionsChart.py
├─ modelSave.py
└─ models/
    └─ ml_model.pth
```

🔎 **File References**
- `main.py` — Entry point: load existing model or train, then plot.【14†source】  
- `liniarRegressionModel.py` — Simple `nn.Linear` model (1→1).【13†source】  
- `trainingLoop.py` — Training routine + loss plot.【17†source】  
- `workingData.py` — Data generator with noise, nonlinearity, outliers.【18†source】  
- `predictionsChart.py` — Plot train/test and predictions overlay.【16†source】  
- `modelSave.py` — Save/load utilities with device handling.【15†source】  

📝 **Notes**
- Default save path: `models/ml_model.pth`.【15†source】  
- On run, the script prefers an existing checkpoint (CPU inference) or trains a fresh model if none is found.【14†source】  

Built with 🔥 using PyTorch.
