import torch
from pathlib import Path

from predictionsChart import plot_predictions
from trainingLoop import train_and_predict
from workingData import working_data
from modelSave import load_model

MODEL_FILE = Path("models/ml_model.pth")


def main():
  # If a saved model exists, load it (on CPU) and run inference.
  if MODEL_FILE.exists():
    print(f"Found saved model at {MODEL_FILE}, loading (CPU)...")
    model = load_model(MODEL_FILE, device=torch.device("cpu"))
    X_train, y_train, X_test, y_test = working_data()
    model.eval()
    with torch.inference_mode():
      y_preds = model(X_test)
  else:
    print("No saved model found — training a new model (this may take a while)...")
    X_train, y_train, X_test, y_test, y_preds = train_and_predict()

  plot_predictions(X_train,
                   y_train,
                   X_test,
                   y_test,
                   predictions=y_preds)


if __name__ == "__main__":
  main()
