import torch
from pathlib import Path
from typing import Union, Optional

from liniarRegressionModel import LinearRegressionModel


def save_model(model: torch.nn.Module, path: Union[str, Path] = Path("models/ml_model.pth")) -> Path:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to: {model_path}")
    torch.save(obj=model.state_dict(), f=model_path)
    print("Model was successfully saved")
    return model_path


def load_model(path: Union[str, Path], device: Optional[torch.device] = None) -> LinearRegressionModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else  "cpu"))

    state = torch.load(Path(path), map_location=device)
    model = LinearRegressionModel()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick demo: save a fresh model then load it back (CPU)
    demo_model = LinearRegressionModel()
    demo_path = save_model(demo_model, Path("models/ml_model.pth"))
    loaded = load_model(demo_path, device=torch.device("cpu"))
    print("Loaded model:", loaded)
