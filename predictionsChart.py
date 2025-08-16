import matplotlib.pyplot as plt
import numpy as np
import torch
from sympy.printing.pretty.pretty_symbology import line_width


def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
  # helper: accept torch tensors or numpy-like inputs
  def _to_numpy(x):
    if isinstance(x, torch.Tensor):
      return x.detach().cpu().ravel()
    return np.ravel(x)

  train_x = _to_numpy(train_data)
  train_y = _to_numpy(train_labels)
  test_x = _to_numpy(test_data)
  test_y = _to_numpy(test_labels)
  preds = _to_numpy(predictions) if predictions is not None else None

  fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

  # Minimalist scatter without explicit colors
  ax.scatter(train_x, train_y, s=18, label="Train", alpha=0.8)
  ax.scatter(test_x, test_y, s=28, marker="s", label="Test", alpha=0.9)

  # Optional preds as a smooth line for contrast
  if preds is not None:
    order = np.argsort(test_x)
    # show predictions as a line for clarity
    ax.plot(test_x[order], preds[order], label="Predictions", linewidth=2, color="g", alpha=0.9)

  # Typography & layout
  ax.set_title("Data & Predictions", pad=12)
  ax.set_xlabel("X")
  ax.set_ylabel("y")
  ax.legend(frameon=False, loc="best")
  ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

  # Cut visual clutter
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.tick_params(which="major", length=0)

  fig.tight_layout()
  plt.show()
  plt.close(fig)
