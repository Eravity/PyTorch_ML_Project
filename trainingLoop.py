import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from liniarRegressionModel import LinearRegressionModel
from workingData import working_data
from modelSave import save_model
# Train the model
def train_and_predict(epochs: int = 2000, lr: float = 0.01):
  X_train, y_train, X_test, y_test = working_data()

  loops = []
  loss_values = []

  torch.manual_seed(1283737888999)
  model = LinearRegressionModel()
  loss_fn = nn.L1Loss()
  optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

  # Diagnostics: print initial params and initial predictions on test set
  model.eval()

  model.train()
  for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    if epoch % 10 == 0:
      loops.append(epoch)
      loss_values.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # get fresh predictions after training
  model.eval()
  with torch.inference_mode():
    y_preds = model(X_test)

  save_model(model)

  plt.plot(loops, np.array(torch.tensor(loss_values).numpy()), label="Train Loss")
  plt.title("Training and loss dependency")
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.legend()
  plt.show()

  return X_train, y_train, X_test, y_test, y_preds

if __name__ == "__main__":
  # optional: quick smoke test
  _ = train_and_predict()

