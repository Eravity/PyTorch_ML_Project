import torch

def get_device():
  return torch.device(
      "cuda" if torch.cuda.is_available() else 
      "mps" if torch.backends.mps.is_available() else
      "cpu"
  )

tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"Current device: {tensor.device}")

device = get_device()
tensor = tensor.to(device)

print(f"Tensor moved to device: {tensor.device}")