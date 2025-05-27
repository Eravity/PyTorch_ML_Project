import torch

tensor = torch.tensor([[[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]]])

print(f"Dimensions: {tensor.ndim}")
print(f"Shape: {tensor.shape}")

# Size: [1, 3, 3] because we have 1 matrix with 3 rows and 3 columns
torch.set_printoptions(precision=4, sci_mode=False)

random_tensor = torch.rand(5, 6, 6)
print(f"\n{random_tensor}")