import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a tensor

# 1) Scalar
scalar = torch.tensor(7)
print(f"Scalar: {scalar}") # tensor(7)
print(f"Dimensions: {scalar.ndim}") # prints scalar dimensions
print(f"Item: {scalar.item()}") # prints item 
print (f"Shape: {scalar.shape}") # prints shape

# 2) Vector
vector = torch.tensor([7, 7])
print (f"\nVector: {vector}") # tensor([7, 7])
shape = vector.shape
print(f"Shape: {shape}") 

# 3) Matrix
MATRIX = torch.tensor([[1, 2, 6, 8],
                      [3, 4, 5, 7]])

print(f"\nMatrix: {MATRIX}")
print (f"Shape: {MATRIX.shape}") # prints shape

# 4) Tensor
TENSOR = torch.tensor([[[1, 2, 3], 
                       [4, 5, 6],
                       [7, 8, 9]]])
print (f"\nTensor: {TENSOR}")
print(f"Shape: {TENSOR.shape}") 
print (f"Dimensions: {TENSOR.ndim}") # prints dimensions
print(TENSOR[0])

random_tensor = torch.rand(3, 4)
print(f"\nRandom Tensor: {random_tensor}")

random_tensor_image = torch.rand(224, 224, 3) # height, width, color channels (Red, Green, Blue)

# create a tensor with zeros
tensor_zeros = torch.zeros(3, 4)
print(f"\nTensor with zeros: {tensor_zeros}")

# create a tensor with ones
tensor_ones = torch.ones(3, 4)
print(f"\nTensor with zeros: {tensor_ones}")

print(tensor_ones.dtype) # prints data type of tensor