import torch
import numpy as np
#Tensor Initialization

#Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data


#With random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#Tensor Attributes
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)