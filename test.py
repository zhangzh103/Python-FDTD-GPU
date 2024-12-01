import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Import the updated classes
from structure2D_Ver2 import structure2D, FDTD_grid
from FDTD_Run_Ver2 import FDTD_Simulator2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")

# Plotting a sine wave (unrelated to FDTD simulation)
x_torch = torch.linspace(0, 2 * np.pi, 1000)
y_torch = torch.sin(x_torch)
x_np = x_torch.cpu().numpy()
y_np = y_torch.cpu().numpy()
plt.figure(1)
plt.plot(x_np, y_np)
plt.title('Sine Function (from PyTorch Tensor)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

# Concatenation examples
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
cat_dim0 = torch.cat((a, b), dim=0)
print("Concatenate along dim=0 (row-wise):\n", cat_dim0)
cat_dim1 = torch.cat((a, b), dim=1)
print("Concatenate along dim=1 (column-wise):\n", cat_dim1)

# FDTD Simulation Setup
start_time = time.time()

# Define layers with swapped x and y
layer1 = structure2D(np.array([0.2,0.4]),0.4, np.array([1,2,3]))
layer2 = structure2D(np.array([]),1, np.array([1]))

# Create FDTD grids
#grider_ur = FDTD_grid(0.6, [layer1], 0.2, 0.1)
grider_er = FDTD_grid(1, [layer2], 0.2, 0.2)
layer1 = structure2D(np.array([]),0.4, np.array([1]))
layer2 = structure2D(np.array([]),1, np.array([1]))
grider_ur = FDTD_grid(1, [layer2], 0.2, 0.2)

end_time = time.time()

# Initialize FDTD Simulator (assuming it's properly defined)
fdtd = FDTD_Simulator2D(grider_er, grider_ur, [0, 0], [1, 1])

print("Simulation setup time:", end_time - start_time)

# Visualize the dielectric grid
plt.figure(2)
plt.imshow(grider_ur.dielectric_grid)
plt.colorbar(label='Dielectric Constant')
plt.title('Dielectric Grid')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()

