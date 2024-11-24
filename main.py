import time
import matplotlib.pyplot as plt
from FDTD_Run import FDTD_Simulator
from struct_gen import FDTD_grid
from struct_gen import structure2D
import numpy as np
import torch
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cuda"
#device=torch.device("cpu")
layer1=structure2D(50,np.array([1]),np.array([1,1]))
layer2=structure2D(20,np.array([20,40,60,100]),np.array([1,2,3,4,4]))
layer3=structure2D(20,np.array([20,40,60,80]),np.array([5,4,3,2,1]))
layers=[layer1]
fdtd_grid = FDTD_grid(1, layers, 1, 1, device=device)
FDTD=FDTD_Simulator(fdtd_grid,1e-9,1e-7)
FDTD.FDTD_Sim()
end_time = time.time()
FDTD.animate_pytorch_matrices()
print(end_time-start_time)
plt.imshow(FDTD.Ey_frame[1].cpu())
plt.colorbar()
plt.figure(1)
plt.show()