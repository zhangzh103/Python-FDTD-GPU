#Is there a way we can turn the whole time slide into one matrix and calculate this one matrix together
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
class structure2D:
    # Define structures in nm
    x_val = 0  # thickness
    y_val = np.array([0])  # width
    dielectric_val = np.array([0])

    def __init__(self, x, y, dielectric_cons):
        if not isinstance(x, (int, float)) or not isinstance(y, np.ndarray) or not isinstance(dielectric_cons, np.ndarray):
            raise ValueError("Invalid input types.")

        if len(y) + 1 != len(dielectric_cons):
            raise ValueError("The length of 'y' and 'dielectric_cons' must be the same.")

        self.x_val = x
        self.y_val = np.append(self.y_val, y)  # Append values to y_val
        self.dielectric_val = dielectric_cons


class FDTD_grid:
    # All the variables inside here need to be tensors
    dielectric_grid = torch.tensor([[]])  # dielectric constant for each pixel
    dx_grid = torch.tensor([[]])  # the time value for each pixel
    dy_grid = torch.tensor([[]])  # the time value for each pixel

    def __init__(self, peroid, layers, dx, dy, device='cpu'):  # layers is composed of structure2D list
        x_max = 0
        y_max = 0
        # Moving all operations to the device (GPU or CPU)
        self.dielectric_grid = self.dielectric_grid.to(device)
        self.dx_grid = self.dx_grid.to(device)
        self.dy_grid = self.dy_grid.to(device)

        for i in layers:
            layer_x_grid = torch.tensor([[]], device=device)
            layer_y_grid = torch.tensor([[]], device=device)
            layer_dile_grid = torch.tensor([[]], device=device)

            dile_grids = []  # Collecting grids for efficient concatenation
            x_ranges = []
            y_ranges = []

            # Pre-compute all the y ranges for the current layer
            for j in range(len(i.y_val) - 1):
                x = torch.arange(0, i.x_val , dx, device=device)  # Ensure the upper bound is included
                y = torch.arange(i.y_val[j], i.y_val[j+1], dy, device=device)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                grid_x = grid_x.clone().add_(x_max)  # Clone and then modify
                die_grid = grid_x * 0 + i.dielectric_val[j]

                dile_grids.append(die_grid)
                x_ranges.append(grid_x)
                y_ranges.append(grid_y)

            # Process the last segment of the current layer
            if not peroid-i.y_val[len(i.y_val)-1]==0:
                x = torch.arange(0, i.x_val , dx, device=device)
                y = torch.arange(i.y_val[len(i.y_val)-1], peroid, dy, device=device)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                grid_x = grid_x.clone().add_(x_max)  # Clone before modifying in-place
                x_max = torch.max(grid_x)
                y_max = torch.max(grid_y)
                die_grid = grid_x * 0 + i.dielectric_val[len(i.y_val)-1]

                dile_grids.append(die_grid)
                x_ranges.append(grid_x)
                y_ranges.append(grid_y)

            # Efficient concatenation of the grids
            layer_dile_grid = torch.cat(dile_grids, dim=1)
            layer_x_grid = torch.cat(x_ranges, dim=1)
            layer_y_grid = torch.cat(y_ranges, dim=1)

            # Concatenate into main grid
            if self.dielectric_grid.shape[0] < 2:
                self.dielectric_grid = layer_dile_grid
                #self.dx_grid = layer_x_grid
                #self.dy_grid = layer_y_grid
            else:
                self.dielectric_grid = torch.cat((self.dielectric_grid, layer_dile_grid), dim=0)
                #self.dx_grid = torch.cat((self.dx_grid, layer_x_grid), dim=0)
                #self.dy_grid = torch.cat((self.dy_grid, layer_y_grid), dim=0)
            self.dx_grid=dx+0*self.dielectric_grid
            self.dy_grid=dy+0*self.dielectric_grid

    def second_order_nonlinear_update(self):
        # Placeholder for update function
        pass

# # Example usage
# start_time = time.time()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #device=torch.device("cpu")
# layer1=structure2D(50,np.array([1]),np.array([1,1]))
# layer2=structure2D(20,np.array([20,40,60,100]),np.array([1,2,3,4,4]))
# layer3=structure2D(20,np.array([20,40,60,80]),np.array([5,4,3,2,1]))
# layers=[layer1]
# fdtd_grid = FDTD_grid(1, layers, 1, 1, device=device)
# end_time = time.time()
# print(end_time - start_time)
# print(fdtd_grid.dielectric_grid)
# plt.imshow(fdtd_grid.dielectric_grid.cpu().numpy())
# plt.show()
