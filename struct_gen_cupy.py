import cupy as cp
import matplotlib.pyplot as plt
import time

class structure2D:
    # Define structures in nm
    x_val = 0  # thickness
    y_val = cp.array([0])  # width
    dielectric_val = cp.array([0])

    def __init__(self, x, y, dielectric_cons):
        if not isinstance(x, (int, float)) or not isinstance(y, cp.ndarray) or not isinstance(dielectric_cons, cp.ndarray):
            raise ValueError("Invalid input types.")

        if len(y) + 1 != len(dielectric_cons):
            raise ValueError("The length of 'y' and 'dielectric_cons' must be the same.")

        self.x_val = x
        self.y_val = cp.append(self.y_val, y)  # Append values to y_val
        self.dielectric_val = dielectric_cons


class FDTD_grid:
    # All the variables inside here need to be cupy arrays
    dielectric_grid = cp.array([[]])  # dielectric constant for each pixel
    dx_grid = cp.array([[]])  # the time value for each pixel
    dy_grid = cp.array([[]])  # the time value for each pixel

    def __init__(self, period, layers, dx, dy):  # layers is composed of structure2D list
        x_max = 0
        y_max = 0
        
        for i in layers:
            layer_x_grid = cp.array([[]])
            layer_y_grid = cp.array([[]])
            layer_dile_grid = cp.array([[]])

            dile_grids = []  # Collecting grids for efficient concatenation
            x_ranges = []
            y_ranges = []

            # Pre-compute all the y ranges for the current layer
            for j in range(len(i.y_val) - 1):
                x = cp.arange(0, i.x_val + dx, dx)  # Ensure the upper bound is included
                y = cp.arange(i.y_val[j], i.y_val[j+1], dy)
                grid_x, grid_y = cp.meshgrid(x, y, indexing='ij')
                grid_x = grid_x + x_max  # Modify directly in CuPy
                die_grid = cp.zeros_like(grid_x) + i.dielectric_val[j]

                dile_grids.append(die_grid)
                x_ranges.append(grid_x)
                y_ranges.append(grid_y)

            # Process the last segment of the current layer
            x = cp.arange(0, i.x_val + dx, dx)
            y = cp.arange(i.y_val[len(i.y_val)-1], period + dy, dy)
            grid_x, grid_y = cp.meshgrid(x, y, indexing='ij')
            grid_x = grid_x + x_max  # Modify in place
            x_max = cp.max(grid_x)
            y_max = cp.max(grid_y)
            die_grid = cp.zeros_like(grid_x) + i.dielectric_val[len(i.y_val)-1]

            dile_grids.append(die_grid)
            x_ranges.append(grid_x)
            y_ranges.append(grid_y)

            # Efficient concatenation of the grids
            layer_dile_grid = cp.concatenate(dile_grids, axis=1)
            layer_x_grid = cp.concatenate(x_ranges, axis=1)
            layer_y_grid = cp.concatenate(y_ranges, axis=1)

            # Concatenate into main grid
            if self.dielectric_grid.shape[0] < 2:
                self.dielectric_grid = layer_dile_grid
                self.dx_grid = layer_x_grid
                self.dy_grid = layer_y_grid
            else:
                self.dielectric_grid = cp.concatenate((self.dielectric_grid, layer_dile_grid), axis=0)
                self.dx_grid = cp.concatenate((self.dx_grid, layer_x_grid), axis=0)
                self.dy_grid = cp.concatenate((self.dy_grid, layer_y_grid), axis=0)

    def second_order_nonlinear_update(self):
        # Placeholder for update function
        pass

# Example usage
start_time = time.time()
layer1 = structure2D(50, cp.array([50]), cp.array([1, 1]))
layer2 = structure2D(20, cp.array([20, 40, 60, 100]), cp.array([1, 2, 3, 4, 4]))
layer3 = structure2D(20, cp.array([20, 40, 60, 80]), cp.array([5, 4, 3, 2, 1]))
layers = [layer1, layer2, layer3, layer3, layer3, layer3, layer3, layer3, layer3, layer3, layer3]
fdtd_grid = FDTD_grid(100, layers, 1, 1)
end_time = time.time()
print(end_time - start_time)
plt.imshow(cp.asnumpy(fdtd_grid.dielectric_grid))  # Convert from CuPy array to NumPy for plotting
plt.show()
