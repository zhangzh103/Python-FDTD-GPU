import numpy as np
import torch

class structure2D:
    # Define structures in nm
    def __init__(self, x, y, dielectric_cons):
        if not isinstance(y, (int, float)) or not isinstance(x, np.ndarray) or not isinstance(dielectric_cons, np.ndarray):
            raise ValueError("Invalid input types.")
        
        if len(x) + 1 != len(dielectric_cons):
            raise ValueError("The length of 'x' plus one must equal the length of 'dielectric_cons'.")
        
        self.y_val = y  # Thickness
        self.x_val = x  # Positions along x
        self.dielectric_val = dielectric_cons

class FDTD_grid:
    # All variables inside here need to be tensors
    def __init__(self, period, layers, dx, dy):
        # Layers is a list of structure2D instances
        self.dx = dx
        self.dy = dy
        self.period = period
        self.layers = layers

        # Calculate total thickness
        total_thickness = sum([layer.y_val for layer in layers])

        # Determine grid dimensions
        self.nx = int(np.ceil(period / dx))
        self.ny = int(np.ceil(total_thickness / dy))

        # Initialize the dielectric grid
        self.dielectric_grid = torch.zeros((self.ny, self.nx))

        # Fill the dielectric grid based on the layers
        y_index = 0
        for layer in layers:
            layer_thickness = layer.y_val
            layer_ny = int(np.ceil(layer_thickness / dy))

            # For each y-grid cell in this layer
            for iy in range(layer_ny):
                y_pos = y_index + iy
                if y_pos >= self.ny:
                    break  # Prevent index overflow

                # Handle x positions
                if len(layer.x_val) == 0:
                    # Uniform layer along x
                    self.dielectric_grid[y_pos, :] = layer.dielectric_val[0]
                else:
                    # Include 0 and period as the start and end points
                    x_edges = np.concatenate(([0], layer.x_val, [self.period]))
                    for idx in range(len(x_edges) - 1):
                        x_start = x_edges[idx]
                        x_end = x_edges[idx + 1]
                        x_start_idx = int(np.floor(x_start / dx))
                        x_end_idx = int(np.ceil(x_end / dx))

                        # Ensure indices are within bounds
                        x_start_idx = max(0, x_start_idx)
                        x_end_idx = min(self.nx, x_end_idx)

                        # Assign dielectric constant
                        self.dielectric_grid[y_pos, x_start_idx:x_end_idx] = layer.dielectric_val[idx]

            y_index += layer_ny
    def plot_dielectric_matrix(self):
        pass
# Test code
# layer1 = structure2D(np.array([]), 10, np.array([1]))
# layer2 = structure2D(np.array([20, 40, 60, 100]), 20, np.array([1, 2, 3, 4, 4]))
# grider_er = FDTD_grid(100, [layer1, layer2], 1, 1)

# Optional: Print the dielectric grid to verify
#print(grider_er.dielectric_grid)
