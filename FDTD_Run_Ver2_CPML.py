import numpy as np
import torch
from structure2D_Ver2 import structure2D, FDTD_grid
import matplotlib.pyplot as plt
import time

class FDTD_Simulator2D:

    def __init__(self, fdtd_er_grid: FDTD_grid, fdtd_ur_grid: FDTD_grid, BC, kinc):
        self.BC = BC
        self.kinc = kinc

        # Grid resolution
        self.dx = fdtd_er_grid.dx
        self.dy = fdtd_er_grid.dy

        # Generate permittivity and permeability grids
        ER2 = fdtd_er_grid.dielectric_grid.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        UR2 = fdtd_ur_grid.dielectric_grid.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        if not ER2.shape == UR2.shape:
            raise ValueError("Shape of the ER matrix does not equal the UR matrix")
        Nx2, Ny2 = ER2.shape

        # Extract tensor elements from ER2
        self.ERxx = ER2[1:Nx2:2, 0:Ny2:2]  # Starting at i=2 (index 1) and j=1 (index 0)
        self.ERyy = ER2[0:Nx2:2, 1:Ny2:2]  # Starting at i=1 (index 0) and j=2 (index 1)
        self.ERzz = ER2[0:Nx2:2, 0:Ny2:2]  # Starting at i=1 (index 0) and j=1 (index 0)

        # Extract tensor elements from UR2
        self.URxx = UR2[0:Nx2:2, 1:Ny2:2]  # Starting at i=1 (index 0) and j=2 (index 1)
        self.URyy = UR2[1:Nx2:2, 0:Ny2:2]  # Starting at i=2 (index 1) and j=1 (index 0)
        self.URzz = UR2[1:Nx2:2, 1:Ny2:2]  # Starting at i=2 (index 1) and j=2 (index 1)

        # Compute derivative matrices
        Nx = int(Nx2 / 2)
        Ny = int(Ny2 / 2)
        self.Nx = Nx
        self.Ny = Ny

        # Generate the differential grid
        RES = [fdtd_er_grid.dx, fdtd_er_grid.dy]
        NS = [Nx, Ny]
        self.DEX, self.DEY, self.DHX, self.DHY = self.yeeder2d(NS, RES, self.BC, self.kinc)

        # Time step (CFL condition)
        c = 1  # Speed of light in normalized units
        self.dt = 0.99 / (c * np.sqrt((1 / self.dx) ** 2 + (1 / self.dy) ** 2))

        # Initialize field variables
        self.Ez = torch.zeros((Nx, Ny), dtype=torch.cfloat)
        self.Hx = torch.zeros((Nx, Ny), dtype=torch.cfloat)
        self.Hy = torch.zeros((Nx, Ny), dtype=torch.cfloat)

        # Set PML parameters
        self.npml_x = min(20, int(Nx / 10))
        self.npml_y = min(20, int(Ny / 10))

        # Initialize PML parameters
        self.sigma_e_x, self.kappa_e_x, self.alpha_e_x = self.compute_pml_parameters(Nx, self.npml_x, self.dx)
        self.sigma_e_y, self.kappa_e_y, self.alpha_e_y = self.compute_pml_parameters(Ny, self.npml_y, self.dy)
        self.sigma_h_x, self.kappa_h_x, self.alpha_h_x = self.compute_pml_parameters(Nx, self.npml_x, self.dx)
        self.sigma_h_y, self.kappa_h_y, self.alpha_h_y = self.compute_pml_parameters(Ny, self.npml_y, self.dy)

        # Compute CPML coefficients for E field
        eps_0 = 1.0  # In normalized units
        self.b_e_x = torch.exp(-(self.sigma_e_x / self.kappa_e_x + self.alpha_e_x) * self.dt / eps_0)
        self.c_e_x = self.sigma_e_x * (self.b_e_x - 1) / (self.sigma_e_x + self.kappa_e_x * self.alpha_e_x)
        self.b_e_y = torch.exp(-(self.sigma_e_y / self.kappa_e_y + self.alpha_e_y) * self.dt / eps_0)
        self.c_e_y = self.sigma_e_y * (self.b_e_y - 1) / (self.sigma_e_y + self.kappa_e_y * self.alpha_e_y)

        # Compute CPML coefficients for H field
        mu_0 = 1.0  # In normalized units
        self.b_h_x = torch.exp(-(self.sigma_h_x / self.kappa_h_x + self.alpha_h_x) * self.dt / mu_0)
        self.c_h_x = self.sigma_h_x * (self.b_h_x - 1) / (self.sigma_h_x + self.kappa_h_x * self.alpha_h_x)
        self.b_h_y = torch.exp(-(self.sigma_h_y / self.kappa_h_y + self.alpha_h_y) * self.dt / mu_0)
        self.c_h_y = self.sigma_h_y * (self.b_h_y - 1) / (self.sigma_h_y + self.kappa_h_y * self.alpha_h_y)

        # Initialize CPML auxiliary variables
        self.psi_Ez_x = torch.zeros((Nx, Ny), dtype=torch.cfloat)
        self.psi_Ez_y = torch.zeros((Nx, Ny), dtype=torch.cfloat)
        self.psi_Hx_y = torch.zeros((Nx, Ny), dtype=torch.cfloat)
        self.psi_Hy_x = torch.zeros((Nx, Ny), dtype=torch.cfloat)

    def compute_pml_parameters(self, N, npml, d, m=3.5, sigma_max=None, kappa_max=5, alpha_max=0.05):
        sigma = torch.zeros(N)
        kappa = torch.ones(N)
        alpha = torch.zeros(N)

        # If sigma_max is not provided, compute it
        if sigma_max is None:
            sigma_max = (m + 1) * 0.8 / (2 * npml * d)

        for i in range(N):
            if i < npml:
                x = (npml - i) / npml
                sigma[i] = sigma_max * x ** m
                kappa[i] = 1 + (kappa_max - 1) * x ** m
                alpha[i] = alpha_max * (1 - x)
            elif i >= N - npml:
                x = (i - (N - npml - 1)) / npml
                sigma[i] = sigma_max * x ** m
                kappa[i] = 1 + (kappa_max - 1) * x ** m
                alpha[i] = alpha_max * (1 - x)
            else:
                sigma[i] = 0
                kappa[i] = 1
                alpha[i] = 0

        return sigma, kappa, alpha

    def update_H(self):
        # Compute the curl of E
        # For Hx update: need dEz/dy
        # For Hy update: need dEz/dx

        # Compute dEz/dy
        dEz_dy = torch.zeros_like(self.Ez, dtype=torch.cfloat)
        dEz_dy[:, :-1] = (self.Ez[:, 1:] - self.Ez[:, :-1]) / self.dy
        if self.BC[1] == 1:
            dEz_dy[:, -1] = (self.Ez[:, 0] - self.Ez[:, -1]) / self.dy

        # Compute dEz/dx
        dEz_dx = torch.zeros_like(self.Ez, dtype=torch.cfloat)
        dEz_dx[:-1, :] = (self.Ez[1:, :] - self.Ez[:-1, :]) / self.dx
        if self.BC[0] == 1:
            dEz_dx[-1, :] = (self.Ez[0, :] - self.Ez[-1, :]) / self.dx

        # Update CPML variables
        # For Hx update
        self.psi_Hx_y *= self.b_h_y.unsqueeze(0)
        self.psi_Hx_y += self.c_h_y.unsqueeze(0) * dEz_dy

        # For Hy update
        self.psi_Hy_x *= self.b_h_x.unsqueeze(1)
        self.psi_Hy_x += self.c_h_x.unsqueeze(1) * dEz_dx

        # Update Hx
        self.Hx -= (self.dt / self.URxx) * ((1 / self.kappa_h_y.unsqueeze(0)) * dEz_dy + self.psi_Hx_y)

        # Update Hy
        self.Hy += (self.dt / self.URyy) * ((1 / self.kappa_h_x.unsqueeze(1)) * dEz_dx + self.psi_Hy_x)

    def update_E(self):
        # Compute the curl of H
        # For Ez update: need dHy/dx - dHx/dy

        # Compute dHy/dx
        dHy_dx = torch.zeros_like(self.Hy, dtype=torch.cfloat)
        dHy_dx[1:, :] = (self.Hy[1:, :] - self.Hy[:-1, :]) / self.dx
        if self.BC[0] == 1:
            dHy_dx[0, :] = (self.Hy[0, :] - self.Hy[-1, :]) / self.dx

        # Compute dHx/dy
        dHx_dy = torch.zeros_like(self.Hx, dtype=torch.cfloat)
        dHx_dy[:, 1:] = (self.Hx[:, 1:] - self.Hx[:, :-1]) / self.dy
        if self.BC[1] == 1:
            dHx_dy[:, 0] = (self.Hx[:, 0] - self.Hx[:, -1]) / self.dy

        # Update CPML variables
        # For Ez update in x
        self.psi_Ez_x *= self.b_e_x.unsqueeze(1)
        self.psi_Ez_x += self.c_e_x.unsqueeze(1) * dHy_dx

        # For Ez update in y
        self.psi_Ez_y *= self.b_e_y.unsqueeze(0)
        self.psi_Ez_y += self.c_e_y.unsqueeze(0) * dHx_dy

        # Update Ez
        self.Ez += (self.dt / self.ERzz) * (
            (1 / self.kappa_e_x.unsqueeze(1)) * dHy_dx + self.psi_Ez_x -
            (1 / self.kappa_e_y.unsqueeze(0)) * dHx_dy - self.psi_Ez_y
        )
    def yeeder2d(self, NS, RES, BC, kinc):
        Nx, Ny = NS
        dx, dy = RES

        if kinc is None:
            kinc = [0, 0]

        M = Nx * Ny

        # Build DEX
        if Nx == 1:
            DEX = -1j * kinc[0] * torch.eye(M, M, dtype=torch.cfloat)
        else:
            # Initialize indices and values for DEX
            row_indices = []
            col_indices = []
            values = []

            # Main diagonal
            indices = torch.arange(M)
            row_indices.extend(indices.tolist())
            col_indices.extend(indices.tolist())
            values.extend((-1 / dx) * np.ones(M))

            # Upper diagonal
            indices = torch.arange(M - 1)
            # Exclude the last element of each row for Dirichlet BCs
            mask = torch.ones(M - 1, dtype=bool)
            mask[Nx - 1::Nx] = False  # Zero at positions where columns wrap to next row
            indices = indices[mask]
            row_indices.extend(indices.tolist())
            col_indices.extend((indices + 1).tolist())
            values.extend((1 / dx) * np.ones(len(indices)))

            # Incorporate Periodic Boundary Conditions
            if BC[0] == 1:
                # Add entries for periodic boundary conditions
                indices = torch.arange(Nx - 1, M, Nx)
                row_indices.extend(indices.tolist())
                col_indices.extend((indices - (Nx - 1)).tolist())
                phase = np.exp(-1j * kinc[0] * Nx * dx)
                values.extend((1 / dx) * phase * np.ones(len(indices)))

            # Create sparse tensor
            indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
            values = torch.tensor(values, dtype=torch.cfloat)
            DEX = torch.sparse_coo_tensor(indices, values, size=(M, M), dtype=torch.cfloat)

        # Build DEY
        if Ny == 1:
            DEY = -1j * kinc[1] * torch.eye(M, M, dtype=torch.cfloat)
        else:
            # Initialize indices and values for DEY
            row_indices = []
            col_indices = []
            values = []

            # Main diagonal
            indices = torch.arange(M)
            row_indices.extend(indices.tolist())
            col_indices.extend(indices.tolist())
            values.extend((-1 / dy) * np.ones(M))

            # Upper diagonal (offset by Nx)
            indices = torch.arange(M - Nx)
            row_indices.extend(indices.tolist())
            col_indices.extend((indices + Nx).tolist())
            values.extend((1 / dy) * np.ones(len(indices)))

            # Incorporate Periodic Boundary Conditions
            if BC[1] == 1:
                # Add entries for periodic boundary conditions
                indices = torch.arange(M - Nx, M)
                row_indices.extend(indices.tolist())
                col_indices.extend((indices % Nx).tolist())
                phase = np.exp(-1j * kinc[1] * Ny * dy)
                values.extend((1 / dy) * phase * np.ones(len(indices)))

            # Create sparse tensor
            indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
            values = torch.tensor(values, dtype=torch.cfloat)
            DEY = torch.sparse_coo_tensor(indices, values, size=(M, M), dtype=torch.cfloat)

        # Build DHX and DHY
        DHX = -DEX.transpose(0, 1)#need conjugate
        DHY = -DEY.transpose(0, 1)

        return DEX.coalesce(), DEY.coalesce(), DHX.coalesce(), DHY.coalesce()

    # ... (existing methods like yeeder2d)

    # (Make sure to adjust any indexing issues that might arise)

##test
# Time-stepping loop
nsteps = 1000  # Number of time steps
# Define layers with swapped x and y
layer1 = structure2D(np.array([2,4]),4, np.array([1,2,3]))
# Create FDTD grids
grider_er = FDTD_grid(10, [layer1], 0.2, 0.1)
layer1 = structure2D(np.array([]),4, np.array([1]))
grider_ur = FDTD_grid(10, [layer1], 0.2, 0.1)

end_time = time.time()

# Initialize FDTD Simulator (assuming it's properly defined)
fdtd = FDTD_Simulator2D(grider_er, grider_ur, [0, 0], [2, 3])
for n in range(nsteps):
    # Update H fields
    fdtd.update_H()

    # Update E fields
    fdtd.update_E()

    # Add source if necessary
    # For example, a point source at position (ix, iy)
    # fdtd.Ez[ix, iy] += source_function[n]

    # Optional: Visualization or data saving
    if n % 50 == 0:
        plt.imshow(fdtd.Ez.real.cpu(), cmap='RdBu')
        plt.title(f'Ez at time step {n}')
        plt.colorbar()
        plt.pause(0.01)
        plt.clf()

plt.show()
