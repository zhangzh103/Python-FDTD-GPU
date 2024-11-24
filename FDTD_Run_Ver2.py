import numpy as np
import torch
from structure2D_Ver2 import FDTD_grid
import matplotlib.pyplot as plt

class FDTD_Simulator2D:

    def __init__(self, fdtd_er_grid: FDTD_grid, fdtd_ur_grid: FDTD_grid, BC, kinc):
        self.BC = BC
        self.kinc = kinc
        #This is the part utilizing 2X method to generate the x,y,z permitivity&permibility 
        # grid based on the defined structure
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
        # Generate the differentia gride
        RES = [fdtd_er_grid.dx, fdtd_er_grid.dy]
        NS = [Nx, Ny]
        self.DEX, self.DEY, self.DHX, self.DHY = self.yeeder2d(NS, RES, self.BC, self.kinc)

    def yeeder2d(self, NS, RES, BC, kinc):
        #This is the function to generate the differentiate gride based on the shape

        # Input matrix:
# [
#     H~x|1,1   
#     H~x|2,1   
#     H~x|3,1
#     H~x|1,2   
#     H~x|2,2   
#     H~x|3,2
#     H~x|1,3   
#     H~x|2,3   
#     H~x|3,3
# ]

# Derivative matrix D_y^h: Notice, without boundary condition
# 1/Δy * [
#     1  0  0  0  0  0  0  0  0
#     0  1  0  0  0  0  0  0  0
#     0  0  1  0  0  0  0  0  0
#    -1  0  0  1  0  0  0  0  0
#     0 -1  0  0  1  0  0  0  0
#     0  0 -1  0  0  1  0  0  0
#     0  0  0 -1  0  0  1  0  0
#     0  0  0  0 -1  0  0  1  0
#     0  0  0  0  0 -1  0  0  1
# ]

# Result of multiplying the derivative matrix with the input matrix:
# 1/Δy * [
#     H~x|1,1 - 0
#     H~x|2,1 - 0
#     H~x|3,1 - 0
#     H~x|1,2 - H~x|1,1
#     H~x|2,2 - H~x|2,1
#     H~x|3,2 - H~x|3,1
#     H~x|1,3 - H~x|1,2
#     H~x|2,3 - H~x|2,2
#     H~x|3,3 - H~x|3,2
# ]
        """
1. Differentiation Operators (DEX, DEY):
   - These sparse matrices approximate spatial derivatives of field components in the x and y directions.
   - Their structure follows the discrete derivative expressions:
     - For electric fields in the x-direction:
       [Ez(i, j+1) - Ez(i, j)]/dy = mu_xx(i, j) * Hx(i, j)
     - For electric fields in the y-direction:
       [Ez(i+1, j) - Ez(i, j)]/dx = mu_yy(i, j) * Hy(i, j)
     - For magnetic fields:
       [Hy(i, j) - Hy(i, j-1)]/dx - [Hx(i, j) + Hx(i-1, j)]/dy = eps_zz(i, j) * Ez(i, j)
   - Dirichlet and periodic boundary conditions are incorporated, modifying the derivative matrices accordingly.

2. Boundary Conditions:
   - Dirichlet boundary conditions are reflected by specific matrix entries adjusted at grid edges.
   - Periodic boundary conditions are implemented for continuity across boundaries, incorporating phase shifts based on the incident wave vector, kinc.
    dEz/dx
    [Ez(i+1,j)-Ez(i,j)]/dx i<N
    [Phix * Ez(1,j)-Ez(N,j)]/dx i=N
    Phix is the phase difference for oblique incidence, this is the phase difference in x direction
3. Construction of Sparse Matrices:
   - The function efficiently constructs sparse derivative matrices (DEX, DEY) using the torch.sparse_coo_tensor format.
   - These matrices approximate the derivative operators for the fields on the Yee grid:
     - D_ex * Ez = mu_xx * Hx
     - D_ey * Ez = mu_yy * Hy
     - D_hx * Hy - D_hy * Hx = eps_zz * Ez

4. Conjugate Transpose for Magnetic Fields (DHX, DHY):
   - The derivative matrices for the magnetic field components (DHX, DHY) are computed as the conjugate transpose of their electric field counterparts.

Inputs:
- NS: Grid size (number of cells in x and y directions).
- RES: Resolution (grid spacing in x and y directions).
- BC: Boundary conditions ([0: Dirichlet, 1: Periodic] for x and y).
- kinc: Incident wave vector (phase factor for periodic boundaries).

Outputs:
- Sparse matrices (DEX, DEY, DHX, DHY) representing discrete derivative operators for the Yee grid.
"""

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
