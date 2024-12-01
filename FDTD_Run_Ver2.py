import numpy as np
import torch
import method
from structure2D_Ver2 import FDTD_grid
import matplotlib.pyplot as plt
from scipy.sparse import diags

class FDTD_Simulator2D:

    def __init__(self, fdtd_er_grid: FDTD_grid, fdtd_ur_grid: FDTD_grid, BC, lda0):
        self.lda0=lda0
        self.BC = BC
        self.kinc = [0,0]
        #This is the part utilizing 2X method to generate the x,y,z permitivity & permibility 
        # grid based on the defined structure
        NPML=[2,2,2,2]
        self.ERxx,self.ERyy,self.ERzz,self.URxx,self.URyy,self.URzz=self.PML_layer(fdtd_er_grid, fdtd_ur_grid,NPML)
        # Compute derivative matrices
        Ny,Nx=self.ERxx.shape
        # Generate the differentia gride
        RES = [fdtd_er_grid.dx, fdtd_er_grid.dy]
        NS = [Nx, Ny]
        self.DEX, self.DEY, self.DHX, self.DHY = self.yeeder2d(NS, RES, self.BC, self.kinc)
        self.to_gpu()

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
            d0 = -np.ones(M)
            d1 = np.ones(M)
            d1[Nx:M:Nx] = 0
            DEX=np.zeros((M,M))
            #DEX = diags([d0,d1], [0,1], shape=(M, M), format='csr')/dx  # Sparse matrix in scipy
            DEX = method.dense_diags([d0,d1],[0,1],DEX)/dx
            # Incorporate Periodic Boundary Conditions
            if BC[0] == 1:
                d1 = np.zeros(M, dtype=np.complex64)
                d1[:M:Nx] = np.exp(-1j * kinc[0] * Nx * dx) / dx
                DEX = method.dense_diags([d1], [1 - Nx], DEX)

            # Create PyTorch sparse tensor
            DEX_torch = torch.tensor(DEX,dtype=torch.complex64)
        # Build DEY
        if Ny == 1:
            DEY = -1j * kinc[1] * torch.eye(M, M, dtype=torch.cfloat)
        else:
            d0 = -np.ones(M)
            d1 = np.ones(M)
            DEY=np.zeros((M,M))
            #DEX = diags([d0,d1], [0,1], shape=(M, M), format='csr')/dx  # Sparse matrix in scipy
            DEY = method.dense_diags([d0,d1],[0,Nx],DEY)/dy
            # Incorporate Periodic Boundary Conditions
            if BC[1] == 1:
                d1 = np.zeros(M, dtype=np.complex64)
                d1[:M:Nx] = np.exp(-1j * kinc[1] * Ny * dy) / dy
                DEY = method.dense_diags([d1], [Nx - M], DEY)

            # Create PyTorch sparse tensor
            DEY_torch = torch.tensor(DEY,dtype=torch.complex64)

        # Build DHX and DHY
        DHX_torch = -DEX_torch.transpose(0, 1)#need conjugate
        DHY_torch = -DEY_torch.transpose(0, 1)

        return DEX_torch, DEY_torch, DHX_torch, DHY_torch
    def PML_layer(self,fdtd_er_grid: FDTD_grid, fdtd_ur_grid: FDTD_grid,NPML):
        #The main purpose of this function is to add the PML layer to the structure
        #This function take Er grid and Ur grid and generate the x,y,z direction grid based on this
        #The x,y,z direction grid generation based on 2X method
        #Input
        #fdtd_er_grid: permitivity grid
        #fdtd_ur_grid: permibility grid
        #NPML: contain 4 numbers, correspond to [NXLO NXHI NYLO NYHI]
        #The permitivity of the layer defined like this: mu'=S*Er
        #S=[sy/sx,0,0],[0,sx/sy,0],[0,0,sx*sy]
        #sx(x)=ax(x)[1-60i*sigma_x(x)]
        #sy(x)=ay(y)[1-60i*sigma_y(y)]
        #sigma_x(x)=sigma_max*sin(pi*x/2/Lx)^2
        #sigma_y(y)=sigma_max*sin(pi*y/2/Ly)^2

        ER2 = fdtd_er_grid.dielectric_grid.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1).numpy()
        UR2 = fdtd_ur_grid.dielectric_grid.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1).numpy()
        if not ER2.shape == UR2.shape:
            raise ValueError("Shape of the ER matrix does not equal the UR matrix")
        Ny2, Nx2 = ER2.shape#in python the x y axis are flipped
        NXLO,NXHI,NYLO,NYHI=np.array(NPML)*2
        amax = 4
        cmax = 1
        p    = 3
        sx=np.zeros((Ny2,Nx2),dtype=complex)
        sy=np.zeros((Ny2,Nx2),dtype=complex)
        #Add layer based on the equation described above
        for nx in range(0,NXLO+1):
            ax=1+(amax-1)*(nx/NXLO)**p
            cx=cmax*np.sin(0.5*np.pi*nx/NXLO)**2
            sx[NXLO-nx,:]=ax*(1-1j*60*cx)
        for nx in range(0,NXHI+1):
            ax=1+(amax-1)*(nx/NXHI)**p
            cx=cmax*np.sin(0.5*np.pi*nx/NXHI)**2
            sx[Nx2-NXHI+nx-1,:]=ax*(1-1j*60*cx)
        for ny in range(0,NYLO+1):
            ay=1+(amax-1)*(ny/NYLO)**p
            cy=cmax*np.sin(0.5*np.pi*ny/NYLO)**2
            sy[:,NYLO-ny]=ay*(1-1j*60*cy)
        for ny in range(0,NYHI+1):
            ay=1+(amax-1)*(ny/NYHI)**p
            cy=cmax*np.sin(0.5*np.pi*ny/NYHI)**2
            sy[:,Ny2-NYHI+ny-1]=ay*(1-1j*60*cy)
        ER2xx = ER2/sx*sy
        ER2yy = ER2*sx/sy
        ER2zz = ER2*sx*sy

        UR2xx = UR2/sx*sy
        UR2yy = UR2*sx/sy
        UR2zz = UR2*sx*sy


        # Extract tensor elements from ER2
        ERxx = torch.tensor(ER2xx[1:Ny2:2, 0:Nx2:2]) # Starting at i=2 (index 1) and j=1 (index 0)
        ERyy = torch.tensor(ER2yy[0:Ny2:2, 1:Nx2:2])  # Starting at i=1 (index 0) and j=2 (index 1)
        ERzz = torch.tensor(ER2zz[0:Ny2:2, 0:Nx2:2])  # Starting at i=1 (index 0) and j=1 (index 0)

        # Extract tensor elements from UR2
        URxx = torch.tensor(UR2xx[0:Ny2:2, 1:Nx2:2])  # Starting at i=1 (index 0) and j=2 (index 1)
        URyy = torch.tensor(UR2yy[1:Ny2:2, 0:Nx2:2])  # Starting at i=2 (index 1) and j=1 (index 0)
        URzz = torch.tensor(UR2zz[1:Ny2:2, 1:Nx2:2])  # Starting at i=2 (index 1) and j=2 (index 1)
        return ERxx,ERyy,ERzz,URxx,URyy,URzz
    def frame_update(self):
        pass
    def to_cpu(self):
        pass
    def to_gpu(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.DEX= self.DEX.to_sparse().to(device)
        self.DEY=self.DEY.to_sparse().to(device) 
        self.DHX = self.DHX.to_sparse().to(device)
        self.DHY =self.DHY.to_sparse().to(device)
        self.ERxx =self.ERxx.to_sparse().to(device)
        self.ERyy = self.ERyy.to_sparse().to(device)
        self.ERzz = self.ERzz.to_sparse().to(device)
        self.URxx = self.URxx.to_sparse().to(device)
        self.URyy = self.URyy.to_sparse().to(device)
        self.URzz = self.URzz.to_sparse().to(device)
