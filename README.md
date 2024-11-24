# Python-FDTD-GPU
#Ver 0.0

This is a Python implementation of the Finite-Difference Time-Domain (FDTD) algorithm. The PyTorch package is used to support GPU acceleration. The project is currently at version 0.0, which means it is highly unfinished. There is no usable simulation in the current version.

## File Description

### FDTD_Run.py & FDTD_Run_Ver2.py

These files contain the main body of the FDTD solver. The class `FDTD_Simulator2D` runs 2D FDTD simulations. It takes an input of type `FDTD_grid` to generate the material grid for the simulation. The class also implements Perfectly Matched Layer (PML), Periodic Boundary Conditions (PBC), and reflective boundary conditions.

`FDTD_Run_Ver2.py` has the same functionality as the original but improves code readability by adding comments and using vectorization methods to enhance performance.

### structure2D.py & structure2D_Ver2.py

These files implement the grid used for structure generation.

`structure2D_Ver2.py` has the same functionality as the original but improves code readability by adding comments and using vectorization methods to enhance performance.

### Others

All other files are currently for testing purposes and will be deleted upon the completion of the main structure of the code.

## Next Update

The next update is planned for the end of Thanksgiving. The goals include:

- Completing the main body of the code by implementing the 2D FDTD algorithm for isotropic linear materials.
- Adding visualization tools.
- Providing a grating example to demonstrate how to use the code.

## Future Update

The ultimate goal of this project is to develop a 3D FDTD algorithm for anisotropic, nonlinear structures utilizing GPU acceleration. This will enable fast FDTD computations, which are crucial for applications like inverse design.
