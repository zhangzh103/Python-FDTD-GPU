# Python-FDTD-GPU  
## Version 0.0  

This is a Python implementation of the Finite-Difference Time-Domain (FDTD) algorithm. The project is inspired by the book *Electromagnetic and Photonic Simulation for the Beginner: Finite-Difference Frequency-Domain in MATLAB* by Raymond C. Rumpf ([link](https://empossible.net/fdfdbook/)). The PyTorch library is used to enable GPU acceleration. The project is currently in version 0.0, indicating it is still in the early stages of development. Note that no usable simulation is available in the current version.  

## File Description  

### `FDTD_Run.py` & `FDTD_Run_Ver2.py`  
These files contain the core implementation of the FDTD solver. The `FDTD_Simulator2D` class is responsible for running 2D FDTD simulations. It accepts an input of type `FDTD_grid` to generate the material grid for the simulation. The class also supports the following boundary conditions:
- Perfectly Matched Layer (PML)  
- Periodic Boundary Conditions (PBC)  
- Reflective boundary conditions  

`FDTD_Run_Ver2.py` includes the same functionalities as the original but improves code readability with added comments and employs vectorization techniques to enhance performance.  

### `structure2D.py` & `structure2D_Ver2.py`  
These files define the grid structures used for simulation material generation.  

`structure2D_Ver2.py` offers the same features as the original but improves code readability with added comments and uses vectorization techniques for better performance.  

### Other Files  
All other files are currently used for testing purposes and will be removed once the main structure of the code is complete.  

## Next Update  

The next update is planned for release by the end of Thanksgiving. Key goals include:  
- Completing the main body of the code by implementing the 2D FDTD algorithm for isotropic linear materials.  
- Adding visualization tools.  
- Providing a grating example to demonstrate how to use the code.  

## Future Updates  

The long-term goal of this project is to develop a 3D FDTD algorithm capable of simulating anisotropic, nonlinear structures while utilizing GPU acceleration. This enhancement will enable faster FDTD computations, which are essential for applications such as inverse design.  
