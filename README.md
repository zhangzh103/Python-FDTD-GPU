# Python-FDTD-GPU  
## Version 0.1  

This is a Python implementation of the Finite-Difference Time-Domain (FDTD) algorithm. The project is inspired by the book *Electromagnetic and Photonic Simulation for the Beginner: Finite-Difference Frequency-Domain in MATLAB* by Raymond C. Rumpf ([link](https://empossible.net/fdfdbook/)). The PyTorch library is used to enable GPU acceleration. The project is currently in version 0.1, indicating it is still in the early stages of development. Note that no usable simulation is available in the current version.  

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

### 'Method.py'
Contain all the general method I wrote which used in my code.

### Other Files  
All other files are currently used for testing purposes and will be removed once the main structure of the code is complete.  

### Update
I fix the bugs in previous yeeder2d function. The bug was caused by misuse of torch.sparse_coo_tensor function. Instead, I wrote a new function in Method.py called dense_diags which add diags mimic the functionality of matlab diag function.
I also add the PML condition which implemented in PML_layer function. The PML layer function directly change the dielectric grid which is different from PBC and reflective layer.
## Next Update  
 I will test the code with a simple 2D waveguide following the process shown in Chapter 6 of Raymond's book. I will add a new sample code which demonstarate the simulation of this 2D waveguide.

## Future Updates  

The long-term goal of this project is to develop a 3D FDTD algorithm capable of simulating anisotropic, nonlinear structures while utilizing GPU acceleration. This enhancement will enable faster FDTD computations, which are essential for applications such as inverse design.  
