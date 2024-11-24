##This one is the older version which is abandoned
import numpy as np
import torch
from struct_gen import FDTD_grid
import matplotlib.pyplot as plt
class FDTD_Simulator:
    Grid=torch.tensor([[]]) 
    mey=torch.tensor([[]]) 
    mhx=torch.tensor([[]])
    E_ini=torch.tensor([[]])
    H_ini=torch.tensor([[]])
    Ey_frame=[]
    Hx_frame=[]
    dt=0
    t=0
    permitivity0=8.854187817e-12
    permeability0=1.256637061e-6
    c=3e8
    
    def __init__(self,fdtd_grid:FDTD_grid,dt,t):
        self.dt=dt
        self.t=t
        self.mey=self.c*self.t/(self.permitivity0*fdtd_grid.dielectric_grid)
        self.mhx=self.c*self.t/(self.permeability0+0*fdtd_grid.dielectric_grid)
        self.E_ini=0*fdtd_grid.dielectric_grid
        self.H_ini=0*fdtd_grid.dielectric_grid
        self.Ey_frame.append(self.E_ini)
        self.Hx_frame.append(self.H_ini)
        self.Ey_frame.append(self.E_ini)
        self.Hx_frame.append(self.H_ini)
        self.Grid=fdtd_grid
    def gaussian_source_update(self):
        pass
    def frame_update(self):
        #add source
        #pass
        #FDTD Method
        pre_Ey=self.Ey_frame[-1]
        pre_Hx=self.Hx_frame[-1]
        pre_pre_Ey=self.Ey_frame[-2]
        pre_pre_Hx=self.Hx_frame[-2]
        new_Hx=pre_Hx*0
        new_Ey=pre_Ey*0
        new_Hx[:-1,:]=pre_Hx[:-1,:]+self.mhx[:-1,:]*torch.diff(pre_Ey,dim=0)/self.Grid.dx_grid[:-1,:]
        new_Ey[1:,:]=pre_Ey[1:,:]+self.mey[1:,:]*torch.diff(pre_Ey,dim=0)/self.Grid.dy_grid[1:,:]
        #boundary condition
        new_Hx[-1,:]=pre_Hx[-1,:]+self.mhx[-1,:]*(pre_pre_Ey[-1,:]-pre_Ey[-1,:])/self.Grid.dx_grid[-1,:]
        new_Ey[0,:]=pre_Hx[0,:]+self.mey[0,:]*(pre_pre_Hx[0,:]-pre_Hx[0,:])/self.Grid.dy_grid[0,:]
        self.Ey_frame.append(new_Ey)
        self.Hx_frame.append(new_Hx)
    def FDTD_Sim(self):
        for i in np.arange(0, self.t, self.dt):
            self.frame_update()
        #self.animate_pytorch_matrices()
    def nonlinear_update():
        pass

    def animate_pytorch_matrices(self, interval=500, cmap='viridis'):
        """
        Animates a list of 2D PyTorch matrices using matplotlib's imshow, 
        and displays the current frame index on the plot.

        Parameters:
        matrix_list (list of torch.Tensor): List of 2D PyTorch tensors to animate.
        interval (int): Time between frames in milliseconds (default: 500ms).
        cmap (str): Colormap to use for imshow (default: 'viridis').

        Returns:
        None: Displays the animation.
        """
        
        # Convert PyTorch tensors to numpy arrays
        matrix_list=self.Ey_frame
        data_list = [matrix.cpu().numpy() for matrix in matrix_list]
        
        # Set up the figure and axis for the plot
        index=0
        for i in data_list:
            plt.figure(1)
            plt.imshow(i)
            plt.show(block=False)
            plt.title(index)
            plt.pause(0.05)
            index=index+1
        # Function to update the image and label in the animation
        
        # Create the animation
