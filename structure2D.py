import numpy as np
import torch
class structure2D:
    #define structures in nm
    y_val=0#thickness
    x_val=np.array([0])#width
    dielectric_val=np.array([0])
    def __init__(self, x, y, dielectric_cons):
        if not isinstance(y, (int, float)) or not isinstance(x, np.ndarray) or not isinstance(dielectric_cons, np.ndarray):
            raise ValueError("Invalid input types.")
        
        if len(y)+1 != len(dielectric_cons):
            raise ValueError("The length of 'y' and 'dielectric_cons' must be the same.")
        
        self.y_val = y
        self.x_val = np.append(self.x_val, y)  # Append values to y_val
        self.dielectric_val = dielectric_cons

class FDTD_grid:#all the varibales inside here need to be tensor
    dielectric_grid=torch.tensor([[]])#dielectric constant for each pixel
    dx=0#the time value for each pixel
    dy=0#the time value for each pixel

    def __init__(self,peroid,layers,dx,dy):#layers is composed by multiple layers, layers need to be structure2D list
        ##each layer is a structure2D, the function stack the layers into dielectric_grid in y direction
        x_max=0
        y_max=0
        for i in layers:
            layer_x_grid=torch.tensor([[]])
            layer_y_grid=torch.tensor([[]])
            layer_dile_grid=torch.tensor([[]])
            for j in range(0,len(i.y_val)-1):
                x = torch.arange(0, i.x_val + dx, dx)  # Ensure the upper bound is included
                y = torch.arange(i.y_val[j], i.y_val[j+1], dy)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                grid_x=grid_x+x_max
                grid_y=grid_y
                die_grid=grid_x*0+i.dielectric_val[j]
                if layer_dile_grid.shape[0]<2:
                    layer_dile_grid=die_grid
                    layer_x_grid=grid_x
                    layer_y_grid=grid_y
                else:
                    layer_dile_grid=torch.cat((layer_dile_grid,die_grid),dim=1)
                    layer_x_grid=torch.cat((layer_x_grid,grid_x),dim=1)
                    layer_y_grid=torch.cat((layer_y_grid,grid_y),dim=1)
            x = torch.arange(0, i.x_val + dx, dx)  # Ensure the upper bound is included
            y = torch.arange(i.y_val[len(i.y_val)-1], peroid+dy, dy)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            grid_x=grid_x+x_max
            grid_y=grid_y
            x_max=torch.max(grid_x)
            y_max=torch.max(grid_y)
            die_grid=grid_x*0+i.dielectric_val[len(i.y_val)-1]
            if layer_dile_grid.shape[0]<2:
                layer_dile_grid=die_grid
                layer_x_grid=grid_x
                layer_y_grid=grid_y
            else:
                layer_dile_grid=torch.cat((layer_dile_grid,die_grid),dim=1)
                layer_x_grid=torch.cat((layer_x_grid,grid_x),dim=1)
                layer_y_grid=torch.cat((layer_y_grid,grid_y),dim=1)
            ###
            if self.dielectric_grid.shape[0]<2:
                self.dielectric_grid=layer_dile_grid
                self.dx_grid=layer_x_grid
                self.dy_grid=layer_y_grid
            else:
                self.dielectric_grid=torch.cat((self.dielectric_grid,layer_dile_grid),dim=0)
                self.dx_grid=torch.cat((self.dx_grid,layer_x_grid),dim=0)
                self.dy_grid=torch.cat((self.dy_grid,layer_y_grid),dim=0)


    def second_order_nonlinear_update():
        pass


        