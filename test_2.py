import numpy as np
from scipy.sparse import diags

M = 5
d1 = np.array([5, 1, 1, 1, 1])

# Trim the diagonal to match valid indices
d1_trimmed = d1[1:]  # Remove last element if offset = +1
A = diags([d1_trimmed], [1], shape=(M, M)).toarray()
Nx,Ny=A.shape
Nx=Nx+1
B=np.zeros((Nx,Ny))
print(A)