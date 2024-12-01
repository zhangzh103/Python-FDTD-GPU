import dask.array as da
import torch

# Step 1: Create very large matrices using Dask
N = 20000  # Matrix size (rows/columns)
M = 20000    # Number of right-hand sides (columns of B)

# Create large random matrices A and B using Dask
A = da.random.random((N, N), chunks=(10000, 10000))  # Chunk size manageable in memory
B = da.random.random((N, M), chunks=(10000, M))

# Step 2: Define a function to solve chunks of the matrix equation
def solve_chunk(A_chunk, B_chunk):
    # Convert Dask chunks to PyTorch tensors on GPU
    A_tensor = torch.tensor(A_chunk, dtype=torch.float32, device='cuda')
    B_tensor = torch.tensor(B_chunk, dtype=torch.float32, device='cuda')
    
    # Solve the system using PyTorch (X = A^-1 B)
    X_tensor = torch.linalg.solve(A_tensor, B_tensor)
    
    # Move the solution back to CPU and convert to NumPy
    return X_tensor.cpu().numpy()

# Step 3: Map the function across chunks of A and B using Dask
X = da.blockwise(
    solve_chunk,
    'ij',
    A, 'ik',
    B, 'kj',
    dtype=float,
    adjust_chunks={'ij': (10000, M)}  # Ensure chunks match solution dimensions
)

# Step 4: Compute the result
X_result = X.compute()  # Trigger computation and get the full solution
print("Solution matrix X shape:", X_result.shape)
