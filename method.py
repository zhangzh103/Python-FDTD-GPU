import numpy as np

def dense_diags(diagonals, offsets, matrix):
    """
    Create a dense matrix with specified diagonals, mimicking MATLAB's spdiags behavior.

    Parameters:
    - diagonals: List of 1D arrays (each diagonal's values).
    - offsets: List of integers (positions of the diagonals, 0 for main, +1 for upper, -1 for lower).
    - matrix: The matrix to be modified.

    Returns:
    - Dense matrix with specified diagonals.
    """
    n_rows, n_cols = matrix.shape
    for diag, offset in zip(diagonals, offsets):
        if offset >= 0:
            # Positive offset (super-diagonal)
            num_elements = min(n_rows, n_cols - offset)
            if num_elements <= 0:
                continue  # Skip if no valid positions
            rows = np.arange(num_elements)
            cols = rows + offset
            # Discard the first 'offset' elements of diag
            diag_elements = diag[-num_elements:]
        else:
            # Negative offset (sub-diagonal)
            num_elements = min(n_rows + offset, n_cols)
            if num_elements <= 0:
                continue  # Skip if no valid positions
            rows = np.arange(num_elements) - offset
            cols = np.arange(num_elements)
            # Discard the last '-offset' elements of diag
            diag_elements = diag[:num_elements]
        # Assign the diagonal elements
        matrix[rows, cols] = diag_elements
    return matrix
