% Parameters
M = 5;                   % Matrix size (M x M)
d1 = ones(M, 1);         % Mx1 array for the diagonal
d1(1)=5;
% Construct the sparse matrix
A = full(spdiags(d1, 1, M, M));