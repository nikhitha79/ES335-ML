#### np.lingalg.inv() computes the inverse of Z whereas np.lingalg.solve() does not compute the inverse of Z. It uses gesv (General Solve) in LAPACK (Linear Algebra Package) routine that factorizes Z using LU decomposition and then solves for theta.
##### Forward substitution - For a given equation AX=B, solve the equation LY=P^{-1}B for Y. This involves substituting the values of Y into L and is an efficient operation.
#### Backward substitution - Once Y is obtained, solve for X using UX=Y. This involves substituting the values of Y into upper triangular matrix.
#### Whereas in np.lingalg.inv, after the LU decomposition, the inverse of the matrix is calculated. Since, calculation of inverse of a matrix is computationally expensive, it is not stable. In this method, for a linear equation AX=Y, the inverse of A is computed using AX=I where I is the identity matrix. 
#####   Forward substitution - solves the equation LY=P^{-1}I for Y. L is the lower trianglular matrix, P is the perutation matrix.
#####   Backward substitution - Once Y is obtained, solve the equation UX=Y for X. U is the upper triangular matrix.
##### The solution vector obtained from UX=Y directly provides the columns of the inverse matrix A.
