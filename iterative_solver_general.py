import numpy as np
import matplotlib.pyplot as plt


# General Iterative Methods

def jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Solve Ax = b using Jacobi iteration.
    
    Parameters:
        A : np.array, coefficient matrix (n x n)
        b : np.array, right-hand side vector (n,)
        x0 : np.array, initial guess (default zeros)
        tol : float, convergence tolerance
        max_iter : int, maximum iterations
    
    Returns:
        x : np.array, solution
        errors : list, error at each iteration
    """
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    D = np.diag(np.diag(A))
    R = A - D
    errors = []
    
    for k in range(max_iter):
        x_new = np.linalg.inv(D) @ (b - R @ x)
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        x = x_new
        if error < tol:
            break
    return x, errors

def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Solve Ax = b using Gauss-Seidel iteration.
    """
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    errors = []
    
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        x = x_new
        if error < tol:
            break
    return x, errors


# Example: 4x4 System

A_example = np.array([[4, -1, 0, 0],
                      [-1, 4, -1, 0],
                      [0, -1, 4, -1],
                      [0, 0, -1, 3]], dtype=float)
b_example = np.array([15, 10, 10, 10], dtype=float)

# Solve
x_jacobi, err_jacobi = jacobi(A_example, b_example)
x_gs, err_gs = gauss_seidel(A_example, b_example)

print("Jacobi Solution:", x_jacobi)
print("Gauss-Seidel Solution:", x_gs)


# Plot convergence

plt.figure(figsize=(8,5))
plt.semilogy(err_jacobi, 'o-', label='Jacobi', linewidth=2)
plt.semilogy(err_gs, 's-', label='Gauss-Seidel', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Infinity Norm of Error', fontsize=12)
plt.title('Convergence of Iterative Methods', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
