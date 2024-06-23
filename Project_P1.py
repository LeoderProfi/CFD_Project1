import numpy
import numpy as np
import scipy.sparse
import scipy.linalg
from matplotlib import pyplot as plt
import os
from functions_project import *
from scipy.ndimage import gaussian_filter


"""
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0

n_cells_x = n_cells_y = 40

vertices, cells = generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y)

#reference_solution_2D(vertices, n_cells_x, n_cells_y)"""


f_1 = lambda x,y: -4 *(6* x**2 - 6* x + 1)* y* (2* y**2 - 3* y + 1) - 12* (1 - x)**2 *x**2* (-1 + 2* y) + 1-2 *x
f_2 = lambda x,y: 12* (2* x - 1) *(1 - y)**2* y**2 + 4* x *(1 - 3* x + 2* x**2)* (1 - 6 *y + 6 *y**2)
g = lambda x,y: 0#x*(1-x)

boundary_conditions = {
    'left': (lambda x, y: 0, lambda x, y: 0),
    'right': (lambda x, y: 0, lambda x, y: 0),
    'bottom': (lambda x, y: 0, lambda x, y: 0),
    'top': (lambda x, y: 0, lambda x, y: 0)
}

def solve_Stokes(f_1, f_2, g, boundary_conditions, n_cells_x, n_cells_y, vertices, cells, lam, penalty = True):

    A1 = compute_global_A_1_A_2(vertices, cells)[1]
    A2 = A1.copy()

    G1 = compute_global_G(vertices, cells)[1]
    G2 = compute_global_G(vertices, cells)[2]
    B1 = -G1.T
    B2 = -G2.T
    

    matrices = [A1, A2, G1, G2]


    F1, F2, F3= compute_forcing_term_2D(f_1, f_2, g, vertices, cells)
    def handle_boundary(vertices, i_idx, j_idx, matrices, boundary_conditions):
        phi_u1 = boundary_conditions[0]
        phi_u2 = boundary_conditions[1]

        basis_indices = i_idx + (n_cells_x + 1)*j_idx

        for matrix in matrices:
            matrix[basis_indices, :] = 0.0

            if matrix is A1 or matrix is A2:
                matrix[basis_indices, basis_indices] = 1.0

        for basis_index in basis_indices:
            vertex = vertices[basis_index]
            F1[basis_index] = phi_u1(vertex[0], vertex[1])
            F2[basis_index] = phi_u2(vertex[0], vertex[1])
        return matrices
    
    # Left boundary
    i_idx = numpy.zeros(n_cells_y + 1, dtype=numpy.int64)
    j_idx = numpy.arange(0, n_cells_y + 1)
    matrices = handle_boundary(vertices,i_idx, j_idx, matrices, boundary_conditions["left"])

    # Right boundary
    i_idx = n_cells_x*numpy.ones(n_cells_y + 1, dtype=numpy.int64)
    j_idx = numpy.arange(0, n_cells_y + 1)
    matrices  = handle_boundary(vertices,i_idx, j_idx, matrices, boundary_conditions["right"])
    
    # Bottom boundary
    i_idx = numpy.arange(0, n_cells_x + 1)
    j_idx = numpy.zeros(n_cells_x + 1, dtype=numpy.int64)
    matrices  = handle_boundary(vertices,i_idx, j_idx, matrices, boundary_conditions["bottom"])
    
    # Top boundary
    i_idx = numpy.arange(0, n_cells_x + 1)
    j_idx = n_cells_y*numpy.ones(n_cells_x + 1, dtype=numpy.int64)
    matrices  = handle_boundary(vertices,i_idx, j_idx, matrices, boundary_conditions["top"])

    if penalty == True:
        
        penalty_M = 1/(lam)*compute_global_lam(vertices, cells)

        #penalty_M = 1/lam *numpy.eye((n_cells_x + 1)*(n_cells_y + 1))
        S_pen = scipy.sparse.bmat([[A1,None, G1], [None, A2, G2], [B1, B2, penalty_M]]).tocsr()
        F = numpy.hstack([F1, F2, F3])

        Solution = scipy.sparse.linalg.spsolve(S_pen, F)
        u_12 = Solution[:2*vertices.shape[0]]
        BB = scipy.sparse.bmat([[B1, B2]]).tocsr()
        Divergence = BB.dot(u_12)

        return Solution, S_pen, Divergence

    else:
        S = scipy.sparse.bmat([[A1,None, G1], [None, A2, G2], [B1, B2, None]]).tocsr()
        
        F = numpy.hstack([F1, F2, F3])
        Solution = scipy.sparse.linalg.spsolve(S, F)
        return Solution, S, None

def plot_results(X, Y, U, V, Divergence, vertices, u1, u2, p, n_cells_x, n_cells_y, penalty):

    penalty_str = 'penalty' if penalty else 'no_penalty'
    magnitude = np.sqrt(U**2 + V**2)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), magnitude, cmap='jet', alpha=0.3)
    plt.streamplot(X, Y, U, V, density=1, color=magnitude, linewidth=2, cmap='jet')
    plt.colorbar(label='Velocity Magnitude')  # Add a colorbar as a legend for magnitude
    plt.title(f"Streamplot of Velocity Field (u_x, u_y) - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'figures_P1/streamplot_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), p.reshape(n_cells_y+1, n_cells_x+1))
    plt.colorbar(label="Pressure (P)")
    plt.title(f"Pressure Field (P) - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'figures_P1/pressure_field_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.close()

    if penalty == True:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), Divergence.reshape(n_cells_y+1, n_cells_x+1))
        plt.colorbar(label="Divergence")
        plt.title(f"Divergence - Cells: {n_cells_x}x{n_cells_y}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(f'figures_P1/divergence_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
        plt.close()

x_min = y_min = 0.0
x_max = y_max = 1.0
n_cells_x = n_cells_y = [64]

penalty = True
#lam = 10
lam = 1000

for i in range(0,len(n_cells_x)):
    
    print(f"Running simulation for {n_cells_x[i]}x{n_cells_y[i]} cells")
    bas_idx = []
    vertices, cells = generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x[i], n_cells_y[i])
    u, S, Divergence= solve_Stokes(f_1, f_2, g, boundary_conditions, n_cells_x[i], n_cells_y[i], vertices, cells, lam, penalty)
    u1 = u[:vertices.shape[0]]; u2 = u[vertices.shape[0]:2*vertices.shape[0]]; p = u[2*vertices.shape[0]:]
    X = vertices[:, 0].reshape(n_cells_x[i]+1, n_cells_y[i]+1)
    Y = vertices[:, 1].reshape(n_cells_x[i]+1, n_cells_y[i]+1)
    U = u1.reshape(n_cells_x[i]+1, n_cells_y[i]+1)
    V = u2.reshape(n_cells_x[i]+1, n_cells_y[i]+1)
    plot_results(X, Y, U, V, Divergence, vertices, u1, u2, p, n_cells_x[i], n_cells_y[i], penalty)

reference_solution_2D(80,80)
"""F1, F2, F3 = compute_forcing_term_2D(f_1, f_2, g, vertices, cells)

plt.pcolormesh(X, Y, F1.reshape(n_cells_x[i]+1, n_cells_y[i]+1))
plt.colorbar()
plt.show()"""
