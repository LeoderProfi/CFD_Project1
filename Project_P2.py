import numpy
import numpy as np
import scipy.sparse
import scipy.linalg
from scipy.sparse.linalg import gmres
from matplotlib import pyplot as plt
import os
from functions_project import *

x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0

n_cells_x = n_cells_y = 40

f_1 = lambda x,y: 0
f_2 = lambda x,y: 0
g = lambda x,y: 0

boundary_conditions = {
    'left': (lambda x, y: 0, lambda x, y: 0),
    'right': (lambda x, y: 0, lambda x, y: 0),
    'bottom': (lambda x, y: 0, lambda x, y: 0),
    'top': (lambda x, y: 4*x*(1-x), lambda x, y: 0)
}

def build_Stokes(f_1, f_2, g, boundary_conditions, n_cells_x, n_cells_y, vertices, cells, lam, penalty = True):

    A1 = compute_global_A_1_A_2(vertices, cells)[1]
    A2 = A1.copy()
    G1 = compute_global_G(vertices, cells)[1]
    G2 = compute_global_G(vertices, cells)[2]
    B1 = -G1.T
    B2 = -G2.T
   
    #rhs
    F1, F2, F3= compute_forcing_term_2D(f_1, f_2, g, vertices, cells)
    
    def handle_boundary(vertices, i_idx, j_idx, matrices, boundary_conditions):
        phi_u1 = boundary_conditions[0]
        phi_u2 = boundary_conditions[1]

        basis_indices = i_idx + (n_cells_x + 1)*j_idx
        
        for matrix in matrices:
            matrix[basis_indices, :] = 0.0
            matrix[basis_indices, basis_indices] = 1.0
        for basis_index in basis_indices:
            vertex = vertices[basis_index]
            F1[basis_index] = phi_u1(vertex[0], vertex[1])
            F2[basis_index] = phi_u2(vertex[0], vertex[1])
            F3[basis_index] = F1[basis_index] + F2[basis_index]
        return matrices
    
    matrices = [A1, A2, G1, G2, B1, B2]
    
    # Left boundary
    i_idx = numpy.zeros(n_cells_y + 1, dtype=numpy.int64)
    j_idx = numpy.arange(0, n_cells_y + 1)
    matrices = handle_boundary(vertices, i_idx, j_idx, matrices, boundary_conditions["left"])
    
    # Right boundary
    i_idx = n_cells_x*numpy.ones(n_cells_y + 1, dtype=numpy.int64)
    j_idx = numpy.arange(0, n_cells_y + 1)
    matrices = handle_boundary(vertices, i_idx, j_idx, matrices, boundary_conditions["right"])
    
    # Bottom boundary
    i_idx = numpy.arange(0, n_cells_x + 1)
    j_idx = numpy.zeros(n_cells_x + 1, dtype=numpy.int64)
    matrices = handle_boundary(vertices, i_idx, j_idx, matrices, boundary_conditions["bottom"])
    
    # Top boundary
    i_idx = numpy.arange(0, n_cells_x + 1)
    j_idx = n_cells_y*numpy.ones(n_cells_x + 1, dtype=numpy.int64)
    matrices = handle_boundary(vertices, i_idx, j_idx, matrices, boundary_conditions["top"])  

    if penalty == True:

        penalty = 1/lam *numpy.eye((n_cells_x + 1)*(n_cells_y + 1))
        S_pen = scipy.sparse.bmat([[A1,None, G1], [None, A2, G2], [B1, B2, penalty]]).tocsr()
        F = numpy.hstack([F1, F2, F3])

        return S_pen, F

    else:
        S = scipy.sparse.bmat([[A1,None, G1], [None, A2, G2], [B1, B2, None]]).tocsr()
        
        F = numpy.hstack([F1, F2, F3])
        return S, F

def plot_results(X, Y, U, V, Divergence, vertices, u1, u2, p, n_cells_x, n_cells_y, penalty):
    if not os.path.exists('figures_P2'):
        os.makedirs('figures_P2')

    penalty_str = 'penalty' if penalty else 'no_penalty'

    plt.figure(figsize=(10, 6))
    plt.streamplot(X, Y, U, V, density=1)
    plt.title(f"Streamplot of Velocity Field (u_x, u_1) - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'figures_P2/streamplot_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_x+1, n_cells_y+1), vertices[:, 1].reshape(n_cells_x+1, n_cells_y+1), u1.reshape(n_cells_x+1, n_cells_y+1))
    plt.colorbar(label="U1")
    plt.title(f"U1 Field - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'figures_P2/u1_field_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), u2.reshape(n_cells_y+1, n_cells_x+1))
    plt.colorbar(label="U2")
    plt.title(f"U2 Field - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'figures_P2/u2_field_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), p.reshape(n_cells_y+1, n_cells_x+1))
    plt.colorbar(label="Pressure (P)")
    plt.title(f"Pressure Field (P) - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'figures_P2/pressure_field_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.close()

    if penalty == True:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), Divergence.reshape(n_cells_y+1, n_cells_x+1))
        plt.colorbar(label="Divergence")
        plt.title(f"Divergence - Cells: {n_cells_x}x{n_cells_y}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(f'figures_P2/divergence_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
        plt.close()


x_min = y_min = 0.0
x_max = y_max = 1.0
n_cells_x = n_cells_y = 16 #[8, 16, 32, 64]

penalty = True
lam = 10000


vertices, cells = generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y)
S, F = build_Stokes(f_1, f_2, g, boundary_conditions, n_cells_x, n_cells_y, vertices, cells, lam, penalty)


Divergence = []
u = scipy.sparse.linalg.spsolve(S,F)# gmres(S, F, atol = 1e-5)[0]
u1 = u[:vertices.shape[0]]; u2 = u[vertices.shape[0]:2*vertices.shape[0]]; p = u[2*vertices.shape[0]:]
X = vertices[:, 0].reshape(n_cells_x+1, n_cells_y+1)
Y = vertices[:, 1].reshape(n_cells_x+1, n_cells_y+1)
U = u1.reshape(n_cells_x+1, n_cells_y+1)
V = u2.reshape(n_cells_x+1, n_cells_y+1)
plot_results(X, Y, U, V, Divergence, vertices, u1, u2, p, n_cells_x, n_cells_y, False)





