import numpy
import numpy as np
import scipy.sparse
import scipy.linalg
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
import os
from functions_project import *

x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0

#n_cells_x = n_cells_y = 40

f_1 = lambda x,y: 0
f_2 = lambda x,y: 0
g = lambda x,y: 0

boundary_conditions = {
    'left': (lambda x, y: 0, lambda x, y: 0),
    'right': (lambda x, y: 0, lambda x, y: 0),
    'bottom': (lambda x, y: 0, lambda x, y: 0),
    'top': (lambda x, y: 4*x*(1-x), lambda x, y: 0)
}


def build_Stokes(f_1, f_2, g, boundary_conditions, n_cells_x, n_cells_y, vertices, cells, lam, penalty = True, nu = 1):
    A1 = nu * compute_global_A_1_A_2(vertices, cells)[1]
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
            if matrix is A1 or matrix is A2:
                matrix[basis_indices, basis_indices] = 1.0
        for basis_index in basis_indices:
            vertex = vertices[basis_index]
            F1[basis_index] = phi_u1(vertex[0], vertex[1])
            F2[basis_index] = phi_u2(vertex[0], vertex[1])
            #F3[basis_index] = F1[basis_index] + F2[basis_index]
        return matrices
    
    matrices = [A1, A2, G1, G2] 
    
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
        penalty_M = 1/lam*compute_global_lam(vertices, cells)

        #penalty = 1 /lam *numpy.eye((n_cells_x + 1)*(n_cells_y + 1)) #(delta_x[0]*delta_y[0]) 
        S_pen = scipy.sparse.bmat([[A1,None, G1], [None, A2, G2], [B1, B2, penalty_M]]).tocsr()
        F = numpy.hstack([F1, F2, F3])

        return S_pen, F, A1, A2, G1, G2, B1, B2, penalty, F1, F2, F3, penalty_M

    else:
        S = scipy.sparse.bmat([[A1,None, G1], [None, A2, G2], [B1, B2, None]]).tocsr()
        F = numpy.hstack([F1, F2, F3])

        return S, F, A1, A2, G1, G2, B1, B2, None, F1, F2, F3, None

def plot_results(X, Y, U, V, Divergence, vertices, u1, u2, p, n_cells_x, n_cells_y, penalty):
    if not os.path.exists('figures_P2'):
        os.makedirs('figures_P2')
    
    penalty_str = 'penalty' if penalty else 'no_penalty'

    magnitude = np.sqrt(U**2 + V**2)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), magnitude, cmap='jet', alpha=0.3)
    plt.streamplot(X, Y, U, V, density=1, color=magnitude, linewidth=2, cmap='jet')
    plt.colorbar(label='Velocity Magnitude')  # Add a colorbar as a legend for magnitude
    plt.title(f"Streamplot of Velocity Field (u_x, u_y) - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'figures_P2/streamplot_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.show()
    plt.close()

    """plt.figure(figsize=(10, 6))

    contour = plt.contourf(X, Y, magnitude.reshape(n_cells_y+1, n_cells_x+1), cmap='jet', alpha=0.7)
    plt.colorbar(contour, label='Velocity Magnitude') 
    plt.streamplot(X, Y, U, V, density=1, color=magnitude, linewidth=2, cmap='jet')

    plt.title(f"Contour Plot of Velocity Magnitude - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.savefig(f'figures_P2/contour_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.close()
    """
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), p.reshape(n_cells_y+1, n_cells_x+1))
    plt.colorbar(label="Pressure (P)")
    plt.title(f"Pressure Field (P) - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'figures_P2/pressure_field_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
    plt.close()

    if penalty == True and Divergence is not None:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), Divergence.reshape(n_cells_y+1, n_cells_x+1))
        plt.colorbar(label="Divergence")
        plt.title(f"Divergence - Cells: {n_cells_x}x{n_cells_y}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(f'figures_P2/divergence_{n_cells_x}x{n_cells_y}_{penalty_str}.pdf')
        plt.close()

def solve_stokes(n_cells_x, n_cells_y, penalty, lam, mode):
    x_min = y_min = 0.0
    x_max = y_max = 1.0
    vertices, cells = generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y)
    S, F, A1, A2, G1, G2, B1, B2, penalty, F1, F2, F3, penalty_M = build_Stokes(f_1, f_2, g, boundary_conditions, n_cells_x, n_cells_y, vertices, cells, lam, penalty)
    K = scipy.sparse.bmat([[A1,None], [None, A2]])
    B = scipy.sparse.bmat([[B1, B2]])
    G = scipy.sparse.bmat([[G1], [G2]])

    num_iter = [0]
    def iteration_callback(residual): 
        num_iter[0] += 1

    if mode == 'Inv_approx':
        #Approximations
        K_inv_approx = scipy.sparse.diags(1/K.diagonal())
        M_inv = scipy.sparse.bmat([[K_inv_approx, None], [None, np.identity(penalty_M.shape[0]) * lam]]) 
        u = gmres(S, F, M = M_inv, atol = 1e-5, callback = iteration_callback)

    elif mode == 'Inv_applied':
        Prec = scipy.sparse.bmat([[A1,None, None], [None, A2, None], [None, None, penalty_M]]).tocsr()
        S_prec = spsolve(Prec, S)
        F_prec = spsolve(Prec, F)
        u = gmres(S_prec, F_prec, callback = iteration_callback)

    elif mode == 'direct':
        u = spsolve(S, F)

    elif mode == 'no_prec':
        u = gmres(S, F, callback = iteration_callback)

    else:
        raise ValueError('Invalid mode')
    return u, num_iter, vertices


def Q1():
    nu =1
    for i in range(0,len(n_cells_x)):
        print(f"Running simulation for {n_cells_x[i]}x{n_cells_y[i]} cells")
        u, num_iter, vertices = solve_stokes(n_cells_x[i], n_cells_y[i], penalty, lam, mode)
        if mode != 'direct':
            u = u[0]
        print(f"Number of iterations: {num_iter}")
        u1 = u[:vertices.shape[0]]; u2 = u[vertices.shape[0]:2*vertices.shape[0]]; p = u[2*vertices.shape[0]:]
        X = vertices[:, 0].reshape(n_cells_x[i]+1, n_cells_y[i]+1)
        Y = vertices[:, 1].reshape(n_cells_x[i]+1, n_cells_y[i]+1)
        U = u1.reshape(n_cells_x[i]+1, n_cells_y[i]+1)
        V = u2.reshape(n_cells_x[i]+1, n_cells_y[i]+1)
        plot_results(X, Y, U, V, None, vertices, u1, u2, p, n_cells_x[i], n_cells_y[i], True)


def solve_NStokes(n_cells_x, n_cells_y, penalty, lam, mode, nu):
    x_min = y_min = 0.0
    x_max = y_max = 1.0
    vertices, cells = generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y)
    S, F, A1, A2, G1, G2, B1, B2, penalty, F1, F2, F3, penalty_M = build_Stokes(f_1, f_2, g, boundary_conditions, n_cells_x, n_cells_y, vertices, cells, lam, penalty, nu)
    
    num_iter = [0]
    def iteration_callback(residual): 
        num_iter[0] += 1

    # Calulate the first guess for u
    u = spsolve(S, F)

    #Newton Iteration
    it = 0
    Residual = 1
    while Residual > 1e-3:
        U_old = u.copy()
        print(f"Iteration {it}")
        u1, u2, du1_dx, du2_dy = velocity_and_derivative_field(n_cells_x, n_cells_y, cells, vertices, u)
        U1_tminus1_scaled, U2_tminus1_scaled, U1_dx_tminus1_scaled, U2_dy_tminus1_scaled = compute_nonlinear_Forcing(u1, u2, du1_dx, du2_dy, vertices, cells)
        U1_tminus1, U2_tminus1, U1_dx_tminus1, U2_dy_tminus1 = compute_nonlinear_velocity(u1, u2, du1_dx, du2_dy, vertices, cells)
        A_nonlinear_x, A_nonlinear_y = compute_global_nonL_in_X(vertices, cells, U1_tminus1, U2_tminus1, U1_dx_tminus1, U2_dy_tminus1)
        
        A1_temp = A1 + A_nonlinear_x
        A2_temp = A2 + A_nonlinear_y

        F1_temp = F1 + U1_tminus1_scaled * U1_dx_tminus1_scaled
        F2_temp = F2 + U2_tminus1_scaled * U2_dy_tminus1_scaled

        def handle_boundary(vertices, i_idx, j_idx, matrices, boundary_conditions):
            phi_u1 = boundary_conditions[0]
            phi_u2 = boundary_conditions[1]

            basis_indices = i_idx + (n_cells_x + 1)*j_idx
            
            for matrix in matrices:
                matrix[basis_indices, :] = 0.0
                if matrix is A1_temp or matrix is A2_temp:
                    matrix[basis_indices, basis_indices] = 1.0
            for basis_index in basis_indices:
                vertex = vertices[basis_index]
                F1_temp[basis_index] = phi_u1(vertex[0], vertex[1])
                F2_temp[basis_index] = phi_u2(vertex[0], vertex[1])
                #F3[basis_index] = F1[basis_index] + F2[basis_index]
            return matrices
        
        matrices = [A1_temp, A2_temp, G1, G2] 
        
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

        S_temp = scipy.sparse.bmat([[A1_temp, None, G1], [None, A2_temp, G2], [B1, B2, penalty_M]]).tocsr()
        F_temp = numpy.hstack([F1_temp, F2_temp, F3])

        diagonal = S_temp.diagonal()

        reciprocal_diagonal = 1 / diagonal
        reciprocal_diagonal[diagonal == 0] = 0 
        reciprocal_matrix = scipy.sparse.diags(reciprocal_diagonal)
        num_iter = [0]
        def iteration_callback(residual): 
            num_iter[0] += 1

        u = gmres(S_temp, F_temp, M = reciprocal_matrix, atol = 1e-5, callback = iteration_callback)[0]; print(f"Number of inner iterations: {num_iter}")

        #u = spsolve(S_temp, F_temp)
        Residual = np.linalg.norm(u - U_old)
        print(f"Residual: {Residual}")
        it += 1

    return u, num_iter, vertices


def Q2():
    for i in range(0,len(n_cells_x)):
        
        print(f"Running simulation for {n_cells_x[i]}x{n_cells_y[i]} cells")
        u, num_iter, vertices = solve_NStokes(n_cells_x[i], n_cells_y[i], penalty, lam, mode, nu)
        u1 = u[:vertices.shape[0]]; u2 = u[vertices.shape[0]:2*vertices.shape[0]]; p = u[2*vertices.shape[0]:]

        X = vertices[:, 0].reshape(n_cells_x[i]+1, n_cells_y[i]+1)
        Y = vertices[:, 1].reshape(n_cells_x[i]+1, n_cells_y[i]+1)
        U = u1.reshape(n_cells_x[i]+1, n_cells_y[i]+1)
        V = u2.reshape(n_cells_x[i]+1, n_cells_y[i]+1)
        plot_results(X, Y, U, V, None, vertices, u1, u2, p, n_cells_x[i], n_cells_y[i], True)


n_cells_x = n_cells_y = [8, 16, 32, 64]
nu = 0.01
penalty = True
lam = 10
mode = 'no_prec' #'Inv_approx'
Q2()





