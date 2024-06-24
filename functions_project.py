import numpy
import numpy as np
import scipy.sparse
import scipy.linalg
from matplotlib import pyplot as plt
import os

def generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y):
    
    n_vertices_x = n_cells_x + 1
    n_vertices_y = n_cells_y + 1
    n_cells = n_cells_x * n_cells_y

    x, y = numpy.meshgrid(numpy.linspace(x_min, x_max, n_vertices_x), numpy.linspace(y_min, y_max, n_vertices_y))
    vertices = numpy.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    cells = numpy.zeros([n_cells, 4], dtype=numpy.int64)
    for j in range(0, n_cells_y):
        for i in range(0, n_cells_x):
            k = i + n_cells_x*j  

            cells[k, 0] = (i) + (n_cells_x + 1)*(j)  # the linear index of the lower left corner of the element
            cells[k, 1] = (i+1) + (n_cells_x + 1)*(j)  # the linear index of the lower right corner of the element
            cells[k, 2] = (i) + (n_cells_x + 1)*(j+1)  # the linear index of the upper right corner of the element
            cells[k, 3] = (i+1) + (n_cells_x + 1)*(j+1)  # the linear index of the upper right corner of the element

    return vertices, cells

def reference_solution_2D( n_cells_x, n_cells_y, x_min = 0.0, x_max = 1.0, y_min = 0.0, y_max = 1.0):

    vertices, cells = generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y)
    def u_1(x, y):
        return x**2 * ((1-x)**2) * (2*y-6*y**2+4*y**3)
    def u_2(x, y):
        return -y**2 * ((1-y)**2) *(2*x-6*x**2+4*x**3)
    def p(x, y=0):
        return x*(1-x)

    def calc_u_1(vertices):
        
        return u_1(vertices[:,0], vertices[:,1])
    def calc_u_2(vertices):
        return u_2(vertices[:,0], vertices[:,1])
    def calc_p(vertices):
        return p(vertices[:,0], vertices[:,1])

    u_1_values = calc_u_1(vertices)
    u_2_values = calc_u_2(vertices)
    p_ = calc_p(vertices)

    # Reshape vertices and values
    x = vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1)
    y = vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1)
    u1 = u_1_values.reshape(n_cells_y+1, n_cells_x+1)
    u2 = u_2_values.reshape(n_cells_y+1, n_cells_x+1)

    magnitude = np.sqrt(u1**2 + u2**2)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), magnitude, cmap='jet', alpha=0.3)
    plt.streamplot(x, y, u1, u2, density=1, color=magnitude, linewidth=2, cmap='jet')
    plt.colorbar(label='Velocity Magnitude')  # Add a colorbar as a legend for magnitude
    plt.title(f"Streamplot of Velocity Field (u_x, u_y) - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'figures_P1/reference_sol_{n_cells_x}x{n_cells_y}.pdf')
    plt.close()

    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), p_.reshape(n_cells_y+1, n_cells_x+1))
    plt.colorbar(label="Pressure (P)")
    plt.title(f"Pressure Field (P) - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'figures_P1/reference_pressure_field_{n_cells_x}x{n_cells_y}.pdf')
    plt.close()

def compute_local_mass_matrix():
    """Int_omega BiBk dx = Int_Omega BjBl dy; Multiply by h"""
    M_local = numpy.zeros([2, 2])
    M_local[0, 0] = 0.5
    M_local[1, 1] = 0.5
    
    return M_local

def compute_local_stiffness_matrix():
    """Int_omega Bix Bkx dx = Int_Omega Bjy Bly dy; Multiply by 1/h"""
    N_local = numpy.ones([2, 2])
    N_local[0, 1] = -1.0
    N_local[1, 0] = -1.0
    
    return N_local

def compute_local_advection_matrix():
    """Int_omega Bix Bk dx = Int_Omega Bjy Bl dy; Multiply by 1"""
    A_local = 0.5*numpy.ones([2, 2])
    A_local[0, 0] = -0.5
    A_local[1, 0] = -0.5
    
    return A_local

def compute_A1_and_A2_local():
    A1_A2 = numpy.zeros([4, 4])
    S_local_1D = compute_local_stiffness_matrix()
    M_local_1D = compute_local_mass_matrix()
    A1_A2 = numpy.kron(S_local_1D, M_local_1D) + numpy.kron(M_local_1D, S_local_1D)
    return A1_A2

def compute_G1_local():
    G_local = numpy.zeros([4, 4])
    A_local = compute_local_advection_matrix()
    M_local = compute_local_mass_matrix()
    G_local = numpy.kron(M_local, A_local) #+ numpy.kron(M_local, A_local)
    return G_local

def compute_G2_local():
    G_local = numpy.zeros([4, 4])
    A_local = compute_local_advection_matrix()
    M_local = compute_local_mass_matrix()
    G_local = numpy.kron(A_local, M_local) #+ numpy.kron(A_local, M_local)
    return G_local

def compute_nonL_1_local():
    nonL_local = numpy.zeros([4, 4])
    A_local = compute_local_advection_matrix()
    M_local = compute_local_mass_matrix()
    nonL_local1 = numpy.kron(M_local, A_local) #+ numpy.kron(A_local, M_local)
    nonL_local2 = np.kron(M_local, M_local)
    return nonL_local1, nonL_local2

def compute_nonL_2_local():
    nonL_local = numpy.zeros([4, 4])
    A_local = compute_local_advection_matrix()
    M_local = compute_local_mass_matrix()
    nonL_local1 = numpy.kron(A_local, M_local)
    nonL_local2 = numpy.kron(M_local, M_local)
    return nonL_local1, nonL_local2


def compute_lam_local():
    lam_local = numpy.zeros([4, 4])
    M_local = compute_local_mass_matrix()
    lam_local = numpy.kron(M_local, M_local) #+ numpy.kron(M_local, A_local)
    return lam_local

def compute_global_A_1_A_2(vertices, cells):

    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]

    N_row_idx = numpy.zeros([n_cells, 4, 4])
    N_col_idx = numpy.zeros([n_cells, 4, 4]) 
    N_data = numpy.zeros([n_cells, 4, 4])

    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()

    A_local = compute_A1_and_A2_local()

    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx
        N_data[cell_idx, :, :] = A_local
    
    N_global = scipy.sparse.csr_array((N_data.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))
    
    A = scipy.sparse.bmat([[N_global, None], [None, N_global]])

    return A.tocsr(), N_global 

def compute_global_nonL_in_X(vertices, cells, u1, u2, ux, uy):
    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]

    N_row_idx = numpy.zeros([n_cells, 4, 4])
    N_col_idx = numpy.zeros([n_cells, 4, 4]) 
    N_datax1 = numpy.zeros([n_cells, 4, 4])
    N_datax2 = numpy.zeros([n_cells, 4, 4])

    N_datay1 = numpy.zeros([n_cells, 4, 4])
    N_datay2 = numpy.zeros([n_cells, 4, 4])

    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()

    nonL_local_1, nonL_local_2 = compute_nonL_1_local()
    #print(max(u1), max(u2), max(ux), max(uy), min(u1), min(u2), min(ux), min(uy))
    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx
        N_datax1[cell_idx, :, :] = nonL_local_1  * u1[cell_idx] * delta_x[cell_idx]
        
    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx
        N_datax2[cell_idx, :, :] = nonL_local_2  * ux[cell_idx] * delta_y[cell_idx] * delta_x[cell_idx]

    nonL_local_1, nonL_local_2 = compute_nonL_2_local()

    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx
        N_datay1[cell_idx, :, :] = nonL_local_1  * u2[cell_idx] * delta_y[cell_idx]
        
    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx
        N_datay2[cell_idx, :, :] = nonL_local_2  * uy[cell_idx] * delta_y[cell_idx] * delta_x[cell_idx]
    
    N_globalx1 = scipy.sparse.csr_array((N_datax1.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))
    N_globalx2 = scipy.sparse.csr_array((N_datax2.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))

    N_globaly1 = scipy.sparse.csr_array((N_datay1.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))
    N_globaly2 = scipy.sparse.csr_array((N_datay2.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))

    N1 = N_globalx1 + N_globalx2
    N2 = N_globaly1 + N_globaly2

    return  N1, N2

def compute_global_lam(vertices, cells):

    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]

    N_row_idx = numpy.zeros([n_cells, 4, 4])
    N_col_idx = numpy.zeros([n_cells, 4, 4]) 
    N_data = numpy.zeros([n_cells, 4, 4])

    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()

    lam_local = compute_lam_local()

    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx
        N_data[cell_idx, :, :] = lam_local * delta_x[cell_idx] * delta_y[cell_idx]
    
    N_global = scipy.sparse.csr_array((N_data.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))
    
    return N_global 

def compute_global_G(vertices, cells):

    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]

    N_row_idx = numpy.zeros([n_cells, 4, 4])
    N_col_idx = numpy.zeros([n_cells, 4, 4]) 
    N_data = numpy.zeros([n_cells, 4, 4])

    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()

    G1_local = compute_G1_local()
    G2_local = compute_G2_local()

    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx

        N_data[cell_idx, :, :] = G1_local * delta_x[cell_idx]

    G_1 = scipy.sparse.csr_array((N_data.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))

    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx

        N_data[cell_idx, :, :] = G2_local * delta_y[cell_idx]
        
    G_2 = scipy.sparse.csr_array((N_data.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))

    G = scipy.sparse.bmat([[G_1], [G_2]])

    B = scipy.sparse.bmat([[G_1.T, G_2.T]])
    return G.tocsr(), G_1, G_2, -B

def assemble_global_matrix(vertices, cells):
    A = compute_global_A_1_A_2(vertices, cells)[0]
    G = compute_global_G(vertices, cells)[0]
    B = compute_global_G(vertices, cells)[3]
    M = scipy.sparse.bmat([[A, -G], [B, None]])

    return M.tocsr()

def compute_forcing_term_2D(f_1, f_2, g, vertices, cells):
    
    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]
    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()
    
    F1 = numpy.zeros(n_vertices)
    F2 = numpy.zeros(n_vertices)
    F3 = numpy.zeros(n_vertices)
    

    for cell_idx, cell in enumerate(cells):
        f_at_cell_vertices = f_1(vertices[cell][:,0],vertices[cell][:,1])
        F1[cell] += 0.25 * f_at_cell_vertices * delta_x[cell_idx] * delta_y[cell_idx]

    for cell_idx, cell in enumerate(cells):
        f_at_cell_vertices = f_2(vertices[cell][:,0],vertices[cell][:,1])
        F2[cell] += 0.25 * f_at_cell_vertices * delta_x[cell_idx] * delta_y[cell_idx]

    for cell_idx, cell in enumerate(cells):
        f_at_cell_vertices = g(vertices[cell][:,0],vertices[cell][:,1])
        F3[cell] += 0.25 * f_at_cell_vertices * delta_x[cell_idx] * delta_y[cell_idx]
    
    return F1, F2, F3

def velocity_and_derivative_field(n_cells_x, n_cells_y, cells, vertices, u):
    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]
    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()
    u1 = u[:vertices.shape[0]]
    u2 = u[vertices.shape[0]:2*vertices.shape[0]]

    u_x = numpy.zeros(n_vertices)
    u_y = numpy.zeros(n_vertices)
    for vertex_idx in range(vertices.shape[0]-1):
        u_x[vertex_idx] = (u1[vertex_idx+1] - u1[vertex_idx-1])/(2*delta_x[0])
        
    for vertex_idx in range(vertices.shape[0]-1- n_cells_x):
        u_y[vertex_idx] = (u2[vertex_idx+n_cells_x+1] - u2[vertex_idx-n_cells_x-1])/(2*delta_y[0])

    return u1, u2, u_x, u_y

def compute_nonlinear_Forcing(u1, u2, du1_dx, du2_dy, vertices, cells):
    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]
    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()
    
    U1_tminus1 = numpy.zeros(n_vertices)
    U2_tminus1 = numpy.zeros(n_vertices)
    U1_dx_tminus1 = numpy.zeros(n_vertices)
    U2_dy_tminus1 = numpy.zeros(n_vertices)
    
    for cell_idx, cell in enumerate(cells):
        for i in cell:
            u1_at_cell_vertices = u1[i]
            U1_tminus1[cell] += 0.25 * u1_at_cell_vertices * delta_x[cell_idx] * delta_y[cell_idx]

    for cell_idx, cell in enumerate(cells):
        for i in cell:
            u2_at_cell_vertices = u2[i]
            U2_tminus1[cell] += 0.25 * u2_at_cell_vertices * delta_x[cell_idx] * delta_y[cell_idx]

    for cell_idx, cell in enumerate(cells):
        for i in cell:
            u1_dx_at_cell_vertices = du1_dx[i]
            U1_dx_tminus1[cell] += 0.25 * u1_dx_at_cell_vertices * delta_x[cell_idx] * delta_y[cell_idx]

    for cell_idx, cell in enumerate(cells):
        for i in cell:
            u2_dy_at_cell_vertices = du2_dy[i]
            U2_dy_tminus1[cell] += 0.25 * u2_dy_at_cell_vertices * delta_x[cell_idx] * delta_y[cell_idx]
    
    return U1_tminus1, U2_tminus1, U1_dx_tminus1, U2_dy_tminus1


def compute_nonlinear_velocity(u1, u2, du1_dx, du2_dy, vertices, cells):
    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]
    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()
    
    U1_tminus1 = numpy.zeros(n_vertices)
    U2_tminus1 = numpy.zeros(n_vertices)
    U1_dx_tminus1 = numpy.zeros(n_vertices)
    U2_dy_tminus1 = numpy.zeros(n_vertices)
    
    for cell_idx, cell in enumerate(cells):
        for i in cell:
            u1_at_cell_vertices = u1[i]
            U1_tminus1[cell] += 0.25 * u1_at_cell_vertices 

    for cell_idx, cell in enumerate(cells):
        for i in cell:
            u2_at_cell_vertices = u2[i]
            U2_tminus1[cell] += 0.25 * u2_at_cell_vertices

    for cell_idx, cell in enumerate(cells):
        for i in cell:
            u1_dx_at_cell_vertices = du1_dx[i]
            U1_dx_tminus1[cell] += 0.25 * u1_dx_at_cell_vertices 

    for cell_idx, cell in enumerate(cells):
        for i in cell:
            u2_dy_at_cell_vertices = du2_dy[i]
            U2_dy_tminus1[cell] += 0.25 * u2_dy_at_cell_vertices 
    
    return U1_tminus1, U2_tminus1, U1_dx_tminus1, U2_dy_tminus1



if __name__ == "__main__":
    import Project_P2
