from utilities.assembly_utilities import experimental_local_A, find_ordering
import utilities.chi_func as helper_chi_func

from utilities.K_func_generator import quick_K_func_eval

import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


# Assemble the h-stifness matrix for the moving domain Darcy problem
def exp_stifness(eta_diff, sd, boundary_grid, 
                 eta_dofs, quad_order, 
                 chi_x3, chi_eta,
                 h_dofs, conductivity_func, verbose=False):
    grad_eta   = eta_diff @ eta_dofs

    # Map the domain to a reference geometry (i.e. equivalent to compute
    # surface coordinates in 1d and 2d)

    _, _, sign = sps.find(sd.cell_faces)
    _, _, _, _, dim, node_coords = pp.map_geometry.map_grid(sd)

    # Allocate the data to store matrix entries, that's the most efficient
    # way to create a sparse matrix.
    size = np.power(sd.dim + 1, 2) * sd.num_cells
    rows_I = np.empty(size, dtype=int)
    cols_J = np.empty(size, dtype=int)
    data_IJ = np.empty(size)
    idx = 0

    cell_nodes = sd.cell_nodes()

    for c in np.arange(sd.num_cells):
        # For the current cell retrieve its nodes
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

        nodes_loc = cell_nodes.indices[loc]
        coord_loc = node_coords[:, nodes_loc]

        eta_cell = np.max(np.where( boundary_grid.nodes[0, :] < sd.cell_centers[0, c] ))

        ls_node = np.min(coord_loc[0, :])
        cell_width = np.max(coord_loc[0, :]) - ls_node

        ls_eta = eta_dofs[eta_cell]
        rs_eta = eta_dofs[eta_cell+1]
        eta = lambda x: ls_eta + (x - ls_node) / cell_width * (rs_eta - ls_eta)
        
        ordering, m = find_ordering(coord_loc)

        h_local = h_dofs[nodes_loc][ordering]
        ordered_coord = coord_loc[:, ordering]

        h_func = lambda x,y: (h_local[0] + 
                             (h_local[2] - h_local[0]) * (x - ordered_coord[0,0]) / (ordered_coord[0,2] - ordered_coord[0,0]) + 
                             (h_local[1] - h_local[0]) * (y - ordered_coord[1,0]) / (ordered_coord[1,1] - ordered_coord[1,0]))
        
        K_loc = lambda x,y: quick_K_func_eval( chi_x3(eta(x), y), 
                                               chi_eta(eta(x), y), 
                                               grad_eta[eta_cell], 
                                               conductivity_func(h_func(x,y), eta(x), y))

        #print(K_loc(np.mean(coord_loc[0, :]), np.mean(coord_loc[1, :])), h_func(coord_loc[0, :], coord_loc[1, :]))
        #K_loc = lambda x,y: conductivity_func(h_func(x,y), eta(x), y) * np.array([[3, 2], [2, 3]])

        A = experimental_local_A(coord_loc, K_loc, quad_order)

        # Save values for stiff-H1 local matrix in the global structure
        cols = np.tile(nodes_loc, (nodes_loc.size, 1))
        loc_idx = slice(idx, idx + cols.size)
        rows_I[loc_idx] = cols.T.ravel()
        cols_J[loc_idx] = cols.ravel()
        data_IJ[loc_idx] = A.ravel()
        idx += cols.size

    # Construct the global matrices
    return sps.csc_matrix((data_IJ, (rows_I, cols_J)))