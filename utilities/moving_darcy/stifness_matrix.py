from utilities.assembly_utilities import local_A
import utilities.chi_func as helper_chi_func

import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


# Assemble the h-stifness matrix for the moving domain Darcy problem
def stifness(eta_diff, sd, boundary_grid, eta_dofs, quad_order, chi_func: helper_chi_func.Chi_Func):
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

        # Compute the stiff-H1 local matrix
        base_height    = np.min(coord_loc[1, :])
        element_height = np.max(coord_loc[1, :]) - base_height
        m = np.prod(sign[loc])

        A = local_A(coord_loc, 
                    lambda x,y: helper_chi_func.chi_quick_K_func_eval(
                        chi_func, 
                        base_height, 
                        element_height, 
                        m, 
                        eta_dofs[eta_cell], 
                        eta_dofs[eta_cell+1], 
                        grad_eta[eta_cell], 
                        1, x, y
                    ), 
                    quad_order)

        # Save values for stiff-H1 local matrix in the global structure
        cols = np.tile(nodes_loc, (nodes_loc.size, 1))
        loc_idx = slice(idx, idx + cols.size)
        rows_I[loc_idx] = cols.T.ravel()
        cols_J[loc_idx] = cols.ravel()
        data_IJ[loc_idx] = A.ravel()
        idx += cols.size

    # Construct the global matrices
    return sps.csc_matrix((data_IJ, (rows_I, cols_J)))