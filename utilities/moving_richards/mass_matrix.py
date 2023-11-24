from utilities.assembly_utilities import experimental_local_Mh

import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


# Assemble the h-stifness matrix for the moving domain Darcy problem
def global_mass(subdomain, boundary_grid, eta_dofs, quad_order, chi_x3):
    size = np.power(subdomain.dim + 1, 2) * subdomain.num_cells
    rows_I = np.empty(size, dtype=int)
    cols_J = np.empty(size, dtype=int)
    data_IJ = np.empty(size)
    idx = 0

    _, _, _, _, _, node_coords = pp.map_geometry.map_grid(subdomain)

    # Allocate the data to store matrix entries, that's the most efficient
    # way to create a sparse matrix.

    cell_nodes = subdomain.cell_nodes()

    for c in np.arange(subdomain.num_cells):
        # For the current cell retrieve its nodes
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

        nodes_loc = cell_nodes.indices[loc]
        coord_loc = node_coords[:, nodes_loc]

        eta_cell = np.max(np.where( boundary_grid.nodes[0, :] < subdomain.cell_centers[0, c] ))
        base_height    = np.min(coord_loc[1, :])
        element_height = np.max(coord_loc[1, :]) - base_height

        ls_node = np.min(coord_loc[0, :])
        cell_width = np.max(coord_loc[0, :]) - ls_node

        ls_eta = eta_dofs[eta_cell]
        rs_eta = eta_dofs[eta_cell+1]
        eta = lambda x: ls_eta + (x - ls_node) / cell_width * (rs_eta - ls_eta)

        loc_term = lambda x,y: chi_x3(eta(x), y)
                                                                
        A = experimental_local_Mh(coord_loc, loc_term, quad_order)

        # Save values for stiff-H1 local matrix in the global structure
        cols = np.tile(nodes_loc, (nodes_loc.size, 1))

        loc_idx = slice(idx, idx + cols.size)
        rows_I[loc_idx] = cols.T.ravel()
        cols_J[loc_idx] = cols.ravel()
        data_IJ[loc_idx] = A.ravel()
        idx += cols.size

    # Construct the global matrices
    return sps.csc_matrix((data_IJ, (rows_I, cols_J)))