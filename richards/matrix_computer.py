from utilities.triangle_integration import triangle_integration
from utilities.assembly_utilities import find_ordering

import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

# Simple class used to compute the relevant matrices to be used with the FEM formulation
class Matrix_Computer:
    """
    A simple class used to generate the various matrices required to perform the FEM method.
    """
    def __init__(self, mdg, key='flow', integrate_order=5, tol=1e-5):
        self.RT0 = pg.RT0(key)
        self.P0  = pg.PwConstants(key)
        self.P1  = pg.Lagrange1(key)

        self.integrate_order = integrate_order
        self.tol = tol

        self.key = key

        self.prepared_dual_C = False
        self.prepared_primal_C = True

        self.mdg = pg.as_mdg(mdg)        

        subdomain, data = self.mdg.subdomains(return_data=True)[0]

        pp.initialize_data(subdomain, data, key, {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(subdomain.num_cells)),
        })

        self.dof_RT0 = self.RT0.ndof(subdomain)
        self.dof_P0  = self.P0.ndof(subdomain)
        self.dof_P1  = self.P1.ndof(subdomain)


    # Helper function to generate the projection of RT0 dofs
    def create_proj_RT0(self):
        """
        It returns the projection matrix from FEM to values in the center of each element for RT0
        """

        # https://stackoverflow.com/a/610923
        try:
            return self.mat_proj_RT0
        except AttributeError:
            self.mat_proj_RT0 = self.RT0.eval_at_cell_centers(self.mdg.subdomains(return_data=False)[0])
            return self.mat_proj_RT0


    # Helper function to generate the projection of P0 dofs
    def create_proj_P0(self):
        """
        It returns the projection matrix from FEM to values in the center of each element for RT0
        """
        try:
            return self.mat_proj_P0
        except AttributeError:
            self.mat_proj_P0 = self.P0.eval_at_cell_centers(self.mdg.subdomains(return_data=False)[0])
            return self.mat_proj_P0


    # Helper function to generate the projection of P1 dofs
    def create_proj_P1(self):
        """
        It returns the projection matrix from FEM to values in the center of each element for RT0
        """
        try:
            return self.mat_proj_P1
        except AttributeError:
            self.mat_proj_P1 = self.P1.eval_at_cell_centers(self.mdg.subdomains(return_data=False)[0])
            return self.mat_proj_P1




    # Helper function to project q to the cell center
    def project_RT0_to_solution(self, to_project):
        """
        It returns the projection of element in to_project from RT0 to the values in the center of each element
        """
        return self.create_proj_RT0() @ to_project

    # Helper function to project psi to the cell center
    def project_P0_to_solution(self, to_project):
        """
        It returns the projection of element in to_project from P0 to the values in the center of each element
        """
        return self.create_proj_P0() @ to_project

    # Helper function to project psi to the cell center
    def project_P1_to_solution(self, to_project):
        """
        It returns the projection of element in to_project from P0 to the values in the center of each element
        """
        return self.create_proj_P1() @ to_project
    



    # Helper function to project a function evaluated in the cell center to FEM (scalar)
    def project_function_to_P0(self, to_project):
        """
        Helper function to project the values in the center of each element to FEM P0
        """
        return self.mdg.subdomains(return_data=False)[0].cell_volumes * to_project
    
    # Helper function to project a function evaluated in the nodes to FEM (scalar)
    def project_function_to_P1(self, to_project):
        """
        Helper function to project the values in the center of each element to FEM P0
        """
        return to_project

    



    # Assemble the mass matrix of P0 elements
    def mass_matrix_P0(self):
        """
        Assemble (and store internally) the P0 mass matrix
        """

        # https://stackoverflow.com/a/610923
        try:
            return self.mass_P0
        except AttributeError:
            self.mass_P0 = self.P0.assemble_mass_matrix(self.mdg.subdomains(return_data=False)[0])
            return self.mass_P0
    

    # Assemble the mass matrix of RT0 elements
    def mass_matrix_RT0(self):
        """
        Assemble (and store internally) the RT0 mass matrix
        """
        # https://stackoverflow.com/a/610923
        try:
            return self.mass_RT0
        except AttributeError:
            self.mass_RT0 = self.RT0.assemble_mass_matrix(self.mdg.subdomains(return_data=False)[0])
            return self.mass_RT0
        
    # Assemble the mass matrix of P1 elements
    def mass_matrix_P1(self):
        """
        Assemble (and store internally) the P0 mass matrix
        """

        # https://stackoverflow.com/a/610923
        try:
            return self.mass_P1
        except AttributeError:
            self.mass_P1 = self.P1.assemble_mass_matrix(self.mdg.subdomains(return_data=False)[0])
            return self.mass_P1
        
    

    def __integrate_local_mass_dtheta(self, model_data, coord, h, quad):
        ordering = find_ordering(coord)

        ordered_coord = coord[:, ordering]
        ordered_h = h[ordering]

        x0 = ordered_coord[:, 0]
        x1 = ordered_coord[:, 1]
        x2 = ordered_coord[:, 2]

        qs = [(lambda x,y: 1-x-y), (lambda x,y: x), (lambda x,y: y)]
        
        J = np.array([[x1[0]-x0[0], x2[0]-x0[0]],
                    [x1[1]-x0[1], x2[1]-x0[1]]])
        
        jacobian = np.linalg.det(J)
        M = np.zeros(shape=(3,3))

        h_fun   = lambda x,y: ordered_h[0] + (ordered_h[1] - ordered_h[0]) * x + (ordered_h[2] - ordered_h[0]) * y
        pos_fun = lambda x,y: x0[1] + (x1[1] - x0[1]) * x + (x2[1] - x0[1]) * y

        func = lambda x,y: model_data.theta(np.array([h_fun(x,y)]), np.array([pos_fun(x,y)]), 1)[0]

        for i in range(3):
            for j in range(3):
                M[ ordering[i], ordering[j] ] = jacobian * triangle_integration( lambda x,y: qs[j](x,y) * qs[i](x,y) * func(x,y), quad)

        return M
    
    def __quick_local_mass_dtheta(self, model_data, coord, h, quad):
        width  = np.max(coord[0, :]) - np.min(coord[0, :])
        height = np.max(coord[1, :]) - np.min(coord[1, :])

        return self.P1.local_mass(width * height / 2, 2) * model_data.theta(np.array([np.mean(h)]), np.array([np.mean(coord[1, :])]), 1)[0]

        
    # Assemble the mass matrix of P1 elements
    def mass_matrix_P1_dtheta(self, model_data, h, quad_order = 0):
        """
        Assemble (and store internally) the P1 mass matrix
        """

        subdomain = self.mdg.subdomains(return_data=False)[0]

        size = np.power(subdomain.dim + 1, 2) * subdomain.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0
        
        _, _, _, _, _, node_coords = pp.map_geometry.map_grid(subdomain)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.

        local_mass = self.__quick_local_mass_dtheta if quad_order <= 1 else self.__integrate_local_mass_dtheta

        cell_nodes = subdomain.cell_nodes()

        for c in np.arange(subdomain.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = local_mass(model_data, coord_loc, h[nodes_loc], quad_order)

            # Save values for stiff-H1 local matrix in the global structure
            cols =np.tile(nodes_loc, (nodes_loc.size, 1))

            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size
        
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))
        

        
    
    def __integrate_local_A(self, coord, h, model_data, order: int):
        ordering = find_ordering(coord)

        ordered_coord = coord[:, ordering]
        ordered_h = h[ordering]

        x0 = ordered_coord[:, 0]
        x1 = ordered_coord[:, 1]
        x2 = ordered_coord[:, 2]
        
        J_T_1_T = np.array([[x2[1]-x0[1], x0[1]-x1[1]],
                            [x0[0]-x2[0], x1[0]-x0[0]]]) / ((x1[0]-x0[0]) * (x2[1]-x0[1]) - (x2[0]-x0[0]) * (x1[1]-x0[1]))
        

        q_funcs = [J_T_1_T @ np.array([-1, -1]), J_T_1_T @ np.array([ 1, 0]), J_T_1_T @ np.array([0,  1])]

        M = np.zeros(shape=(3,3))

        h_fun   = lambda x,y: ordered_h[0] + (ordered_h[1] - ordered_h[0]) * x + (ordered_h[2] - ordered_h[0]) * y
        pos_fun = lambda x,y: x0[1] + (x1[1] - x0[1]) * x + (x2[1] - x0[1]) * y

        func = lambda x, y: model_data.hydraulic_conductivity_coefficient(np.array([h_fun(x,y)]), np.array([pos_fun(x,y)]))[0]

        jacobian = 1 / np.linalg.det( J_T_1_T.T )

        for i in range(3):
            for j in range(3):
                M[ ordering[i], ordering[j] ] = q_funcs[j].T @ q_funcs[i] * jacobian * triangle_integration(func, order)

        return M
    
    def __quick_local_A(self, coord, h, model_data, order: int):
        #element_height = (np.max(coord[1, :]) - np.min(coord[1, :]))
        #element_width  = (np.max(coord[0, :]) - np.min(coord[0, :]))

        ordering = find_ordering(coord)

        x0 = coord[:, ordering][:, 0]
        x1 = coord[:, ordering][:, 1]
        x2 = coord[:, ordering][:, 2]
        
        J_T_1_T = np.array([[x2[1]-x0[1], x0[1]-x1[1]],
                            [x0[0]-x2[0], x1[0]-x0[0]]]) / ((x1[0]-x0[0]) * (x2[1]-x0[1]) - (x2[0]-x0[0]) * (x1[1]-x0[1]))
        

        q_funcs = [J_T_1_T @ np.array([-1, -1]), J_T_1_T @ np.array([ 1, 0]), J_T_1_T @ np.array([0,  1])]

        M = np.zeros(shape=(3,3))

        jacobian = 1 / np.linalg.det( J_T_1_T.T )

        tmp = model_data.hydraulic_conductivity_coefficient( np.array([np.mean(h)]), np.array([np.mean(coord[1, :])]) )[0]

        for i in range(3):
            for j in range(3):
                M[ ordering[i], ordering[j] ] = tmp * q_funcs[j].T @ q_funcs[i] * jacobian / 2

        return M
    

    def stifness_matrix_P1_conductivity(self, h, model_data, order = 0):
        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)

        subdomain = self.mdg.subdomains(return_data=False)[0]

        _, _, _, _, _, node_coords = pp.map_geometry.map_grid(subdomain)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(subdomain.dim + 1, 2) * subdomain.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = subdomain.cell_nodes()

        local_A = self.__quick_local_A if order <= 1 else self.__integrate_local_A

        for c in np.arange(subdomain.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]


            # Compute the stiff-H1 local matrix
            A = local_A(coord_loc, h[nodes_loc], model_data, order)

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))
    

    # Assemble the mass matrix of RT0 elements with the conductivity tensor
    def mass_matrix_RT0_conductivity(self, tensor):
        """
        Assemble the RT0 mass matrix (with the conductivity tensor). The result is NOT cached.
        """
        _, data = self.mdg.subdomains(return_data=True)[0]
        data[pp.PARAMETERS][self.key].update({"second_order_tensor": tensor})
        
        return pg.face_mass(self.mdg, self.RT0, keyword=self.key)
    

    # Assemble the divergence matrix for RT0/P0
    def divergence_matrix(self):
        """
        Assemble the divergence of RT0 elements into a matrix
        """
        return pg.div(self.mdg)

    # Assemble the mixed matrix (scalar, div(vect))_{\Omega} for RT0/P0
    def matrix_B(self):
        """
        Assemle the B matrix (product between P0 and the divergence of RT0 )
        """
        # https://stackoverflow.com/a/610923
        try:
            return self.B
        except AttributeError:
            self.B = - (self.mass_matrix_P0() @ pg.div(self.mdg)).tocsc()
            return self.B
    

    # Prepare the assembly of matrix C for the Newton method with RT0 elements
    def __setup_dual_C(self):
        """
        Prepare the common requirments to compute the C matrix (same as mass matrix of RT0, with the derivative of the inverse hydraulic conductivity)
        """
        if not self.prepared_dual_C:
            self.prepared_dual_C = True
        else:
            return
        
        subdomain, data = self.mdg.subdomains(return_data=True)[0]
        dim = subdomain.dim

        size_HB = dim * (dim + 1)
        self.HB = np.zeros((size_HB, size_HB))
        for it in np.arange(0, size_HB, dim):
            self.HB += np.diagflat(np.ones(size_HB - it), it)
        self.HB += self.HB[-1].T
        self.HB /= dim * dim * (dim + 1) * (dim + 2)
        
        deviation_from_plane_tol = data.get("deviation_from_plane_tol", 1e-5)
        _, _, _, _, _, node_coords = pp.map_geometry.map_grid(subdomain, deviation_from_plane_tol)

        self.node_coords = node_coords

        faces, cells, sign = sps.find(subdomain.cell_faces)
        index = np.argsort(cells)

        self.faces = faces[index]
        self.sign  = sign[index]

        self.RT0._compute_cell_face_to_opposite_node(subdomain, data)
        self.cell_face_to_opposite_node = data["rt0_class_cell_face_to_opposite_node"]

        self.size_A = np.power(subdomain.dim + 1, 1) * subdomain.num_cells


    # Assemble the matrix C for the Newton method with RT0 elements
    def dual_C(self, model_data, h, q):
        """
        Assemble the C matrix (same as mass matrix of RT0, with the derivative of the inverse hydraulic conductivity).
        It will firstly call the setup function, if it was not called before.
        """
        self.__setup_dual_C()

        subdomain, data = self.mdg.subdomains(return_data=True)[0]

        cond_inv_coeff = model_data.inverse_hydraulic_conductivity_coefficient(h, subdomain.cell_centers[1, :], 1)

        rows_A = np.empty(self.size_A, dtype=int)
        cols_A = np.empty(self.size_A, dtype=int)
        data_A = np.empty(self.size_A)
        idx_A = 0

        data[pp.PARAMETERS].update({"second_order_tensor": np.ones(subdomain.num_cells)})

        for c in np.arange(subdomain.num_cells):
            loc = slice(subdomain.cell_faces.indptr[c], subdomain.cell_faces.indptr[c + 1])
            faces_loc = self.faces[loc]
                
            node = self.cell_face_to_opposite_node[c, :]
            coord_loc = self.node_coords[:, node]

            A = pp.RT0.massHdiv(
                np.eye(subdomain.dim),
                subdomain.cell_volumes[c],
                coord_loc,
                self.sign[loc],
                subdomain.dim,
                self.HB,
            )

            # Save values for Hdiv-mass local matrix in the global structure
            loc_idx = range(idx_A, idx_A + subdomain.dim + 1)
            rows_A[loc_idx] = faces_loc
            cols_A[loc_idx] = c
            data_A[loc_idx] = (A @ q[faces_loc]).ravel() * cond_inv_coeff[c] / subdomain.cell_volumes[c]
                
            idx_A += (subdomain.dim + 1)

        return sps.coo_matrix((data_A, (rows_A, cols_A))).tocsc()




    def __integrate_primal_local_C(self, coord, model_data, h, order = 2):
        ordering = find_ordering(coord)

        ordered_coord = coord[:, ordering]
        ordered_h = h[ordering]

        x0 = ordered_coord[:, 0]
        x1 = ordered_coord[:, 1]
        x2 = ordered_coord[:, 2]
        
        J_T_1_T = np.array([[x2[1]-x0[1], x0[1]-x1[1]],
                            [x0[0]-x2[0], x1[0]-x0[0]]]) / ((x1[0]-x0[0]) * (x2[1]-x0[1]) - (x2[0]-x0[0]) * (x1[1]-x0[1]))

        q_funcs = [J_T_1_T @ np.array([-1, -1]), J_T_1_T @ np.array([ 1, 0]), J_T_1_T @ np.array([0,  1])]
        m_funcs = [(lambda x,y: 1-x-y), (lambda x,y: x), (lambda x,y: y)]

        jacobian = 1 / np.linalg.det( J_T_1_T.T )

        grad_h = q_funcs[0] * ordered_h[0] + q_funcs[1] * ordered_h[1] + q_funcs[2] * ordered_h[2]

        M = np.zeros(shape=(3,3))

        h_fun   = lambda x,y: ordered_h[0] + (ordered_h[1] - ordered_h[0]) * x + (ordered_h[2] - ordered_h[0]) * y
        pos_fun = lambda x,y: x0[1] + (x1[1] - x0[1]) * x + (x2[1] - x0[1]) * y
        kappa = lambda x,y: model_data.hydraulic_conductivity_coefficient(np.array([h_fun(x,y)]), np.array([pos_fun(x,y)]), 1)[0]

        for i in range(3):
            for j in range(3):
                M[ ordering[i], ordering[j] ] = jacobian * q_funcs[i].T @ grad_h * triangle_integration(lambda x, y: kappa(x,y) * m_funcs[j](x,y), order)
        
        return M
    
    def __quick_primal_local_C(self, coord, model_data, h, order = 0):

        ordering = find_ordering(coord)

        x0 = coord[:, ordering][:, 0]
        x1 = coord[:, ordering][:, 1]
        x2 = coord[:, ordering][:, 2]
        
        J_T_1_T = np.array([[x2[1]-x0[1], x0[1]-x1[1]],
                            [x0[0]-x2[0], x1[0]-x0[0]]]) / ((x1[0]-x0[0]) * (x2[1]-x0[1]) - (x2[0]-x0[0]) * (x1[1]-x0[1]))

        q_funcs = [J_T_1_T @ np.array([-1, -1]), J_T_1_T @ np.array([ 1, 0]), J_T_1_T @ np.array([0,  1])]

        jacobian = 1 / np.linalg.det( J_T_1_T.T )
        ordered_h = h[ordering]

        kappa = model_data.hydraulic_conductivity_coefficient(np.array([np.mean(h)]), np.array([np.mean(coord[1, :])]), 1)[0]

        grad_h = q_funcs[0] * ordered_h[0] + q_funcs[1] * ordered_h[1] + q_funcs[2] * ordered_h[2]

        M = np.zeros(shape=(3,3))

        for i in range(3):
            for j in range(3):
                M[ ordering[i], ordering[j] ] = jacobian * kappa * q_funcs[i].T @ grad_h / 6
        
        return M


    def primal_C(self, model_data, h, quad_order = 0):
        subdomain, data = self.mdg.subdomains(return_data=True)[0]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, _, _, node_coords = pp.map_geometry.map_grid(subdomain)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(subdomain.dim + 1, 2) * subdomain.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = subdomain.cell_nodes()

        local_C = self.__quick_primal_local_C if quad_order <= 1 else self.__integrate_primal_local_C

        for c in np.arange(subdomain.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            C = local_C(coord_loc, model_data, h[nodes_loc], quad_order)

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = C.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))