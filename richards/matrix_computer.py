import porepy as pp
import pygeon as pg
import numpy as np
import scipy.sparse as sps
import scipy.integrate as integrate

# Simple class used to compute the relevant matrices to be used with the FEM formulation
class Matrix_Computer:
    """
    A simple class used to generate the various matrices required to perform the FEM method.
    """
    def __init__(self, mdg, key='flow', integrate_order=5, tol=1e-5):
        self.RT0 = pg.RT0(key)
        self.P0  = pg.PwConstants(key)

        self.integrate_order = integrate_order
        self.tol = tol

        self.key = key

        self.prepared_C = False
        self.HB = []


        self.mdg = mdg
        
        self.mass_P0  = None
        self.mass_RT0 = None
        self.B = None
        self.mixed_matrix = None

        self.dof_q = []
        self.dof_psi = []


        for subdomain, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(subdomain, data, key, {
                "second_order_tensor": pp.SecondOrderTensor(np.ones(subdomain.num_cells)),
            })

            self.dof_q.append( self.RT0.ndof(subdomain) )
            self.dof_psi.append( self.P0.ndof(subdomain) )
            

        self.proj_q = []
        self.proj_psi = []

    # Helper function to generate the projection matrix of q
    def compute_proj_q_mat(self):
        """
        It returns the projection matrix from FEM to values in the center of each element for RT0
        """
        if len(self.proj_q) == 0:
            for subdomain in self.mdg.subdomains(return_data=False):
                self.proj_q.append( self.RT0.eval_at_cell_centers(subdomain) )
        return self.proj_q

    # Helper function to generate the projection matrix of psi
    def compute_proj_psi_mat(self):
        """
        It returns the projection matrix from FEM to values in the center of each element for P0
        """
        if len(self.proj_psi) == 0:
            for subdomain in self.mdg.subdomains(return_data=False):
                self.proj_psi.append( self.P0.eval_at_cell_centers(subdomain) )
        return self.proj_psi

    # Helper function to generate the projection matrix of h
    def compute_proj_h_mat(self):
        """
        It returns the projection matrix from FEM to values in the center of each element for P0
        """
        return self.compute_proj_psi_mat()

    # Helper function to project q to the cell center
    def project_q_to_solution(self, to_project: list):
        """
        It returns the projection of element in to_project from RT0 to the values in the center of each element
        """
        mats = self.compute_proj_q_mat()

        assert(len(mats) == len(to_project))

        res = []

        for matrix, vector in zip(mats, to_project):
            res.append( matrix @ vector )

        return res

    # Helper function to project psi to the cell center
    def project_psi_to_solution(self, to_project: list):
        """
        It returns the projection of element in to_project from P0 to the values in the center of each element
        """
        mats = self.compute_proj_psi_mat()

        assert(len(mats) == len(to_project))

        res = []

        for matrix, vector in zip(mats, to_project):
            res.append( matrix @ vector )
            
        return res

    # Helper function to project h to the cell center
    def project_h_to_solution(self, to_project: list):
        """
        It returns the projection of element in to_project from P0 to the values in the center of each element
        """
        return self.project_psi_to_solution(to_project)
    
    # Helper function to project a function evaluated in the cell center to FEM (scalar)
    def project_psi_to_fe(self, to_project: list):
        """
        Helper function to project the values in the center of each element to FEM P0
        """
        res = []
        for subdomain, vector in zip(self.mdg.subdomains(return_data=False), to_project):
            res.append( subdomain.cell_volumes * vector)
        return res
    
    # Helper function to project a function evaluated in the cell center to FEM (scalar)
    def project_h_to_fe(self, to_project: list):
        """
        Helper function to project the values in the center of each element to FEM P0
        """
        return self.project_psi_to_fe(to_project)

    
    # Assemble the mass matrix of P0 elements
    def mass_matrix_P0(self):
        """
        Assemble (and store internally) the P0 mass matrix
        """
        if self.mass_P0 == None:
            self.mass_P0 = []
            for subdomain in self.mdg.subdomains(return_data=False):
                self.mass_P0.append( self.P0.assemble_mass_matrix(subdomain) )
        
        return self.mass_P0
    

    # Assemble the mass matrix of RT0 elements
    def mass_matrix_RT0(self):
        """
        Assemble (and store internally) the RT0 mass matrix
        """
        if self.mass_RT0 == None:
            self.mass_RT0 = []
            for subdomain in self.mdg.subdomains(return_data=False):
                self.mass_RT0.append( self.RT0.assemble_mass_matrix(subdomain) )
        
        return self.mass_RT0
    

    # Assemble the mass matrix of RT0 elements with the conductivity tensor
    def mass_matrix_RT0_conductivity(self, tensors: list):
        """
        Assemble the RT0 mass matrix (with the conductivity tensor). The result is NOT cached.
        """
        for subdomain_data, tensor in zip(self.mdg.subdomains(return_data=True), tensors):
            _, data = subdomain_data
            data[pp.PARAMETERS][self.key].update({"second_order_tensor": tensor})
        
        return pg.face_mass(self.mdg, self.RT0, keyword=self.key)


    def __find_ordering(self, coord: np.array):
        lx = np.argmin(coord[0, :])
        rx = np.argmax(coord[0, :])
        mx = np.setdiff1d(np.array([0,1,2]), np.array([lx, rx]))[0]

        # Vertical Alignment
        if np.abs( coord[0, lx] - coord[0, mx] ) < self.tol:
            # lx and mx vertical aligned, rx no
            up =   lx if np.argmax(coord[1, np.array([lx, mx])]) == 0 else mx
            down = lx if np.argmin(coord[1, np.array([lx, mx])]) == 0 else mx

            if np.abs( coord[1, up] - coord[1, rx] ) < self.tol:
                return [up, down, rx]
            else:
                return [down, rx, up]
        else:
            # rx and mx vertical aligned, lx no
            up =   rx if np.argmax(coord[1, np.array([rx, mx])]) == 0 else mx
            down = rx if np.argmin(coord[1, np.array([rx, mx])]) == 0 else mx

            if np.abs( coord[1, up] - coord[1, lx] ) < self.tol:
                return [up, lx, down]
            else:
                return [down, up, lx]

    
    def __local_q(self, coord, sign, K11_func, K12_func, K21_func, K22_func):
        M = np.zeros(shape=(3,3))

        ordering = self.__find_ordering(coord)

        orientation = [-1, 1, -1] * sign[ordering]

        q_funcs = [lambda x,y: np.array([-x, -y]), lambda x,y: np.array([x-1, y]), lambda x,y: np.array([-x, 1-y])]

        K_local = lambda x,y: np.array([[K11_func(x,y), K12_func(x,y)],
                                         K21_func(x,y), K22_func(x,y)])

        for i in range(3):
            for j in range(3):
                integrand = lambda ys,x: np.array([q_funcs[j](x,y).T @ K_local(x, y) @ q_funcs[i](x,y) for y in np.array(ys)])
                inside = lambda xs, n: np.array([integrate.fixed_quad(integrand, 0, 1-x, args=(x,), n=n)[0] for x in np.array(xs)])
                M[ordering[i], ordering[j]] = orientation[j] * orientation[i] * integrate.fixed_quad(inside, 0, 1, n=self.integrate_order, args=(self.integrate_order,))[0]
        
        return M
    

    def mass_matrix_RT0_conductivity_manual(self, K11_func, K12_func, K21_func, K22_func):
        subdomain, data = self.mdg.subdomains(return_data=True)[0]

        faces, _, sign = sps.find(subdomain.cell_faces)

        _, _, _, _, _, node_coords = pp.map_geometry.map_grid(
                subdomain, data.get("deviation_from_plane_tol", 1e-5)
            )
        
        dim = subdomain.dim
        
        node_coords = node_coords[: dim, :]

        self.RT0._compute_cell_face_to_opposite_node(subdomain, data)
        cell_face_to_opposite_node = data[self.RT0.cell_face_to_opposite_node]
        
        size_A = np.power(subdomain.dim + 1, 2) * subdomain.num_cells
        rows_A = np.empty(size_A, dtype=int)
        cols_A = np.empty(size_A, dtype=int)
        data_A = np.empty(size_A)
        idx_A = 0

        for c in range(subdomain.num_cells):
            # For the current cell retrieve its faces
            loc = slice(subdomain.cell_faces.indptr[c], subdomain.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]
        
            # Get the opposite node id for each face (BUGGED)
            # node = cell_face_to_opposite_node[c, :]
            node = np.flip(np.sort(cell_face_to_opposite_node[c, :]))

            coord_loc = node_coords[:, node]

            #print( 'Face: ' + str(faces_loc) + ', Sign: ' + str(sign[loc]) + ', Node: ' + str(node))
            #print(coord_loc)

            A = self.__local_q(coord_loc, sign[loc], K11_func, K12_func, K21_func, K22_func)

            # Save values for Hdiv-mass local matrix in the global structure
            cols = np.concatenate(faces_loc.size * [[faces_loc]])
            loc_idx = slice(idx_A, idx_A + A.size)
            rows_A[loc_idx] = cols.T.ravel()
            cols_A[loc_idx] = cols.ravel()
            data_A[loc_idx] = A.ravel()
            idx_A += A.size

            #print('')
        
        return sps.coo_matrix((data_A, (rows_A, cols_A)))
        
    

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
        if self.B == None:
            self.B = - (self.mass_matrix_P0()[0] @ pg.div(self.mdg)).tocsc()
        return self.B
    

    # Prepare the assembly of matrix C for the Newton method with RT0 elements
    def __setup_C(self):
        """
        Prepare the common requirments to compute the C matrix (same as mass matrix of RT0, with the derivative of the inverse hydraulic conductivity)
        """
        if self.prepared_C == False:
            self.prepared_C = True
        else:
            return
        
        self.HB = []
        self.node_coords = []
        self.faces = []
        self.sign = []
        self.size_A = []
        self.cell_face_to_opposite_node = []

        for sd, data in self.mdg.subdomains(return_data=True):
            size_HB = sd.dim * (sd.dim + 1)
            self.HB.append( np.zeros((size_HB, size_HB)) )
            for it in np.arange(0, size_HB, sd.dim):
                self.HB[-1] += np.diagflat(np.ones(size_HB - it), it)
            self.HB[-1] += self.HB[-1].T
            self.HB[-1] /= sd.dim * sd.dim * (sd.dim + 1) * (sd.dim + 2)
        
            deviation_from_plane_tol = data.get("deviation_from_plane_tol", 1e-5)
            c_centers, f_normals, f_centers, R, dim, node_coords = pp.map_geometry.map_grid(sd, deviation_from_plane_tol)

            self.node_coords.append( node_coords )

            faces, cells, sign = sps.find(sd.cell_faces)
            index = np.argsort(cells)

            self.faces.append( faces[index] )
            self.sign.append( sign[index] )

            self.RT0._compute_cell_face_to_opposite_node(sd, data)
            self.cell_face_to_opposite_node.append( data["rt0_class_cell_face_to_opposite_node"] )

            self.size_A.append( np.power(sd.dim + 1, 1) * sd.num_cells )


    # Assemble the matrix C for the Newton method with RT0 elements
    def C(self, model_data, psi, q):
        """
        Assemble the C matrix (same as mass matrix of RT0, with the derivative of the inverse hydraulic conductivity).
        It will firstly call the setup function, if it was not called before.
        """
        self.__setup_C()

        res = []
        for size_A, sd_data, faces, sign, cell_face_to_opposite_node, node_coords, HB in zip(self.size_A, self.mdg.subdomains(return_data=True), self.faces, self.sign, self.cell_face_to_opposite_node, self.node_coords, self.HB):
            sd, data = sd_data
            
            cond_inv_coeff = model_data.inverse_hydraulic_conductivity_coefficient(psi, 1)

            rows_A = np.empty(size_A, dtype=int)
            cols_A = np.empty(size_A, dtype=int)
            data_A = np.empty(size_A)
            idx_A = 0

            data[pp.PARAMETERS].update({"second_order_tensor": np.ones(sd.num_cells)})

            for c in np.arange(sd.num_cells):
                loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
                faces_loc = faces[loc]
                
                node = cell_face_to_opposite_node[c, :]
                coord_loc = node_coords[:, node]

                A = pp.RT0.massHdiv(
                    np.eye(sd.dim),
                    sd.cell_volumes[c],
                    coord_loc,
                    sign[loc],
                    sd.dim,
                    HB,
                )

                # Save values for Hdiv-mass local matrix in the global structure
                loc_idx = range(idx_A, idx_A + sd.dim + 1)
                rows_A[loc_idx] = faces_loc
                cols_A[loc_idx] = c
                data_A[loc_idx] = (A @ q[faces_loc]).ravel() * cond_inv_coeff[c] / sd.cell_volumes[c]
                
                idx_A += (sd.dim + 1)

            res.append( sps.coo_matrix((data_A, (rows_A, cols_A))).tocsc() )
            
        return res