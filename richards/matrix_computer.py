import porepy as pp
import pygeon as pg
import numpy as np
import scipy.sparse as sps

class Matrix_Computer:
    def __init__(self, mdg, key='flow'):
        self.RT0 = pg.RT0(key)
        self.P0  = pg.PwConstants(key)


        self.key = key

        self.prepared_C = False


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


    def compute_proj_q_mat(self):
        if len(self.proj_q) == 0:
            for subdomain in self.mdg.subdomains(return_data=False):
                self.proj_q.append( self.RT0.eval_at_cell_centers(subdomain) )
        return self.proj_q


    def compute_proj_psi_mat(self):
        if len(self.proj_psi) == 0:
            for subdomain in self.mdg.subdomains(return_data=False):
                self.proj_psi.append( self.P0.eval_at_cell_centers(subdomain) )
        return self.proj_psi

    def project_q_to_solution(self, to_project: list):
        mats = self.compute_proj_q_mat()

        assert(len(mats) == len(to_project))

        res = []

        for matrix, vector in zip(mats, to_project):
            res.append( matrix @ vector )

        return res

    def project_psi_to_solution(self, to_project: list):
        mats = self.compute_proj_psi_mat()

        assert(len(mats) == len(to_project))

        res = []

        for matrix, vector in zip(mats, to_project):
            res.append( matrix @ vector )
            
        return res
    
    def project_psi_to_fe(self, to_project: list):
        res = []
        for subdomain, vector in zip(self.mdg.subdomains(return_data=False), to_project):
            res.append( subdomain.cell_volumes * vector)
        return res

    

    def mass_matrix_P0(self):
        if self.mass_P0 == None:
            self.mass_P0 = []
            for subdomain in self.mdg.subdomains(return_data=False):
                self.mass_P0.append( self.P0.assemble_mass_matrix(subdomain) )
        
        return self.mass_P0
    


    def mass_matrix_RT0(self):
        if self.mass_RT0 == None:
            self.mass_RT0 = []
            for subdomain in self.mdg.subdomains(return_data=False):
                self.mass_RT0.append( self.RT0.assemble_mass_matrix(subdomain) )
        
        return self.mass_RT0
    

    
    def mass_matrix_RT0_conductivity(self, tensors: list):
        for subdomain_data, tensor in zip(self.mdg.subdomains(return_data=True), tensors):
            sd, data = subdomain_data
            data[pp.PARAMETERS][self.key].update({"second_order_tensor": tensor})
        
        return pg.face_mass(self.mdg, self.RT0, keyword=self.key)
    

    
    def divergence_matrix(self):
        return pg.div(self.mdg)
    

    
    def matrix_B(self):
        if self.B == None:
            self.B = - self.mass_matrix_P0()[0] @ pg.div(self.mdg)
        return self.B
    

    
    def __setup_C(self):
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



    def C(self, model_data, psi, q):
        if self.prepared_C == False:
            self.__setup_C()
            self.prepared_C = True


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

                q_loc = q[faces_loc]

                partial_C = A @ q_loc

                # Save values for Hdiv-mass local matrix in the global structure
                loc_idx = range(idx_A, idx_A + sd.dim + 1)
                rows_A[loc_idx] = faces_loc
                cols_A[loc_idx] = c
                data_A[loc_idx] = partial_C.ravel() * cond_inv_coeff[c] / sd.cell_volumes[c]
                
                idx_A += (sd.dim + 1)

            res.append( sps.coo_matrix((data_A, (rows_A, cols_A))) )
            
        return res