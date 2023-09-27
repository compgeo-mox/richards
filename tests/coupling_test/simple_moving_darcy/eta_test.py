# %%
import numpy as np
import scipy.sparse as sps
from math import ceil, floor, log10, exp, isnan
import os, shutil

import porepy as pp
import pygeon as pg

# %%

# %% [markdown]
# We create now the grid, since we will use a Raviart-Thomas approximation for ${q}$ we are restricted to simplices. In this example we consider a 2-dimensional structured grid, but the presented code will work also in 1d and 3d. PyGeoN works with mixed-dimensional grids, so we need to convert the grid.

# %%
T = 10
S_s = 0.1
phi = 0.1


for N in [10, 25, 50, 100]:
    for ll in range(6):

        dt = 1 / np.power(2, ll)
        
        output_directory = str(dt) + '_' + str(N) + '_' + 'output'

        # %%
        abs_tol = 1e-6
        rel_tol = 1e-6
        max_iterations_per_step = 100

        # %%
        # convert the grid into a mixed-dimensional grid
        sd = pp.StructuredTriangleGrid([N, N], [1, 1])
        sd.compute_geometry()

        # %%
        boundary_grid, boundary_face_map, boundary_node_map = pp.partition.extract_subgrid(sd, sd.face_centers[1, :] == 1, faces=True)

        # %%
        mdg = pp.meshing.subdomains_to_mdg([sd])

        # %% [markdown]
        # With the following code we set the data, in particular the permeability tensor and the boundary conditions. Since we need to identify each side of $\partial \Omega$ we need few steps.

        # %%
        key = "flow"

        darcy_data = {}
        richards_data = {}

        bc_val = []
        bc_ess = []
        initial_pressure = []

        # %%
        velocity_discretization_field = pg.RT0(key)
        boundary_discretization_field = pg.Lagrange1(key)
        head_discretization_field     = pg.PwConstants(key)

        # %%
        subdomain, data = mdg.subdomains(return_data=True)[0]

        # %%
        pp.initialize_data(subdomain, data, key, {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(subdomain.num_cells)),
        })

        # %%
        # Usual BC (no slip on the left and right, fixed unitary head on the bottom. No condition on the top boundary)
        left_right = np.logical_or(sd.face_centers[0, :] == 0,  sd.face_centers[0, :] == 1)

        bottom = sd.face_centers[1, :] == 0

        ess_p_dofs = np.zeros(head_discretization_field.ndof(sd), dtype=bool)

        def h_bc(x, t): return 1
        def initial_h_func(x): return 1
        def infiltration(x): return 0#1e-3

        bc_val = lambda t: -velocity_discretization_field.assemble_nat_bc(sd, lambda x: h_bc(x,t), bottom)
        bc_ess.append(np.hstack((left_right, ess_p_dofs, np.zeros(N+1, dtype=bool))))

        # %% [markdown]
        # The $RT_0$ elements are constructed in such a wat that, on the non-diagonal faces, $\bm{v} \cdot \nu = 0$. Then, since $\Gamma$ is made by one of the two catheti of each boundary facing element, the only non-zero terms will be associated to the basis functions associated to the ``boundary'' cathetus. Then:

        # %% [markdown]
        # $$
        # \begin{bmatrix} -x \\ -y \end{bmatrix} \cdot \begin{bmatrix}-1 \\ 0 \end{bmatrix} = 0 \text{ on } \Gamma
        # $$

        # %% [markdown]
        # $$
        # \begin{bmatrix} -x \\ 1-y \end{bmatrix} \cdot \begin{bmatrix}-1 \\ 0 \end{bmatrix} = 0 \text{ on } \Gamma
        # $$

        # %% [markdown]
        # $$
        # \begin{bmatrix} x-1 \\ y \end{bmatrix} \cdot \begin{bmatrix}-1 \\ 0 \end{bmatrix} \neq 0 \text{ on } \Gamma
        # $$

        # %% [markdown]
        # $$
        # \int_0^1 \left| \bm{x}_1 - \bm{x}_0 \right| \begin{bmatrix} -1 \\ s \end{bmatrix} \cdot \begin{bmatrix}-1 \\ 0 \end{bmatrix} s ds = \int_0^1 \left| \bm{x}_1 - \bm{x}_0 \right| \begin{bmatrix} -1 \\ s \end{bmatrix} \cdot \begin{bmatrix}-1 \\ 0 \end{bmatrix} (1-s) ds = \frac{\left| \bm{x}_1 - \bm{x}_0 \right|}{2}
        # $$

        # %%
        def assemble_B_gamma():
            data = []
            row = []
            col = []

            # Take the x-coordinate of each face center
            faces_center_pos = sd.face_centers[0,:]

            # Look for the boundary faces ids
            index_up_face = np.where(sd.face_centers[1, :] == 1)[0]

            # Loop thorough the boundary faces
            for i in range(N):
                # s-element
                col.append(index_up_face[i])
                row.append(i)
                data.append( np.abs(faces_center_pos[i] - faces_center_pos[i+1]) / 2 )
                
                # (1-s)-element
                col.append(index_up_face[i])
                row.append(i+1)
                data.append( np.abs(faces_center_pos[i] - faces_center_pos[i+1]) / 2 )
            
            return sps.coo_matrix( (data, (row, col)), shape=(N+1, sd.num_faces) )

        # %%
        B_gamma = assemble_B_gamma()
        M_gamma = boundary_discretization_field.assemble_mass_matrix( boundary_grid )

        # %%
        M_gamma

        # %%
        # B matrix
        div = - pg.cell_mass(mdg, head_discretization_field) @ pg.div(mdg)

        dof_p, dof_q = div.shape
        dof_eta = B_gamma.shape[0]

        div.shape, B_gamma.shape

        # %%
        def vertical_projection_matrix():

            data = []
            row = []
            col = []

            for c in range(subdomain.num_cells):
                x_center = subdomain.cell_centers[:, c]
                id = np.max(np.where( boundary_grid.nodes[0, :] < x_center[0] ))

                data.append(1)
                row.append(c)
                col.append(id)

            return sps.coo_matrix( (data, (row, col)), shape=(subdomain.num_cells, boundary_grid.num_cells) )

        # %%
        cell_proj_eta = vertical_projection_matrix()

        # Helper function to save the given solution to a VTU file
        def save_step(sol, proj_q, proj_psi, proj_eta, saver, i):
            ins = list()

            ins.append((sd, "cell_q", ( proj_q @ sol[:dof_q] ).reshape((3, -1), order="F")))
            ins.append((sd, "cell_h", proj_psi @ sol[dof_q:(dof_q+dof_p)]))
            ins.append((sd, "cell_eta", cell_proj_eta @proj_eta @ sol[-dof_eta:]))
            # print( cell_proj_eta @ proj_eta @ sol[-1][-dof_eta:] )

            saver.write_vtu(ins, time_step=i)

        # %%
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        # %%
        # Initial conditions
        sol = [np.zeros(dof_p + dof_q + dof_eta)]
        sol[-1][dof_q:(dof_q+dof_p)] = head_discretization_field.interpolate(sd, initial_h_func)
        sol[-1][-dof_eta:] = np.ones_like(sol[-1][-dof_eta:])

        # %%
        # Prepare helper matrices

        proj_q = velocity_discretization_field.eval_at_cell_centers(sd)
        proj_psi = head_discretization_field.eval_at_cell_centers(sd)
        proj_eta = boundary_discretization_field.eval_at_cell_centers(boundary_grid)

        eta_diff = boundary_discretization_field.assemble_diff_matrix(boundary_grid)

        # %%
        # Save the initial solution

        saver = pp.Exporter(mdg, 'sol', folder_name=output_directory)
        save_step(sol[-1], proj_q, proj_psi, proj_eta, saver, 0)

        # %%
        data[pp.PARAMETERS][key].update({"second_order_tensor": pp.SecondOrderTensor(np.ones(subdomain.num_cells))})

        # %%
        # Fixed rhs
        fixex_rhs = np.zeros(dof_p + dof_q + dof_eta)

        # %%
        # Time Loop

        try:
                

            for i in range(1, int(T/dt)+1):
                print('Time ' + str(i * dt))

                # Prepare the solution at the previous time step and ...
                prev = sol[-1].copy()

                # Prepare the rhs
                rhs = fixex_rhs.copy()
                rhs[:dof_q] += bc_val(i*dt)
                rhs[dof_q:(dof_q+dof_p)] += S_s / dt * head_discretization_field.assemble_mass_matrix(sd) @ prev[dof_q:(dof_q+dof_p)]
                rhs[-dof_eta:] += (phi / dt * M_gamma @ prev[-dof_eta:] + M_gamma @ boundary_discretization_field.interpolate(boundary_grid, infiltration))

                debug_saver = pp.Exporter(mdg, str(i) + '_sol', folder_name=os.path.join(output_directory, 'debug'))
                save_step(sol[-1], proj_q, proj_psi, proj_eta, debug_saver, 0)
                
                # Non-linear loop
                for k in range(max_iterations_per_step):
                    
                    # Prepare the conductivity
                    kxx = np.zeros(shape=(sd.num_cells,))
                    kyy = np.zeros(shape=(sd.num_cells,))
                    kxy = np.zeros(shape=(sd.num_cells,))

                    # Gradient of eta and pointwise value
                    grad_eta   = eta_diff @ prev[-dof_eta:]
                    center_eta = proj_eta @ prev[-dof_eta:]
                    
                    # "Dumb" K preparation
                    for c in np.arange(sd.num_cells):
                        x_center = sd.cell_centers[:, c]
                        
                        eta_cell = np.max(np.where( boundary_grid.nodes[0, :] < x_center[0] ))

                        kxx[c] = center_eta[eta_cell]
                        kxy[c] = -x_center[1] * grad_eta[eta_cell]
                        kyy[c] = (1 + np.power(x_center[1] * grad_eta[eta_cell], 2)) / center_eta[eta_cell]

                    _, data = mdg.subdomains(return_data=True)[0]

                    # Update conductivity and generate mass with conductivity
                    data[pp.PARAMETERS][key].update({"second_order_tensor": pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy)})
                    mass = pg.face_mass(mdg, velocity_discretization_field)

                    # Assemble the saddle point problem
                    spp = sps.bmat([[    mass,                                                         div.T,          B_gamma.T], 
                                    [    -div, S_s / dt * head_discretization_field.assemble_mass_matrix(sd),               None],
                                    [-B_gamma,                                                          None, phi / dt * M_gamma]], format="csc")
                    
                    # Prepare the solver
                    ls = pg.LinearSystem(spp, rhs)
                    ls.flag_ess_bc(np.hstack(bc_ess), np.zeros(dof_q + dof_p + dof_eta))

                    current = ls.solve()

                    #print(current[dof_q:])

                    # Compute the errors (with eta). Should I consider only psi? Should I compute the error on the "actual" psi values or on the dofs
                    abs_err_psi  = np.sqrt(np.sum(np.power(current[dof_q:] - prev[dof_q:], 2)))
                    abs_err_prev = np.sqrt(np.sum(np.power(prev[dof_q:], 2)))

                    print('Iteration #' + format(k+1, '0' + str(ceil(log10(max_iterations_per_step)) + 1) + 'd') 
                        + ', error L2 relative psi: ' + format(abs_err_psi, str(5 + ceil(log10(1 / abs_tol)) + 4) 
                                                                + '.' + str(ceil(log10(1 / abs_tol)) + 4) + 'f') )
                    
                    save_step(current, proj_q, proj_psi, proj_eta, debug_saver, k+1)

                    if abs_err_psi < abs_tol + rel_tol * abs_err_prev:
                        break
                    else:
                        prev = None
                        prev = current.copy()

                print('')        

                sol.append( current.copy() )

                save_step(sol[-1], proj_q, proj_psi, proj_eta, saver, i)

            saver.write_pvd([t * dt for t in range(int(T/dt)+1)])
        except Exception as ex:
            print('Crashed! Details: ')
            print(ex)