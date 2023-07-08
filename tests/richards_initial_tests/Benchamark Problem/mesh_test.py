### IMPORTS

import sys
sys.path.insert(0, "/workspaces/richards/")

from richards.model_params import Model_Data
from richards.matrix_computer import Matrix_Computer

from richards.solver import Solver
from richards.solver_params import Solver_Data, Solver_Enum

from richards.csv_exporter import Csv_Exporter


import porepy as pp
import pygeon as pg

import numpy as np
import time


### PARAMETERS
scheme = Solver_Enum.LSCHEME
L_value = 4.501e-3

K = 150

eps_psi_abs = 1e-6
eps_psi_rel = 1e-6

dt_D = 1/16
problem_name = 'benchmark'

output_directory = 'output_evolutionary'

model_data = Model_Data(theta_r=0.131, theta_s=0.396, alpha=0.423, n=2.06, K_s=4.96e-2, T=9/48, num_steps=9)

def g_func(x,t): 
    return np.array([0, -1, -1])

def initial_pressure_func(x):
    return 1-x[1]

    
def bc_gamma_d(x, t, tolerance):
    if   x[0] > 2-tolerance and x[1] > 0-tolerance and x[1] < 1+tolerance:
        res =  1 - x[1]
    elif x[1] > 3-tolerance and x[0] > 0-tolerance and x[0] < 1+tolerance:
        res = min( 0.2, -2 + 2.2 * t / dt_D )
    else:
        res = 0

    return res


def run_experiment(N, prefix_file_name):
    ### DOMAIN CONSTRUCTION
    domain = pp.StructuredTriangleGrid([N, N], [1,1])
    mdg = pp.meshing.subdomains_to_mdg([domain])
    domain_tolerance = 1 / (100 * N)

    key = "flow"

    bc_value = []
    bc_essential = []
    initial_pressure = []

    RT0 = pg.RT0(key)
    P0  = pg.PwConstants(key)

    for subdomain, data in mdg.subdomains(return_data=True):
        initial_pressure.append(P0.interpolate(subdomain, initial_pressure_func))
            
        # with the following steps we identify the portions of the boundary
        # to impose the boundary conditions
        boundary_faces_indexes = subdomain.get_boundary_faces()

        gamma_d1 = np.logical_and(subdomain.face_centers[0, :] > 0-domain_tolerance, np.logical_and(subdomain.face_centers[0, :] < 1+domain_tolerance, subdomain.face_centers[1, :] > 3-domain_tolerance))
        gamma_d2 = np.logical_and(subdomain.face_centers[0, :] > 2-domain_tolerance, np.logical_and(subdomain.face_centers[1, :] > 0-domain_tolerance, subdomain.face_centers[1, :] < 1+domain_tolerance))

        gamma_d  = np.logical_or(gamma_d1, gamma_d2)
        
        gamma_n  = gamma_d.copy()
        gamma_n[boundary_faces_indexes] = np.logical_not(gamma_n[boundary_faces_indexes])
        

        pp.initialize_data(subdomain, data, key, {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(subdomain.num_cells)),
        })

        bc_value.append(lambda t: - RT0.assemble_nat_bc(subdomain, lambda x: bc_gamma_d(x,t, domain_tolerance), gamma_d))

        essential_pressure_dofs = np.zeros(P0.ndof(subdomain), dtype=bool)
        bc_essential.append(np.hstack((gamma_n, essential_pressure_dofs)))

    ### PREPARE SOLVER DATA
    cp = Matrix_Computer(mdg)
    initial_solution = np.zeros(cp.dof_q[0] + cp.dof_psi[0])
    initial_solution[-cp.dof_psi[0]:] += np.hstack(initial_pressure)

    solver_data = Solver_Data(mdg=mdg, initial_solution=initial_solution, scheme=scheme, 
                            bc_essential=lambda t: bc_essential, eps_psi_abs=eps_psi_abs,
                            eps_psi_rel=eps_psi_rel, max_iterations_per_step=K,
                            output_directory=output_directory, L_Scheme_value=L_value,
                            report_name=prefix_file_name)

    solver_data.set_rhs_vector_q(lambda t: np.hstack(list(cond(t) for cond in bc_value)))
    solver_data.set_rhs_function_q(g_func)

    ### PREPARE SOLVER
    start = time.time()
    solver = Solver(model_data=model_data, solver_data=solver_data, verbose=False)
    solver.solve()
    end = time.time()

    return end - start


print('Problem name: ' + problem_name)

exporter = Csv_Exporter('N,time')

for N in range(10, 81, 10):
    print('Running experiment with N=' + str(N))
    exporter.add_entry(N, run_experiment(N, str(N) + '_' + problem_name))

exporter.export_file('report', problem_name + '_' + scheme.name + '_richards_solver.csv')