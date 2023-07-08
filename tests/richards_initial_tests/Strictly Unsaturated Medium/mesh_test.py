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
L_value = 0.136

K = 150

eps_psi_abs = 1e-7
eps_psi_rel = 0

problem_name = 'unsaturated'

output_directory = 'output_evolutionary'

model_data = Model_Data(theta_r=0.026, theta_s=0.42, alpha=0.551, n=2.9, K_s=0.12, T=1, num_steps=1)

def g_func(x,t): 
    return np.array([0, -1, -1])

def initial_pressure_func(x, tolerance):
    if x[1] > 1/4 + tolerance:
        return -4
    else:
        return -x[1]-1/4
    
def initial_velocity_func(x, tolerance):
    if x[1] > 1/4 + tolerance:
        return - model_data.hydraulic_conductivity_coefficient(np.array([-4])) * np.array([0,1,1])
    else:
        return np.zeros(shape=3)

def f(x,t, tolerance):
    res = 0
    if x[1] > 1/4 + tolerance:
        res = 0.06*np.cos(4/3*np.pi*x[1])*np.sin(x[0])

    return res

def bc_gamma_d(x, t): 
    return -4


def run_experiment(N, prefix_file_name):
    ### DOMAIN CONSTRUCTION
    domain = pp.StructuredTriangleGrid([N, N], [1,1])
    mdg = pp.meshing.subdomains_to_mdg([domain])
    domain_tolerance = 1 / (100 * N)

    key = "flow"

    bc_value = []
    bc_essential = []
    initial_pressure = []
    initial_velocity = []

    RT0 = pg.RT0(key)
    P0  = pg.PwConstants(key)

    for subdomain, data in mdg.subdomains(return_data=True):
        initial_pressure.append( P0.interpolate(subdomain, lambda x: initial_pressure_func(x, domain_tolerance)))
        initial_velocity.append(RT0.interpolate(subdomain, lambda x: initial_velocity_func(x, domain_tolerance)))
            
        # with the following steps we identify the portions of the boundary
        # to impose the boundary conditions
        boundary_faces_indexes = subdomain.get_boundary_faces()

        gamma_d  = subdomain.face_centers[1, :] > 1-domain_tolerance
        gamma_n  = gamma_d.copy()
        
        gamma_n[boundary_faces_indexes] = np.logical_not(gamma_n[boundary_faces_indexes])
        

        pp.initialize_data(subdomain, data, key, {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(subdomain.num_cells)),
        })
        

        bc_value.append(lambda t: - RT0.assemble_nat_bc(subdomain, lambda x: bc_gamma_d(x,t), gamma_d))

        essential_pressure_dofs = np.zeros(P0.ndof(subdomain), dtype=bool) # No essential BC for pressure
        bc_essential.append(np.hstack((gamma_n, essential_pressure_dofs))) # Essential BC for velocity on gamma_n

    ### PREPARE SOLVER DATA
    cp = Matrix_Computer(mdg)
    initial_solution = np.zeros(cp.dof_q[0] + cp.dof_psi[0])
    initial_solution[-cp.dof_psi[0]:] += np.hstack(initial_pressure)
    initial_solution[:cp.dof_q[0]] += np.hstack(initial_velocity)

    solver_data = Solver_Data(mdg=mdg, initial_solution=initial_solution, scheme=scheme, 
                            bc_essential=lambda t: bc_essential, eps_psi_abs=eps_psi_abs,
                            eps_psi_rel=eps_psi_rel, max_iterations_per_step=K,
                            output_directory=output_directory, L_Scheme_value=L_value,
                            report_name=prefix_file_name)

    solver_data.set_rhs_vector_q(lambda t: np.hstack(list(cond(t) for cond in bc_value)))
    solver_data.set_rhs_function_q(g_func)
    solver_data.set_rhs_function_psi(lambda x,t: f(x,t, domain_tolerance))

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