### IMPORTS

import sys, time, os, shutil
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
K = 200

eps_psi_abs = 1e-5
eps_psi_rel = 1e-5

problem_name = 'saturated'

output_directory = 'output_evolutionary'

model_data = Model_Data(theta_r=0.026, theta_s=0.42, alpha=0.95, n=2.9, K_s=0.12, T=0.01, num_steps=1)

def g_func(x,t): 
    return np.array([0, -1, -1])

def initial_pressure_func(x, tolerance):
    if x[1] > 1/4 + tolerance:
        return -3
    else:
        return -x[1]+1/4
    
def initial_velocity_func(x, tolerance):
    if x[1] > 1/4 + tolerance:
        return - model_data.hydraulic_conductivity_coefficient(np.array([-3])) * np.array([0,1,1])
    else:
        return np.zeros(shape=3)

def f(x,t, tolerance):
    res = 0
    if x[1] > 1/4 + tolerance:
        res = 0.006*np.cos(4/3*np.pi*(x[1]-1))*np.sin(2*np.pi*x[0])

    return res

def bc_gamma_d(x, t): 
    return -3


def run_experiment(N, prefix_file_name, L_Value, report_output_directory, scheme_info):
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

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    ### PREPARE SOLVER DATA
    cp = Matrix_Computer(mdg)
    initial_solution = np.zeros(cp.dof_q[0] + cp.dof_psi[0])
    initial_solution[-cp.dof_psi[0]:] += np.hstack(initial_pressure)
    initial_solution[:cp.dof_q[0]] += np.hstack(initial_velocity)

    solver_data = Solver_Data(mdg=mdg, initial_solution=initial_solution, scheme=scheme_info, 
                            bc_essential=lambda t: bc_essential, eps_psi_abs=eps_psi_abs,
                            eps_psi_rel=eps_psi_rel, max_iterations_per_step=K,
                            output_directory=output_directory, L_Scheme_value=L_Value,
                            report_name=prefix_file_name, report_directory=report_output_directory,
                            step_output_allowed=False)

    solver_data.set_rhs_vector_q(lambda t: np.hstack(list(cond(t) for cond in bc_value)))
    solver_data.set_rhs_function_q(g_func)
    solver_data.set_rhs_function_psi(lambda x,t: f(x,t, domain_tolerance))

    ### PREPARE SOLVER
    start = time.time()
    solver = Solver(model_data=model_data, solver_data=solver_data)
    solver.solve()
    end = time.time()

    return end - start


def run_experiments(schemes, L_values, directory_prefixes):
    exporters = []
    report_output_directories = []

    for scheme, directory_prefix in zip(schemes, directory_prefixes):
        if directory_prefix is None:
            report_output_directories.append('report/' + problem_name + '_' + scheme.name)
        else:
            report_output_directories.append('report/' + problem_name + '_' + directory_prefix  + '_' + scheme.name)
            
        exporters.append( Csv_Exporter(report_output_directories[-1], problem_name + '_' + scheme.name  + '_richards_solver.csv', ['N', 'time'], overwrite_existing=False) )


    for N in range(10, 81, 10):
        for scheme, L_value, exporter, report_output_directory in zip(schemes, L_values, exporters, report_output_directories):
            print('Running experiment with N=' + str(N) + ' with scheme ' + scheme.name)
            exporter.add_entry([N, run_experiment(N, str(N) + '_' + problem_name, L_value, report_output_directory, scheme)])




print('Problem name: ' + problem_name)

schemes = [Solver_Enum.NEWTON, Solver_Enum.LSCHEME, Solver_Enum.PICARD, Solver_Enum.LSCHEME]
L_values = [None, 0.15, None, 0.2341]
prefixes = [None, '1', None, '2']


run_experiments(schemes, L_values, prefixes)