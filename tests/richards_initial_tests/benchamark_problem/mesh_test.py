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


### PARAMETERS

K = 250

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


def run_experiment(N, prefix_file_name, L_Value, report_output_directory, scheme_info):
    ### DOMAIN CONSTRUCTION
    domain = pp.StructuredTriangleGrid([2*N, 3*N], [2,3])
    mdg = pp.meshing.subdomains_to_mdg([domain])
    domain_tolerance = 1 / (100 * N)

    key = "flow"

    bc_value = []
    bc_essential = []
    initial_pressure = []

    RT0 = pg.RT0(key)
    P0  = pg.PwConstants(key)

    subdomain, data = mdg.subdomains(return_data=True)[0]
    initial_pressure.append(P0.interpolate(subdomain, initial_pressure_func))
            
    # with the following steps we identify the portions of the boundary to impose the boundary conditions
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

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    ### PREPARE SOLVER DATA
    cp = Matrix_Computer(mdg)
    initial_solution = np.zeros(cp.dof_q[0] + cp.dof_psi[0])
    initial_solution[-cp.dof_psi[0]:] += np.hstack(initial_pressure)

    solver_data = Solver_Data(mdg=mdg, initial_solution=initial_solution, scheme=scheme_info, 
                            bc_essential=lambda t: bc_essential, eps_psi_abs=eps_psi_abs,
                            eps_psi_rel=eps_psi_rel, max_iterations_per_step=K,
                            output_directory=output_directory, L_Scheme_value=L_Value,
                            report_name=prefix_file_name, report_directory=report_output_directory,
                            step_output_allowed=False)

    solver_data.set_rhs_vector_q(lambda t: np.hstack(list(cond(t) for cond in bc_value)))
    solver_data.set_rhs_function_q(g_func)

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

schemes = [Solver_Enum.LSCHEME, Solver_Enum.NEWTON, Solver_Enum.PICARD, Solver_Enum.LSCHEME]
L_values = [3.501e-2, None, None, 4.501e-2]
prefixes = ['1', None, None, '2']


run_experiments(schemes, L_values, prefixes)