### IMPORTS

from richards.model_params import Model_Data
from richards.solver import Solver
from richards.csv_exporter import Csv_Exporter
from richards.matrix_computer import Matrix_Computer
from richards.solver_params import Solver_Data, Solver_Enum, Norm_Error


import sys, time, os, shutil
sys.path.insert(0, "/workspaces/richards/")


import porepy as pp
import pygeon as pg

import numpy as np


### PARAMETERS

K = 500

eps_psi_abs = 1e-5
eps_psi_rel = 1e-5

dt_D = 1/16
problem_name = 'benchmark_L_test'

output_directory = 'dual_L_test_output_evolutionary'
report_directory = 'dual_report_L_test'

def initial_pressure_func(x):
    return 1

    
def bc_gamma_d(x, t, tolerance):
    if   x[0] > 2-tolerance and x[1] > 0-tolerance and x[1] < 1+tolerance:
        res =  1
    elif x[1] > 3-tolerance and x[0] > 0-tolerance and x[0] < 1+tolerance:
        res = min( 3.2, 1 + 2.2 * t / dt_D )
    else:
        res = 0

    return res


def run_experiment(N, prefix_file_name, L_Value, report_output_directory, scheme_info, int_model_data):
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

        bc_value = lambda t: - RT0.assemble_nat_bc(subdomain, lambda x: bc_gamma_d(x,t, domain_tolerance), gamma_d)

        essential_pressure_dofs = np.zeros(P0.ndof(subdomain), dtype=bool)
        bc_essential = np.hstack((gamma_n, essential_pressure_dofs))

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    ### PREPARE SOLVER DATA
    cp = Matrix_Computer(mdg)
    initial_solution = np.zeros(cp.dof_RT0 + cp.dof_P0)
    initial_solution[-cp.dof_P0:] += np.hstack(initial_pressure)

    solver_data = Solver_Data(mdg=mdg, initial_solution=initial_solution, scheme=scheme_info, 
                            bc_essential=lambda t: bc_essential, eps_psi_abs=eps_psi_abs,
                            eps_psi_rel=eps_psi_rel, max_iterations_per_step=K,
                            output_directory=output_directory, L_Scheme_value=L_Value,
                            report_name=prefix_file_name, report_directory=report_output_directory,
                            step_output_allowed=False, norm_error=Norm_Error.EUCLIDIAN)

    solver_data.set_rhs_vector_q(lambda t: bc_value(t))

    ### PREPARE SOLVER
    start = time.time()
    solver = Solver(model_data=int_model_data, solver_data=solver_data)
    solver.solve()
    end = time.time()

    return end - start


def run_experiments(L_values, directory_prefixes, Ns, int_model_data):
    exporters = []
    report_output_directories = []

    for directory_prefix in directory_prefixes:
        if directory_prefix is None:
            report_output_directories.append(os.path.join(report_directory, problem_name + '_LSCHEME'))
        else:
            report_output_directories.append(os.path.join(report_directory, directory_prefix + '_' + problem_name + '_LSCHEME'))
            
        exporters.append( Csv_Exporter(report_output_directories[-1], problem_name + '_LSCHEME_richards_solver.csv', ['N', 'time'], overwrite_existing=False) )


    for N in Ns:
        for L_value, exporter, report_output_directory in zip(L_values, exporters, report_output_directories):
            print('Running experiment with N=' + str(N) + ' with scheme LSCHEME and L=' + str(L_value * 0.1e-2))
            exporter.add_entry([N, run_experiment(N, "{:03d}".format(N) + '_' + problem_name, L_value * 0.1e-2, report_output_directory, Solver_Enum.LSCHEME, int_model_data)])


def variable_L():
    steps = 9
    print('Problem name: ' + problem_name + ', Variable_L, num_steps=' + str(steps))
    L_values = [i for i in range(23, 60)]
    prefixes = []

    for L_value in L_values:
        prefixes.append('VARL_' + str(L_value) + '_steps_' + str(steps))

    model_data = Model_Data(theta_r=0.131, theta_s=0.396, alpha=0.423, n=2.06, K_s=4.96e-2, T=9/48, num_steps=steps)
    run_experiments(L_values, prefixes, [40], model_data)

def variable_N():
    steps = 9
    print('Problem name: ' + problem_name + ', Variable_N, num_steps=' + str(steps))
    L_values = [34]
    prefixes = []

    for L_value in L_values:
        prefixes.append('VARN_' + str(L_value) + '_steps_' + str(steps))


    model_data = Model_Data(theta_r=0.131, theta_s=0.396, alpha=0.423, n=2.06, K_s=4.96e-2, T=9/48, num_steps=steps)
    run_experiments(L_values, prefixes, [10, 20, 40, 60, 80, 100, 120, 140, 160], model_data)

def variable_dt():
    steps = [i for i in range(9, 100, 10)]
    L_values = [34]

    for step in steps:
        prefixes = []
        for pref in L_values:
            prefixes.append('VART_' + str(pref) + '_steps_' + str(step))
        
        model_data = Model_Data(theta_r=0.131, theta_s=0.396, alpha=0.423, n=2.06, K_s=4.96e-2, T=9/48, num_steps=step)
        run_experiments(L_values, prefixes, [40], model_data)



variable_L()
variable_N()
variable_dt()