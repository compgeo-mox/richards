import numpy as np
from enum import Enum

class Solver_Enum(Enum):
    NEWTON  = 1,
    PICARD  = 2,
    LSCHEME = 3

class Norm_Error(Enum):
    EUCLIDIAN = 1,
    L2 = 1

# Simple class used to store the solver related parameters and to generate the rhs of the problem
class Solver_Data:
    def __init__(self, mdg, initial_solution, scheme: Solver_Enum, bc_essential, eps_psi_abs, eps_psi_rel, max_iterations_per_step, 
                 L_Scheme_value=None, output_directory='output', report_directory='report', report_name=None, step_output_allowed=True, primal=False, 
                 bc_essential_value = None, integration_order = 0, prepare_plots = False, shape_x=None, shape_y=None,
                 norm_error=Norm_Error.L2):
        self.mdg = mdg
        self.initial_solution = initial_solution
        self.output_directory = output_directory
        self.scheme = scheme
        self.bc_essential = bc_essential

        self.norm_error = norm_error

        if bc_essential_value == None:
            self.bc_essential_value = lambda t: np.zeros_like(bc_essential(t))
        else:
            self.bc_essential_value = bc_essential_value

        self.eps_psi_abs = eps_psi_abs
        self.eps_psi_rel = eps_psi_rel
        self.max_iterations_per_step = max_iterations_per_step
        self.L_Scheme_value = L_Scheme_value
        self.report_directory = report_directory
        self.report_name = report_name
        self.step_output_allowed = step_output_allowed

        self.primal = primal

        self.integration_order = integration_order
        self.prepare_plots = prepare_plots

        if self.prepare_plots and (shape_x == None or shape_y == None):
            print('The plots will not be performed without a provided shape')
        self.shape_x = shape_x
        self.shape_y = shape_y

        self.rhs_func_q = None
        self.rhs_func_psi = None
        self.rhs_vect_q = None
        self.rhs_vect_psi = None
    
    def set_rhs_function_q(self, rhs_q):
        if self.primal:
            print('You are currently setting a rhs for q with primal formulation...')

        self.rhs_func_q = rhs_q
    
    def set_rhs_function_psi(self, rhs_psi):
        if self.primal:
            print('You are currently setting a rhs for psi with primal formulation...')

        self.rhs_func_psi = rhs_psi
    
    def set_rhs_function_h(self, rhs_h):
        self.rhs_func_psi = rhs_h


    
    def set_rhs_vector_q(self, rhs_q):
        if self.primal:
            print('You are currently setting a rhs for q with primal formulation...')

        self.rhs_vect_q = rhs_q
    
    def set_rhs_vector_psi(self, rhs_psi):
        if self.primal:
            print('You are currently setting a rhs for psi with primal formulation...')

        self.rhs_vect_psi = rhs_psi
    
    def set_rhs_vector_h(self, rhs_h):
        self.rhs_vect_psi = rhs_h