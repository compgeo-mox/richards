from enum import Enum

class Solver_Enum(Enum):
    PICARD = 1,
    NEWTON = 2,
    MODIFIED_PICARD=3,
    L_SCHEME = 4



class Solver_Data:
    def __init__(self, mdg, initial_solution, scheme: Solver_Enum, bc_essential, eps_psi_abs, eps_psi_rel, max_iterations_per_step, L_Scheme_value=None, output_directory='output'):
        self.mdg = mdg
        self.initial_solution = initial_solution
        self.output_directory = output_directory
        self.scheme = scheme
        self.bc_essential = bc_essential

        self.eps_psi_abs = eps_psi_abs
        self.eps_psi_rel = eps_psi_rel
        self.max_iterations_per_step = max_iterations_per_step
        self.L_Scheme_value = L_Scheme_value

        self.rhs_func_q = None
        self.rhs_func_psi = None
        self.rhs_vect_q = None
        self.rhs_vect_psi = None
    
    def set_rhs_function_q(self, rhs_q):
        self.rhs_func_q = rhs_q
    
    def set_rhs_function_psi(self, rhs_psi):
        self.rhs_func_psi = rhs_psi
    
    def set_rhs_vector_q(self, rhs_q):
        self.rhs_vect_q = rhs_q
    
    def set_rhs_vector_psi(self, rhs_psi):
        self.rhs_vect_psi = rhs_psi