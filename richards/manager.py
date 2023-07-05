from enum import Enum
import os
from math import ceil, floor, log10, exp
import numpy as np
import scipy.sparse as sps

from richards.step_exporter import Step_Exporter
from richards.csv_exporter import Csv_Exporter

from richards.model_params import Model_Data
from richards.matrix_computer import Matrix_Computer

import porepy as pp
import pygeon as pg



class Solver_Enum(Enum):
    PICARD = 1,
    NEWTON = 2,
    MODIFIED_PICARD=3,
    L_SCHEME = 4



class Solver_Data:
    def __init__(self, mdg, initial_solution, scheme: Solver_Enum, bc_essential, eps_psi_abs, eps_psi_rel, max_iterations_per_step, output_directory='output'):
        self.mdg = mdg
        self.initial_solution = initial_solution
        self.output_directory = output_directory
        self.scheme = scheme
        self.bc_essential = bc_essential

        self.eps_psi_abs = eps_psi_abs
        self.eps_psi_rel = eps_psi_rel
        self.max_iterations_per_step = max_iterations_per_step

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



class Solver:
    def __init__(self, model_data : Model_Data, solver_data: Solver_Data):
        self.model_data = model_data
        self.solver_data = solver_data

        self.computer = Matrix_Computer(self.solver_data.mdg)
        self.csv_exporter = Csv_Exporter('time_instant,iteration,absolute_error_norm,relative_error_norm')

        for sd in self.solver_data.mdg.subdomains(return_data=False):
            self.subdomain = sd

    def __get_method(self):
        if self.solver_data.scheme == Solver_Enum.MODIFIED_PICARD:
            return self._modified_picard
        elif self.solver_data.scheme == Solver_Enum.NEWTON:
            return self._newton
        elif self.solver_data.scheme == Solver_Enum.PICARD:
            return self._picard
        elif self.solver_data.scheme == Solver_Enum.L_SCHEME:
            return self._L_scheme
        else:
            return None



    def solve(self, step_override = None):
        if step_override is not None:
            backup_step = self.solver_data.max_iterations_per_step
            self.solver_data.max_iterations_per_step = step_override


        sol = [[self.solver_data.initial_solution]]
        step_exporter = Step_Exporter(self.solver_data.mdg, "sol", self.solver_data.output_directory)
        step_exporter.export( sol[-1] )

        method = self.__get_method()

        # Time Loop
        for step in range(1, self.model_data.num_steps + 1):
            sol.append( method(sol[-1][0], step * self.model_data.dt) )
            step_exporter.export(sol[-1])

        step_exporter.export_final_pvd(np.array(range(0, self.model_data.num_steps + 1)) * self.model_data.dt)
        self.csv_exporter.export_file(os.path.join(self.solver_data.output_directory, 'richards_variablySaturated_modifiedPicard.csv'))

        
        if step_override is not None:
            self.solver_data.max_iterations_per_step = backup_step



    def __prepare_time_rhs(self, t):
        
        fixed_rhs = np.zeros(self.computer.dof_psi[0] + self.computer.dof_q[0])

        if self.solver_data.rhs_func_psi is not None:
            fixed_rhs[-self.computer.dof_psi[0]:] += self.computer.mass_matrix_P0()[0] @ self.computer.P0.interpolate(self.subdomain, lambda x: self.solver_data.rhs_func_psi(x, t))

        if self.solver_data.rhs_func_q is not None:
            fixed_rhs[:self.computer.dof_q[0]] += self.computer.mass_matrix_RT0()[0] @ self.computer.RT0.interpolate(self.subdomain, lambda x: self.solver_data.rhs_func_q(x, t))

        if self.solver_data.rhs_vect_psi is not None:
            fixed_rhs[-self.computer.dof_psi[0]:] += self.solver_data.rhs_vect_psi(t)

        if self.solver_data.rhs_vect_q is not None:
            fixed_rhs[:self.computer.dof_q[0]] += self.solver_data.rhs_vect_q(t)
        
        return fixed_rhs



    def _L_scheme(self, sol_n, t_n_1):
        pass



    def _picard(self, sol_n, t_n_1):
        pass



    def _newton(self, sol_n, t_n_1):
        proj_psi = self.computer.compute_proj_psi_mat()[0]
        dof_psi = self.computer.dof_psi[0]
        dof_q = self.computer.dof_q[0]
        dt = self.model_data.dt

        prev = sol_n.copy()

        Mass_psi = self.computer.mass_matrix_P0()[0]
        B = self.computer.matrix_B()


        # assemble the right-hand side
        fixed_rhs = self.__prepare_time_rhs(t_n_1)

        # Theta^n
        fixed_rhs[-dof_psi:] += Mass_psi @ self.computer.project_psi_to_fe([self.model_data.theta(proj_psi @ sol_n[-dof_psi:])])[0]

        save_debug = Step_Exporter(self.solver_data.mdg, "sol_" + str(t_n_1), self.solver_data.output_directory + "/debug")
        save_debug.export( [prev] )

        print('Time ' + str(round(t_n_1, 5)))

        for k in range(self.solver_data.max_iterations_per_step):
            psi = prev[-dof_psi:]
            q   = prev[:dof_q]

            rhs = None
            rhs = fixed_rhs.copy()

            C = self.computer.C(self.model_data, proj_psi @ psi, q)[0]
            B = self.computer.matrix_B()

            rhs[:dof_q]    += C @ psi


            # Theta^{n+1}_k
            rhs[-dof_psi:] -= Mass_psi @ self.computer.project_psi_to_fe([self.model_data.theta(proj_psi @ psi)])[0]
            
            D = Mass_psi @ np.diag(self.model_data.theta(proj_psi @ psi, 1))
            rhs[-dof_psi:] += D @ psi

            # construct the local matrices
            M_k_n_1 = self.computer.mass_matrix_RT0_conductivity([pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(proj_psi @ psi))])
            
            spp = sps.bmat([[M_k_n_1, (B.T + C)], 
                            [-dt * B,         D]], format="csc")
            

            
            # solve the problem
            ls = None
            ls = pg.LinearSystem(spp, rhs)
            ls.flag_ess_bc(np.hstack(self.solver_data.bc_essential(t_n_1)), np.zeros(dof_q + dof_psi))
        
            current = None
            current = ls.solve()

            save_debug.export( [current] )


            abs_err_psi  = np.sqrt(np.sum(np.power(current[-dof_psi:]-prev[-dof_psi:], 2)))
            abs_err_prev = np.sqrt(np.sum(np.power(prev[-dof_psi:], 2)))


            print('Iteration #' + format(k+1, '0' + str(ceil(log10(self.solver_data.max_iterations_per_step)) + 1) + 'd') 
                + ', error L2 relative psi: ' + format(abs_err_psi, str(5 + ceil(log10(1 / self.solver_data.eps_psi_abs)) + 4) + '.' + str(ceil(log10(1 / self.solver_data.eps_psi_abs)) + 4) + 'f') )
            
            self.csv_exporter.add_entry(str(t_n_1) + ',' + str(k+1) + ',' + str(abs_err_psi) + ',' + str(abs_err_psi / abs_err_prev))



            if abs_err_psi < self.solver_data.eps_psi_abs + self.solver_data.eps_psi_rel * abs_err_prev:
                break
            else:
                prev = current
                
        print('')
        return [current]



    def _modified_picard(self, sol_n, t_n_1):
        proj_psi = self.computer.compute_proj_psi_mat()[0]
        dof_psi = self.computer.dof_psi[0]
        dof_q = self.computer.dof_q[0]
        dt = self.model_data.dt

        prev = sol_n.copy()

        Mass_psi = self.computer.mass_matrix_P0()[0]
        B = self.computer.matrix_B()

        # Assemble the right-hand side
        fixed_rhs = self.__prepare_time_rhs(t_n_1)

        # Theta^n
        fixed_rhs[-dof_psi:] += Mass_psi @ self.computer.project_psi_to_fe([self.model_data.theta(proj_psi @ sol_n[-dof_psi:])])[0] / dt


        save_debug = Step_Exporter(self.solver_data.mdg, "sol_" + str(t_n_1), self.solver_data.output_directory + "/debug")
        save_debug.export( [prev] )

        print('Time ' + str(round(t_n_1, 5)))

        for k in range(self.solver_data.max_iterations_per_step):
            psi = prev[-dof_psi:]
            q   = prev[:dof_q]
            


            N = None
            N = Mass_psi @ np.diag(self.model_data.theta(proj_psi @ psi, 1))

            rhs = None
            rhs = fixed_rhs.copy()

            # Theta^{n+1}_k
            rhs[-dof_psi:] -= Mass_psi @ self.computer.project_psi_to_fe([self.model_data.theta(proj_psi @ psi)])[0] / dt

            # Derivative Thetha^{n+1}_k
            rhs[-dof_psi:] += N @ psi / dt


            # Construct the local matrices
            M_k_n_1 = self.computer.mass_matrix_RT0_conductivity([pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(proj_psi @ psi))])
            
            spp = sps.bmat([[M_k_n_1,    B.T], 
                            [-B,      N / dt]], format="csc")

            

            # Solve the problem
            ls = None
            ls = pg.LinearSystem(spp, rhs)
            ls.flag_ess_bc(np.hstack(self.solver_data.bc_essential(t_n_1)), np.zeros(dof_q + dof_psi))
        
            current = None
            current = ls.solve()


            save_debug.export( [current])


            abs_err_psi  = np.sqrt(np.sum(np.power(current[-dof_psi:] - psi, 2)))
            abs_err_prev = np.sqrt(np.sum(np.power(psi, 2)))



            print('Iteration #' + format(k+1, '0' + str(ceil(log10(self.solver_data.max_iterations_per_step)) + 1) + 'd') 
                + ', error L2 relative psi: ' + format(abs_err_psi, str(5 + ceil(log10(1 / self.solver_data.eps_psi_abs)) + 4) + '.' + str(ceil(log10(1 / self.solver_data.eps_psi_abs)) + 4) + 'f') )
            


            self.csv_exporter.add_entry(str(t_n_1) + ',' + str(k+1) + ',' + str(abs_err_psi) + ',' + str(abs_err_psi / abs_err_prev))

            if abs_err_psi < self.solver_data.eps_psi_abs + self.solver_data.eps_psi_rel * abs_err_prev:
                break
            else:
                prev = None
                prev = current.copy()
                
        print('')
        return [current]