from enum import Enum

import os
from math import ceil, floor, log10, exp, isnan
import numpy as np
import scipy.sparse as sps

from richards.solver_params import Solver_Data, Solver_Enum
from richards.step_exporter import Step_Exporter
from richards.csv_exporter import Csv_Exporter

from richards.model_params import Model_Data
from richards.matrix_computer import Matrix_Computer

import porepy as pp
import pygeon as pg



class Solver:
    def __init__(self, model_data : Model_Data, solver_data: Solver_Data, verbose=True):
        self.model_data = model_data
        self.solver_data = solver_data
        self.verbose = verbose

        self.computer = Matrix_Computer(self.solver_data.mdg)

        for sd in self.solver_data.mdg.subdomains(return_data=False):
            self.subdomain = sd


    def __get_method(self, scheme):
        if scheme == Solver_Enum.PICARD:
            return self._modified_picard
        elif scheme == Solver_Enum.NEWTON:
            return self._newton
        elif scheme == Solver_Enum.LSCHEME:
            if self.solver_data.L_Scheme_value is None:
                raise Exception('Solver: Missing L parameter to employ with the L scheme!')

            return self._L_scheme
        else:
            return None

    def __exporter_name(self, suffix):
        if self.solver_data.report_name is not None:
            return self.solver_data.report_name + '_' + suffix
        return suffix


    def solve(self, max_iterations_per_step_override = None):

        if max_iterations_per_step_override is not None:
            backup_step = self.solver_data.max_iterations_per_step
            self.solver_data.max_iterations_per_step = max_iterations_per_step_override


        if self.solver_data.step_output_allowed:
            step_exporter = Step_Exporter(self.solver_data.mdg, "sol", self.solver_data.output_directory)
            sol = [[self.solver_data.initial_solution]]
            step_exporter.export( sol[-1] )
        else:
            sol = self.solver_data.initial_solution

        csv_exporter = Csv_Exporter(self.solver_data.report_directory, 
                                    self.__exporter_name(Solver_Enum(self.solver_data.scheme).name + '_richards_solver.csv'), 
                                    ['time_instant', 'iteration', 'absolute_error_norm' , 'relative_error_norm'])


        method = self.__get_method(self.solver_data.scheme)

        # Time Loop
        for step in range(1, self.model_data.num_steps + 1):
            instant = step * self.model_data.dt
            
            if self.verbose:
                print('Time ' + str(round(instant, 5)))
        
            if self.solver_data.step_output_allowed:
                sol.append( method(sol[-1][0], instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_exporter) )
                step_exporter.export(sol[-1])
            else:
                sol = method(sol, instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_exporter)[0]

        if self.solver_data.step_output_allowed:    
            step_exporter.export_final_pvd(np.array(range(0, self.model_data.num_steps + 1)) * self.model_data.dt)

        
        if max_iterations_per_step_override is not None:
            self.solver_data.max_iterations_per_step = backup_step


    def multistage_solver(self, schemes: list, iterations: list, abs_tolerances: list, rel_tolerances: list):
        if schemes is None or iterations is None or len(schemes) <= 0 or len(iterations) <= 0 or len(schemes) != len(iterations):
            print('Solver: multistage_solver will not work since either schemes or iterations are not defined (or empty) or they have different lengths. I\'ll default to solve')
            return self.solve()
        
        backup_step = self.solver_data.max_iterations_per_step

        if self.solver_data.step_output_allowed:
            step_exporter = Step_Exporter(self.solver_data.mdg, "sol", self.solver_data.output_directory)
            sol = [[self.solver_data.initial_solution]]
            step_exporter.export( sol[-1] )
        else:
            sol = self.solver_data.initial_solution

        csv_exporters = list()
        
        name_schemes = []
        for scheme in schemes:
            name_schemes.append( scheme.name )
        name_schemes.append(self.solver_data.scheme.name)

        base_path = os.path.join(self.solver_data.report_directory, self.__exporter_name('_'.join(name_schemes)))

        entries = ['time_instant', 'iteration', 'absolute_error_norm' , 'relative_error_norm']
        for i in range(len(iterations)):
            csv_exporters.append( Csv_Exporter(base_path, self.__exporter_name(str(i) + '_' + name_schemes[i] + '_richards_solver.csv'), entries) )

        csv_final_exporter = Csv_Exporter(base_path, self.__exporter_name(str(len(iterations)) + '_' + self.solver_data.scheme.name + '_richards_solver.csv'), entries)


        # Time Loop
        for step in range(1, self.model_data.num_steps + 1):
            if self.solver_data.step_output_allowed:
                tmp_sol = sol[-1][0]
            else:
                tmp_sol = sol

            instant = step * self.model_data.dt
            
            if self.verbose:
                print('Time ' + str(round(instant, 5)))
            
            id_solver = 0
            for scheme, iteration, abs_tol, rel_tol, exporter in zip(schemes, iterations, abs_tolerances, rel_tolerances, csv_exporters):
                print(Solver_Enum(scheme).name)

                self.solver_data.max_iterations_per_step = iteration
                tmp_sol = self.__get_method(scheme)(tmp_sol, instant, abs_tol, rel_tol, exporter, id_solver)[0]
                id_solver = id_solver + 1

            self.solver_data.max_iterations_per_step = backup_step
            
            if self.verbose:
                print(Solver_Enum(self.solver_data.scheme).name)

            if self.solver_data.step_output_allowed:
                sol.append( self.__get_method(self.solver_data.scheme)(tmp_sol, instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_final_exporter, id_solver) )
                step_exporter.export(sol[-1])
            else:
                sol = self.__get_method(self.solver_data.scheme)(tmp_sol, instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_final_exporter, id_solver)[0]

        if self.solver_data.step_output_allowed:
            step_exporter.export_final_pvd(np.array(range(0, self.model_data.num_steps + 1)) * self.model_data.dt)


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





    def _generic_step_solver(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, method_prepare, method_step, id_solver):
        dof_psi = self.computer.dof_psi[0]

        preparation = method_prepare(sol_n, t_n_1)

        prev = sol_n.copy()

        mass_psi = self.computer.mass_matrix_P0()

        if self.solver_data.step_output_allowed:
            save_debug = Step_Exporter(self.solver_data.mdg, str(id_solver) + "_sol_" + str(t_n_1), self.solver_data.output_directory + "/debug")
            save_debug.export( [prev] )
        
        for k in range(self.solver_data.max_iterations_per_step):
            psi = prev[-dof_psi:]

            current = None
            current = method_step(preparation, k, prev)

            if self.solver_data.step_output_allowed:
                save_debug.export( [current] )


            abs_err_psi  = np.sqrt((current[-dof_psi:] - psi).T @ mass_psi @ (current[-dof_psi:] - psi))
            abs_err_prev = np.sqrt(psi.T @ mass_psi @ psi)


            if self.verbose:
                print('Iteration #' + format(k+1, '0' + str(ceil(log10(self.solver_data.max_iterations_per_step)) + 1) + 'd') 
                    + ', error L2 relative psi: ' + str(abs_err_psi))

            if isnan(abs_err_psi) or isnan(abs_err_prev):
                break

            exporter.add_entry([t_n_1, k+1, abs_err_psi, abs_err_psi / abs_err_prev])


            if abs_err_psi < abs_tol + rel_tol * abs_err_prev:
                break
            else:
                prev = None
                prev = current.copy()


        print('')
        return [current]





    def _L_scheme(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0):
        return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._L_scheme_preparation, self._L_scheme_method_step, id_solver)
    
    def _L_scheme_preparation(self, sol_n, t_n_1):
        proj_psi = self.computer.compute_proj_psi_mat()[0]
        dof_psi = self.computer.dof_psi[0]
        dt = self.model_data.dt

        # Assemble the right-hand side
        fixed_rhs = self.__prepare_time_rhs(t_n_1)

        # Theta^n
        fixed_rhs[-dof_psi:] += self.computer.mass_matrix_P0()[0] @ self.computer.project_psi_to_fe([self.model_data.theta(proj_psi @ sol_n[-dof_psi:])])[0] / dt

        return { 'mass_psi': self.computer.mass_matrix_P0()[0], 'B': self.computer.matrix_B(), 'fixed_rhs': fixed_rhs.copy(), 
                'dof_psi': dof_psi, 'dof_q': self.computer.dof_q[0], 'proj_psi': proj_psi, 'dt': dt, 't_n_1': t_n_1}

    def _L_scheme_method_step(self, preparation, k, prev):
        dof_psi = preparation['dof_psi']

        dt = preparation['dt']
        psi = prev[-dof_psi:]

        N = None
        N = preparation['mass_psi'] @ sps.diags(np.ones_like(psi) * self.solver_data.L_Scheme_value, format="csc")
            
        rhs = None
        rhs = preparation['fixed_rhs'].copy()

        # Theta^{n+1}_k
        rhs[-dof_psi:] -= preparation['mass_psi'] @ self.computer.project_psi_to_fe([self.model_data.theta(preparation['proj_psi'] @ psi)])[0] / dt

        # Derivative Thetha^{n+1}_k
        rhs[-dof_psi:] += N @ psi / dt


        # Construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity([pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(preparation['proj_psi'] @ psi))])
            
        spp = sps.bmat([[M_k_n_1,    preparation['B'].T], 
                        [-preparation['B'],      N / dt]], format="csc")

            

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(np.hstack(self.solver_data.bc_essential(preparation['t_n_1'])), np.zeros(preparation['dof_q'] + dof_psi))
        
        return ls.solve()





    def _newton(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0):
        return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._newton_preparation, self._newton_method_step, id_solver)

    def _newton_preparation(self, sol_n, t_n_1):
        proj_psi = self.computer.compute_proj_psi_mat()[0]
        dof_psi = self.computer.dof_psi[0]

        # assemble the right-hand side
        fixed_rhs = self.__prepare_time_rhs(t_n_1)

        # Theta^n
        fixed_rhs[-dof_psi:] += self.computer.mass_matrix_P0()[0] @ self.computer.project_psi_to_fe([self.model_data.theta(proj_psi @ sol_n[-dof_psi:])])[0]

        return {'proj_psi': proj_psi, 'dof_psi': dof_psi, 'dof_q': self.computer.dof_q[0], 'dt': self.model_data.dt, 't_n_1': t_n_1,
                'mass_psi': self.computer.mass_matrix_P0()[0], 'B': self.computer.matrix_B(), 'fixed_rhs': fixed_rhs.copy()}
    
    def _newton_method_step(self, preparation, k, prev):
        dof_psi = preparation['dof_psi']
        dof_q = preparation['dof_q']

        psi = prev[-dof_psi:]
        q   = prev[:dof_q]

        rhs = preparation['fixed_rhs'].copy()

        C = self.computer.C(self.model_data, preparation['proj_psi'] @ psi, q)[0]
        B = self.computer.matrix_B()

        rhs[:dof_q]    += C @ psi


        # Theta^{n+1}_k
        rhs[-dof_psi:] -= preparation['mass_psi'] @ self.computer.project_psi_to_fe([self.model_data.theta(preparation['proj_psi'] @ psi)])[0]
            
        D = preparation['mass_psi'] @ sps.diags(self.model_data.theta(preparation['proj_psi'] @ psi, 1), format="csc")
        rhs[-dof_psi:] += D @ psi

        # construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity([pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(preparation['proj_psi'] @ psi))])
            
        spp = sps.bmat([[M_k_n_1,                (B.T + C)], 
                        [-preparation['dt'] * B,         D]], format="csc")
            
            
        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(np.hstack(self.solver_data.bc_essential(preparation['t_n_1'])), np.zeros(dof_q + dof_psi))
        
        return ls.solve()






    def _modified_picard(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0):
        return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._modified_picard_preparation, self._modified_picard_method_step, id_solver)

    def _modified_picard_preparation(self, sol_n, t_n_1):
        proj_psi = self.computer.compute_proj_psi_mat()[0]
        dof_psi = self.computer.dof_psi[0]
        dt = self.model_data.dt

        Mass_psi = self.computer.mass_matrix_P0()[0]

        # Assemble the right-hand side
        fixed_rhs = self.__prepare_time_rhs(t_n_1)

        # Theta^n
        fixed_rhs[-dof_psi:] += Mass_psi @ self.computer.project_psi_to_fe([self.model_data.theta(proj_psi @ sol_n[-dof_psi:])])[0] / dt

        return {'proj_psi': proj_psi, 'dof_psi': dof_psi, 'dof_q': self.computer.dof_q[0], 'dt': dt, 't_n_1': t_n_1,
                'mass_psi': self.computer.mass_matrix_P0()[0], 'B': self.computer.matrix_B(), 'fixed_rhs': fixed_rhs.copy()}
    
    def _modified_picard_method_step(self, preparation, k, prev):
        dof_psi = preparation['dof_psi']
        dof_q = preparation['dof_q']
        dt = preparation['dt']

        psi = prev[-dof_psi:]

        N = preparation['mass_psi'] @ sps.diags(self.model_data.theta(preparation['proj_psi'] @ psi, 1), format="csc")

        rhs = preparation['fixed_rhs'].copy()

        # Theta^{n+1}_k
        rhs[-dof_psi:] -= preparation['mass_psi'] @ self.computer.project_psi_to_fe([self.model_data.theta(preparation['proj_psi'] @ psi)])[0] / dt

        # Derivative Thetha^{n+1}_k
        rhs[-dof_psi:] += N @ psi / dt


        # Construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity([pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(preparation['proj_psi'] @ psi))])
            
        spp = sps.bmat([[M_k_n_1,    preparation['B'].T], 
                        [-preparation['B'],      N / dt]], format="csc")

            

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(np.hstack(self.solver_data.bc_essential(preparation['t_n_1'])), np.zeros(dof_q + dof_psi))
        
        return ls.solve()