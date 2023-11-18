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

from richards.plot_exporter import Plot_Exporter

import porepy as pp
import pygeon as pg



class Solver:
    def __init__(self, model_data : Model_Data, solver_data: Solver_Data, verbose=True):
        self.model_data = model_data
        self.solver_data = solver_data
        self.verbose = verbose

        self.computer = Matrix_Computer(self.solver_data.mdg)

        self.subdomain = self.solver_data.mdg.subdomains(return_data=False)[0]

        self.primal = self.solver_data.primal


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
    


    def __export_plot(self, sol, filename):
        if self.primal:
            self.plotter.export_surface( self.subdomain.nodes[0, :], self.subdomain.nodes[1, :], sol, filename, self.solver_data.shape_x, self.solver_data.shape_y)
        else:
            print('Plots are available only for the primal formualtion!')


    def solve(self, max_iterations_per_step_override = None):
        if max_iterations_per_step_override is not None:
            backup_step = self.solver_data.max_iterations_per_step
            self.solver_data.max_iterations_per_step = max_iterations_per_step_override

        if self.solver_data.prepare_plots:
            self.plotter = Plot_Exporter(self.solver_data.output_directory)

        if self.solver_data.step_output_allowed:
            step_exporter = Step_Exporter(self.solver_data.mdg, "sol", self.solver_data.output_directory, primal=self.primal)
            sol = [self.solver_data.initial_solution]
            step_exporter.export( sol[-1] )

            if self.solver_data.prepare_plots:
                self.__export_plot(sol[-1], '0')
            
        else:
            sol = self.solver_data.initial_solution

            if self.solver_data.prepare_plots:
                self.__export_plot(sol, '0')

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
                sol.append( method(sol[-1], instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_exporter) )
                step_exporter.export(sol[-1])
                self.__export_solution_csv(sol[-1], str(step))

                if self.solver_data.prepare_plots:
                    self.__export_plot(sol[-1], str(step))
            else:
                sol = method(sol, instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_exporter)

                if self.solver_data.prepare_plots:
                    self.__export_plot(sol, str(step))

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
            step_exporter = Step_Exporter(self.solver_data.mdg, "sol", self.solver_data.output_directory, primal=self.primal)
            sol = [self.solver_data.initial_solution]
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
                tmp_sol = sol[-1]
            else:
                tmp_sol = sol

            instant = step * self.model_data.dt
            
            if self.verbose:
                print('Time ' + str(round(instant, 5)))
            
            id_solver = 0
            for scheme, iteration, abs_tol, rel_tol, exporter in zip(schemes, iterations, abs_tolerances, rel_tolerances, csv_exporters):
                print(Solver_Enum(scheme).name)

                self.solver_data.max_iterations_per_step = iteration
                tmp_sol = self.__get_method(scheme)(tmp_sol, instant, abs_tol, rel_tol, exporter, id_solver)
                id_solver = id_solver + 1

            self.solver_data.max_iterations_per_step = backup_step
            
            if self.verbose:
                print(Solver_Enum(self.solver_data.scheme).name)

            if self.solver_data.step_output_allowed:
                sol.append( self.__get_method(self.solver_data.scheme)(tmp_sol, instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_final_exporter, id_solver) )
                step_exporter.export(sol[-1])
            else:
                sol = self.__get_method(self.solver_data.scheme)(tmp_sol, instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_final_exporter, id_solver)

        if self.solver_data.step_output_allowed:
            step_exporter.export_final_pvd(np.array(range(0, self.model_data.num_steps + 1)) * self.model_data.dt)



    def __export_solution_csv(self, sol, filename):
        exporter = Csv_Exporter( os.path.join(self.solver_data.output_directory, 'csv'), filename + '.csv', ['x', 'y', 'h'], True)

        if self.solver_data.primal:
            for x, y, h in zip(self.subdomain.nodes[0, :], self.subdomain.nodes[1, :], sol):
                exporter.add_entry([x,y,h])
        else:
            for x, y, h in zip(self.subdomain.cell_centers[0, :], self.subdomain.cell_centers[1, :], sol[-self.computer.dof_P0:]):
                exporter.add_entry([x,y,h])




    def __prepare_time_rhs(self, t):
        if self.solver_data.primal:
            return self.__primal_prepare_time_rhs(t)
        else:
            return self.__dual_prepare_time_rhs(t)

    def __dual_prepare_time_rhs(self, t):
        
        fixed_rhs = np.zeros(self.computer.dof_P0 + self.computer.dof_RT0)

        if self.solver_data.rhs_func_psi is not None:
            fixed_rhs[-self.computer.dof_P0:] +=  self.computer.mass_matrix_P0() @  self.computer.P0.interpolate(self.subdomain, lambda x: self.solver_data.rhs_func_psi(x, t))

        if self.solver_data.rhs_func_q is not None:
            fixed_rhs[:self.computer.dof_RT0] += self.computer.mass_matrix_RT0() @ self.computer.RT0.interpolate(self.subdomain, lambda x: self.solver_data.rhs_func_q(x, t))

        if self.solver_data.rhs_vect_psi is not None:
            fixed_rhs[-self.computer.dof_P0:] += self.solver_data.rhs_vect_psi(t)

        if self.solver_data.rhs_vect_q is not None:
            fixed_rhs[:self.computer.dof_RT0] += self.solver_data.rhs_vect_q(t)
        
        return fixed_rhs
    
    def __primal_prepare_time_rhs(self, t):
        fixed_rhs = np.zeros(self.computer.dof_P1)

        if self.solver_data.rhs_func_psi is not None:
            fixed_rhs += self.computer.mass_matrix_P1() @ self.computer.P1.interpolate(self.subdomain, lambda x: self.solver_data.rhs_func_psi(x, t))

        if self.solver_data.rhs_vect_psi is not None:
            fixed_rhs += self.solver_data.rhs_vect_psi(t)

        return fixed_rhs





    def _generic_step_solver(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, method_prepare, method_step, id_solver):
        preparation = method_prepare(sol_n, t_n_1)

        prev = sol_n.copy()

        if self.solver_data.step_output_allowed:
            save_debug = Step_Exporter(self.solver_data.mdg, str(id_solver) + "_sol_" + str(t_n_1), self.solver_data.output_directory + "/debug", primal=self.solver_data.primal)
            save_debug.export( prev )
        
        for k in range(self.solver_data.max_iterations_per_step):
            current = None
            current = method_step(preparation, k, prev)

            if self.solver_data.step_output_allowed:
                save_debug.export( current )


            abs_err, abs_prev = self.__compute_errors(current, prev)

            if self.verbose:
                print('Iteration #' + format(k+1, '0' + str(ceil(log10(self.solver_data.max_iterations_per_step)) + 1) + 'd') 
                    + ', error L2 relative psi: ' + format(abs_err, str(5 + ceil(log10(1 / abs_tol)) + 4) + '.' + str(ceil(log10(1 / abs_tol)) + 4) + 'f'))


            if isnan(abs_err) or isnan(abs_prev):
                break

            exporter.add_entry([t_n_1, k+1, abs_err, abs_err / abs_prev])

            if abs_err < abs_tol + rel_tol * abs_prev:
                break
            else:
                prev = None
                prev = current.copy()

        print('')
        return current
    
    def __compute_errors(self, current, prev):
        if self.solver_data.primal:
            return np.sqrt((current - prev).T @ self.computer.mass_matrix_P1() @ (current - prev)), np.sqrt(prev.T @ self.computer.mass_matrix_P1() @ prev)
        else:
            cur = current[-self.computer.dof_P0:]
            h   = prev[-self.computer.dof_P0:]
            return np.sqrt((cur - h).T @ self.computer.mass_matrix_P0() @ (cur - h)), np.sqrt(h.T @ self.computer.mass_matrix_P0() @ h)





    def _L_scheme(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0):
        if self.solver_data.primal:
            return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._L_scheme_preparation, self._primal_L_scheme_method_step, id_solver)
        else:
            return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._L_scheme_preparation, self._dual_L_scheme_method_step, id_solver)
    
    def _L_scheme_preparation(self, sol_n, t_n_1):

        # Assemble the right-hand side
        fixed_rhs = self.__prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            adj_psi = sol_n - self.subdomain.nodes[1, :]
            fixed_rhs += self.computer.mass_matrix_P1() @ self.model_data.theta( adj_psi ) / self.model_data.dt
        else:
            adj_psi = self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ) - self.subdomain.cell_centers[1, :]
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( adj_psi ) ) / self.model_data.dt

        return { 'fixed_rhs': fixed_rhs.copy(), 't_n_1': t_n_1}

    def _dual_L_scheme_method_step(self, preparation, k, prev):
        dof_P0 = self.computer.dof_P0
        dt = self.model_data.dt

        h = prev[-dof_P0:]

        N = self.computer.mass_matrix_P0()
        rhs = preparation['fixed_rhs'].copy()
        

        # Theta^{n+1}_k
        adj_psi = self.computer.project_P0_to_solution( h ) - self.subdomain.cell_centers[1, :]
        rhs[-dof_P0:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( adj_psi )) / dt

        # Derivative Thetha^{n+1}_k
        rhs[-dof_P0:] += self.solver_data.L_Scheme_value * N @ h / dt


        # Construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient( adj_psi )))

        spp = sps.bmat([[                  M_k_n_1,               self.computer.matrix_B().T], 
                        [-self.computer.matrix_B(), self.solver_data.L_Scheme_value * N / dt]], format="csc")


        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()

    def _primal_L_scheme_method_step(self, preparation, k, prev):
        dt = self.model_data.dt

        M = self.computer.mass_matrix_P1()
        rhs = preparation['fixed_rhs'].copy()
        

        # Theta^{n+1}_k
        adj_psi = prev - self.subdomain.nodes[1, :]
        rhs -= self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( adj_psi )) / dt

        # Derivative Thetha^{n+1}_k
        rhs += self.solver_data.L_Scheme_value * M @ prev / dt

        # Construct the local matrices
        spp = self.solver_data.L_Scheme_value * M / dt + self.computer.stifness_matrix_P1_conductivity( adj_psi, self.model_data, self.solver_data.integration_order )

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))

        return ls.solve()





    def _newton(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0):
        if self.solver_data.primal:
            return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._newton_preparation, self._primal_newton_method_step, id_solver)
        else:
            return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._newton_preparation, self._dual_newton_method_step, id_solver)

    def _newton_preparation(self, sol_n, t_n_1):
        dof_psi = self.computer.dof_P0

        # assemble the right-hand side
        fixed_rhs = self.__prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            adj_psi = sol_n - self.subdomain.nodes[1, :]
            fixed_rhs += self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( adj_psi ) ) / self.model_data.dt
        else:
            adj_psi = self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ) - self.subdomain.cell_centers[1, :]
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( adj_psi ) ) / self.model_data.dt
            
        return {'t_n_1': t_n_1, 'fixed_rhs': fixed_rhs.copy()}
    
    def _dual_newton_method_step(self, preparation, k, prev):
        dof_psi = self.computer.dof_P0
        dof_q = self.computer.dof_RT0
        dt = self.model_data.dt

        h = prev[-dof_psi:]
        q   = prev[:dof_q]

        rhs = preparation['fixed_rhs'].copy()

        adj_psi = self.computer.project_P0_to_solution( h ) - self.subdomain.cell_centers[1, :]
        C = self.computer.dual_C(self.model_data, adj_psi, q)
        B = self.computer.matrix_B()

        rhs[:dof_q]    += C @ h


        # Theta^{n+1}_k
        rhs[-dof_psi:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( adj_psi )) / dt
            
        D = self.computer.mass_matrix_P0() @ sps.diags(self.model_data.theta(adj_psi, 1), format="csc")
        rhs[-dof_psi:] += D @ h / dt

        # construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(adj_psi)))
            
        spp = sps.bmat([[M_k_n_1, B.T + C], 
                        [     -B,  D / dt]], format="csc")
            
            
        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()
    
    def _primal_newton_method_step(self, preparation, k, prev):
        dt = self.model_data.dt
        rhs = preparation['fixed_rhs'].copy()

        adj_psi = prev - self.subdomain.nodes[1, :]
        C = self.computer.primal_C(self.model_data, adj_psi)
        D = self.computer.mass_matrix_P1_dtheta(self.model_data, adj_psi, self.solver_data.integration_order)

        # Theta^{n+1}_k
        rhs -= self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( adj_psi )) / dt
        rhs += C @ prev
        rhs += D @ prev / dt
            
        spp = D / dt + C + self.computer.stifness_matrix_P1_conductivity(adj_psi, self.model_data, self.solver_data.integration_order )
            
            
        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(np.hstack(self.solver_data.bc_essential(preparation['t_n_1'])), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()






    def _modified_picard(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0):
        if self.solver_data.primal:
            return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._modified_picard_preparation, self._primal_modified_picard_method_step, id_solver)
        else:
            return self._generic_step_solver(sol_n, t_n_1, abs_tol, rel_tol, exporter, self._modified_picard_preparation, self._dual_modified_picard_method_step, id_solver)

    def _modified_picard_preparation(self, sol_n, t_n_1):
        dof_psi = self.computer.dof_P0

        # Assemble the right-hand side
        fixed_rhs = self.__prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            adj_psi = sol_n - self.subdomain.nodes[1, :]

            fixed_rhs += self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( adj_psi ) ) / self.model_data.dt
        else:
            adj_psi = self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ) - self.subdomain.cell_centers[1, :]
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( adj_psi ) ) / self.model_data.dt
        
        return {'t_n_1': t_n_1, 'fixed_rhs': fixed_rhs.copy()}
    
    def _dual_modified_picard_method_step(self, preparation, k, prev):
        dof_psi = self.computer.dof_P0

        dt = self.model_data.dt

        h = prev[-dof_psi:]
        B = self.computer.matrix_B()

        adj_psi = self.computer.project_P0_to_solution( h ) - self.subdomain.cell_centers[1, :]
        N = self.computer.mass_matrix_P0() @ sps.diags(self.model_data.theta(adj_psi, 1), format="csc")

        rhs = preparation['fixed_rhs'].copy()

        # Theta^{n+1}_k
        rhs[-dof_psi:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta(adj_psi)) / dt

        # Derivative Thetha^{n+1}_k
        rhs[-dof_psi:] += N @ h / dt


        # Construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(adj_psi)))
            
        spp = sps.bmat([[M_k_n_1,    B.T], 
                        [     -B, N / dt]], format="csc")

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()
    
    def _primal_modified_picard_method_step(self, preparation, k, prev):
        dt = self.model_data.dt

        adj_psi = prev - self.subdomain.nodes[1, :]
        N = self.computer.mass_matrix_P1_dtheta(self.model_data, adj_psi, self.solver_data.integration_order)
        rhs = preparation['fixed_rhs'].copy()
        

        # Theta^{n+1}_k
        rhs -= self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( adj_psi )) / dt

        # Derivative Thetha^{n+1}_k
        rhs += N @ prev / dt


        # Construct the local matrices
        spp = N / dt + self.computer.stifness_matrix_P1_conductivity( adj_psi, self.model_data, self.solver_data.integration_order )

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()