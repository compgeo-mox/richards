from math import ceil, floor, log10, exp, isnan

import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

from richards.solver_params import Solver_Data, Norm_Error
from richards.model_params import Model_Data
from richards.step_exporter import Step_Exporter
from richards.matrix_computer import Matrix_Computer



class Nonlinear_Solver:
    def __init__(self, model_data : Model_Data, solver_data: Solver_Data, verbose=True):
        self.solver_data = solver_data
        self.model_data = model_data
        self.verbose = verbose
        
        self.computer = Matrix_Computer(self.solver_data.mdg)
        self.subdomain = self.solver_data.mdg.subdomains(return_data=False)[0]

    def solve(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0, prev=None):
        pass

    def _generic_step_solver(self, sol_n, prev, t_n_1, abs_tol, rel_tol, exporter, method_prepare, method_step, id_solver):
        preparation = method_prepare(sol_n, t_n_1)

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
                    + ', relative norm of the error: ' + format(abs_err / abs_prev, str(5 + ceil(log10(1 / (abs_tol + rel_tol))) + 4) + '.' + str(ceil(log10(1 / (abs_tol + rel_tol))) + 4) + 'f')
                    + ', norm of the error: ' + format(abs_err, str(5 + ceil(log10(1 / (abs_tol + rel_tol))) + 4) + '.' + str(ceil(log10(1 / (abs_tol + rel_tol))) + 4) + 'f'))


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
    
    def _prepare_time_rhs(self, t):
        if self.solver_data.primal:
            return self.__primal_prepare_time_rhs(t)
        else:
            return self.__dual_prepare_time_rhs(t)

    

    def __compute_errors(self, current, prev):
        if self.solver_data.primal:
            var = current - prev
                        
            if self.solver_data.norm_error == Norm_Error.L2:
                return np.sqrt(var.T @ self.computer.mass_matrix_P1() @ var), np.sqrt(prev.T @ self.computer.mass_matrix_P1() @ prev)
            else:
                return np.sqrt(var.T @ var), np.sqrt(prev.T @ prev)
        else:
            prev_h = prev[-self.computer.dof_P0:]
            var = current[-self.computer.dof_P0:] - prev_h
            
            if self.solver_data.norm_error == Norm_Error.L2:
                return np.sqrt(var.T @ self.computer.mass_matrix_P0() @ var), np.sqrt(prev_h.T @ self.computer.mass_matrix_P0() @ prev_h)
            else:
                prev_h = self.computer.project_P0_to_solution(prev_h)
                var = self.computer.project_P0_to_solution(var)
                return np.sqrt(var.T @ var), np.sqrt(prev_h.T @ prev_h)

    def __dual_prepare_time_rhs(self, t):
        
        fixed_rhs = np.zeros(self.computer.dof_P0 + self.computer.dof_RT0)

        if self.solver_data.rhs_func_psi is not None:
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @  self.computer.P0.interpolate(self.subdomain, lambda x: self.solver_data.rhs_func_psi(x, t))

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




class L_scheme_Nonlinear_Solver(Nonlinear_Solver):
    def __init__(self, model_data : Model_Data, solver_data: Solver_Data, verbose=True):
        super().__init__(model_data, solver_data, verbose)

    def solve(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0, prev=None):
        if prev is None:
            prev = sol_n

        if self.solver_data.primal:
            return self._generic_step_solver(sol_n, prev, t_n_1, abs_tol, rel_tol, exporter, self._L_scheme_preparation, self._primal_L_scheme_method_step, id_solver)
        else:
            return self._generic_step_solver(sol_n, prev, t_n_1, abs_tol, rel_tol, exporter, self._L_scheme_preparation, self._dual_L_scheme_method_step, id_solver)
    
    def _L_scheme_preparation(self, sol_n, t_n_1):

        # Assemble the right-hand side
        fixed_rhs = self._prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            fixed_rhs += self.computer.mass_matrix_P1() @ self.model_data.theta( sol_n, self.subdomain.nodes[1, :] ) / self.model_data.dt
        else:
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ), self.subdomain.cell_centers[1, :] ) ) / self.model_data.dt

        return { 'fixed_rhs': fixed_rhs.copy(), 't_n_1': t_n_1}
    
    def _L_scheme_preparation_psi(self, sol_n, t_n_1):

        # Assemble the right-hand side
        fixed_rhs = self._prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            fixed_rhs += self.computer.mass_matrix_P1() @ self.model_data.theta( sol_n, np.zeros_like(self.subdomain.nodes[1, :]) ) / self.model_data.dt
        else:
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ), np.zeros_like(self.subdomain.cell_centers[1, :]) ) ) / self.model_data.dt

        return { 'fixed_rhs': fixed_rhs.copy(), 't_n_1': t_n_1}

    def _dual_L_scheme_method_step(self, preparation, k, prev):
        dof_P0 = self.computer.dof_P0
        dt = self.model_data.dt

        h = prev[-dof_P0:]

        N = self.computer.mass_matrix_P0()
        rhs = preparation['fixed_rhs'].copy()

        # Theta^{n+1}_k
        rhs[-dof_P0:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( h ), self.subdomain.cell_centers[1, :] )) / dt

        # Derivative Thetha^{n+1}_k
        rhs[-dof_P0:] += self.solver_data.L_Scheme_value * N @ h / dt


        # Construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient( self.computer.project_P0_to_solution( h ), self.subdomain.cell_centers[1, :] )))

        spp = sps.bmat([[                  M_k_n_1,               self.computer.matrix_B().T], 
                        [-self.computer.matrix_B(), self.solver_data.L_Scheme_value * N / dt]], format="csc")


        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()
    
    def _dual_L_scheme_method_step_psi(self, preparation, k, prev):
        dof_P0 = self.computer.dof_P0
        dt = self.model_data.dt

        psi = prev[-dof_P0:]

        N = self.computer.mass_matrix_P0()
        
        rhs = preparation['fixed_rhs'].copy()
        
        # Theta^{n+1}_k
        rhs[-dof_P0:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( psi ), np.zeros_like(self.subdomain.cell_centers[1, :]) )) / dt

        # Derivative Thetha^{n+1}_k
        rhs[-dof_P0:] += self.solver_data.L_Scheme_value * N @ psi / dt

        # Construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient( self.computer.project_P0_to_solution( psi ), np.zeros_like(self.subdomain.cell_centers[1, :]) )))

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
        rhs -= self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( prev, self.subdomain.nodes[1, :] )) / dt

        # Derivative Thetha^{n+1}_k
        rhs += self.solver_data.L_Scheme_value * M @ prev / dt

        # Construct the local matrices
        spp = self.solver_data.L_Scheme_value * M / dt + self.computer.stifness_matrix_P1_conductivity( prev, self.model_data, self.solver_data.integration_order )

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))

        return ls.solve()
    


class Newton_Nonlinear_Solver(Nonlinear_Solver):
    def __init__(self, model_data : Model_Data, solver_data: Solver_Data, verbose=True):
        super().__init__(model_data, solver_data, verbose)

    def solve(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0, prev=None):
        if prev is None:
            prev = sol_n

        if self.solver_data.primal:
            return self._generic_step_solver(sol_n, prev, t_n_1, abs_tol, rel_tol, exporter, self._newton_preparation, self._primal_newton_method_step, id_solver)
        else:
            return self._generic_step_solver(sol_n, prev, t_n_1, abs_tol, rel_tol, exporter, self._newton_preparation, self._dual_newton_method_step, id_solver)

    def _newton_preparation(self, sol_n, t_n_1):
        # assemble the right-hand side
        fixed_rhs = self._prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            fixed_rhs += self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( sol_n, self.subdomain.nodes[1, :] ) ) / self.model_data.dt
        else:
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ), self.subdomain.cell_centers[1, :] ) ) / self.model_data.dt
            
        return {'t_n_1': t_n_1, 'fixed_rhs': fixed_rhs.copy()}
    
    def _newton_preparation_psi(self, sol_n, t_n_1):
        # assemble the right-hand side
        fixed_rhs = self._prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            fixed_rhs += self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( sol_n, np.zeros_like(self.subdomain.nodes[1, :]) ) ) / self.model_data.dt
        else:
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ), np.zeros_like(self.subdomain.cell_centers[1, :]) ) ) / self.model_data.dt
            

        return {'t_n_1': t_n_1, 'fixed_rhs': fixed_rhs.copy()}
    
    def _dual_newton_method_step(self, preparation, k, prev):
        dof_h = self.computer.dof_P0
        dof_q = self.computer.dof_RT0
        dt = self.model_data.dt

        h = prev[-dof_h:]
        q   = prev[:dof_q]

        rhs = preparation['fixed_rhs'].copy()

        C = self.computer.dual_C(self.model_data, self.computer.project_P0_to_solution( h ), q)
        B = self.computer.matrix_B()

        rhs[:dof_q] += C @ h


        # Theta^{n+1}_k
        rhs[-dof_h:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( h ), self.subdomain.cell_centers[1, :] )) / dt
            
        D = self.computer.mass_matrix_P0() @ sps.diags(self.model_data.theta(self.computer.project_P0_to_solution( h ), self.subdomain.cell_centers[1, :], 1), format="csc")
        rhs[-dof_h:] += D @ h / dt

        # construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(self.computer.project_P0_to_solution( h ), self.subdomain.cell_centers[1, :])))

        spp = sps.bmat([[M_k_n_1, B.T + C], 
                        [     -B,  D / dt]], format="csc")
            
            
        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()
    
    def _dual_newton_method_step_psi(self, preparation, k, prev):
        dof_h = self.computer.dof_P0
        dof_q = self.computer.dof_RT0
        dt = self.model_data.dt

        h = prev[-dof_h:]
        q   = prev[:dof_q]

        rhs = preparation['fixed_rhs'].copy()

        C = self.computer.dual_C(self.model_data, self.computer.project_P0_to_solution( h ), q)
        B = self.computer.matrix_B()

        rhs[:dof_q] += C @ h


        # Theta^{n+1}_k
        rhs[-dof_h:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( h ), np.zeros_like(self.subdomain.cell_centers[1, :]) )) / dt
            
        D = self.computer.mass_matrix_P0() @ sps.diags(self.model_data.theta(self.computer.project_P0_to_solution( h ), np.zeros_like(self.subdomain.cell_centers[1, :]), 1), format="csc")
        rhs[-dof_h:] += D @ h / dt

        # construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(self.computer.project_P0_to_solution( h ), np.zeros_like(self.subdomain.cell_centers[1, :]))))

        spp = sps.bmat([[M_k_n_1, B.T + C], 
                        [     -B,  D / dt]], format="csc")
        
        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()
    
    def _primal_newton_method_step(self, preparation, k, prev):
        dt = self.model_data.dt
        rhs = preparation['fixed_rhs'].copy()

        C = self.computer.primal_C(self.model_data, prev)
        D = self.computer.mass_matrix_P1_dtheta(self.model_data, prev, self.solver_data.integration_order)

        # Theta^{n+1}_k
        rhs -= self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( prev, self.subdomain.nodes[1, :] )) / dt
        rhs += C @ prev
        rhs += D @ prev / dt
            
        spp = D / dt + C + self.computer.stifness_matrix_P1_conductivity(prev, self.model_data, self.solver_data.integration_order )
            
            
        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(np.hstack(self.solver_data.bc_essential(preparation['t_n_1'])), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()



class Modified_Picard_Nonlinear_Solver(Nonlinear_Solver):
    def __init__(self, model_data : Model_Data, solver_data: Solver_Data, verbose=True):
        super().__init__(model_data, solver_data, verbose)
        
    def solve(self, sol_n, t_n_1, abs_tol, rel_tol, exporter, id_solver=0, prev=None):
        if prev is None:
            prev = sol_n

        if self.solver_data.primal:
            return self._generic_step_solver(sol_n, prev, t_n_1, abs_tol, rel_tol, exporter, self._modified_picard_preparation, self._primal_modified_picard_method_step, id_solver)
        else:
            return self._generic_step_solver(sol_n, prev, t_n_1, abs_tol, rel_tol, exporter, self._modified_picard_preparation, self._dual_modified_picard_method_step, id_solver)

    def _modified_picard_preparation(self, sol_n, t_n_1):
        dof_psi = self.computer.dof_P0

        # Assemble the right-hand side
        fixed_rhs = self._prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            fixed_rhs += self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( sol_n, self.subdomain.nodes[1, :] ) ) / self.model_data.dt
        else:
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ), self.subdomain.cell_centers[1, :] ) ) / self.model_data.dt
        
        return {'t_n_1': t_n_1, 'fixed_rhs': fixed_rhs.copy()}
    
    def _modified_picard_preparation_psi(self, sol_n, t_n_1):
        dof_psi = self.computer.dof_P0

        # Assemble the right-hand side
        fixed_rhs = self._prepare_time_rhs(t_n_1)

        # Theta^n
        if self.solver_data.primal:
            fixed_rhs += self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( sol_n, np.zeros_like(self.subdomain.nodes[1, :]) ) ) / self.model_data.dt
        else:
            fixed_rhs[-self.computer.dof_P0:] += self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta( self.computer.project_P0_to_solution( sol_n[-self.computer.dof_P0:] ), np.zeros_like(self.subdomain.cell_centers[1, :]) ) ) / self.model_data.dt
        
        return {'t_n_1': t_n_1, 'fixed_rhs': fixed_rhs.copy()}
    
    def _dual_modified_picard_method_step(self, preparation, k, prev):
        dof_psi = self.computer.dof_P0

        dt = self.model_data.dt

        h = prev[-dof_psi:]
        B = self.computer.matrix_B()

        N = self.computer.mass_matrix_P0() @ sps.diags(self.model_data.theta(self.computer.project_P0_to_solution( h ), self.subdomain.cell_centers[1, :], 1), format="csc")

        rhs = preparation['fixed_rhs'].copy()

        # Theta^{n+1}_k
        rhs[-dof_psi:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta(self.computer.project_P0_to_solution( h ), self.subdomain.cell_centers[1, :])) / dt

        # Derivative Thetha^{n+1}_k
        rhs[-dof_psi:] += N @ h / dt


        # Construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(self.computer.project_P0_to_solution( h ), self.subdomain.cell_centers[1, :])))
            
        spp = sps.bmat([[M_k_n_1,    B.T], 
                        [     -B, N / dt]], format="csc")

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()
    
    def _dual_modified_picard_method_step_psi(self, preparation, k, prev):
        dof_psi = self.computer.dof_P0

        dt = self.model_data.dt

        h = prev[-dof_psi:]
        B = self.computer.matrix_B()

        N = self.computer.mass_matrix_P0() @ sps.diags(self.model_data.theta(self.computer.project_P0_to_solution( h ), np.zeros_like(self.subdomain.cell_centers[1, :]), 1), format="csc")

        rhs = preparation['fixed_rhs'].copy()

        # Theta^{n+1}_k
        rhs[-dof_psi:] -= self.computer.mass_matrix_P0() @ self.computer.project_function_to_P0(self.model_data.theta(self.computer.project_P0_to_solution( h ), np.zeros_like(self.subdomain.cell_centers[1, :]))) / dt

        # Derivative Thetha^{n+1}_k
        rhs[-dof_psi:] += N @ h / dt


        # Construct the local matrices
        M_k_n_1 = self.computer.mass_matrix_RT0_conductivity(pp.SecondOrderTensor(self.model_data.hydraulic_conductivity_coefficient(self.computer.project_P0_to_solution( h ), np.zeros_like(self.subdomain.cell_centers[1, :]))))
            
        spp = sps.bmat([[M_k_n_1,    B.T], 
                        [     -B, N / dt]], format="csc")

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()
    
    def _primal_modified_picard_method_step(self, preparation, k, prev):

        dt = self.model_data.dt

        N = self.computer.mass_matrix_P1_dtheta(self.model_data, prev, self.solver_data.integration_order)
        rhs = preparation['fixed_rhs'].copy()

        # Theta^{n+1}_k
        rhs -= self.computer.mass_matrix_P1() @ self.computer.project_function_to_P1(self.model_data.theta( prev, self.subdomain.nodes[1, :] )) / dt

        # Derivative Thetha^{n+1}_k
        rhs += N @ prev / dt


        # Construct the local matrices
        spp = N / dt + self.computer.stifness_matrix_P1_conductivity( prev, self.model_data, self.solver_data.integration_order )

        # Solve the problem
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(self.solver_data.bc_essential(preparation['t_n_1']), self.solver_data.bc_essential_value(preparation['t_n_1']))
        
        return ls.solve()
    