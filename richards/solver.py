from enum import Enum

import os
from math import ceil, floor, log10, exp, isnan
import numpy as np
import scipy.sparse as sps

from richards.solver_params import Solver_Data, Solver_Enum, Norm_Error
from richards.step_exporter import Step_Exporter
from richards.csv_exporter import Csv_Exporter

from richards.model_params import Model_Data
from richards.matrix_computer import Matrix_Computer

from richards.plot_exporter import Plot_Exporter

from richards.nonlinear_solvers.nonlinear_solver import Nonlinear_Solver, L_scheme_Nonlinear_Solver, Newton_Nonlinear_Solver, Modified_Picard_Nonlinear_Solver

import porepy as pp
import pygeon as pg



class Solver:
    def __init__(self, model_data : Model_Data, solver_data: Solver_Data, verbose=True):
        self.model_data = model_data
        self.solver_data = solver_data
        self.verbose = verbose

        # self.computer = Matrix_Computer(self.solver_data.mdg)

        self.subdomain = self.solver_data.mdg.subdomains(return_data=False)[0]

        self.primal = self.solver_data.primal


    def __get_nonlinear_solver(self, scheme):
        if scheme == Solver_Enum.PICARD:
            return Modified_Picard_Nonlinear_Solver(self.model_data, self.solver_data, self.verbose)
        elif scheme == Solver_Enum.NEWTON:
            return Newton_Nonlinear_Solver(self.model_data, self.solver_data, self.verbose)
        elif scheme == Solver_Enum.LSCHEME:
            if self.solver_data.L_Scheme_value is None:
                raise Exception('Solver: Missing L parameter to employ with the L scheme!')

            return L_scheme_Nonlinear_Solver(self.model_data, self.solver_data, self.verbose)
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
            self.__export_solution_csv(sol[-1], '0')

            if self.solver_data.prepare_plots:
                self.__export_plot(sol[-1], '0')
            
        else:
            sol = self.solver_data.initial_solution

            if self.solver_data.prepare_plots:
                self.__export_plot(sol, '0')

        csv_exporter = Csv_Exporter(self.solver_data.report_directory, 
                                    self.__exporter_name(Solver_Enum(self.solver_data.scheme).name + '_richards_solver.csv'), 
                                    ['time_instant', 'iteration', 'absolute_error_norm' , 'relative_error_norm'])

        nonlinear_solver = self.__get_nonlinear_solver(self.solver_data.scheme)

        # Time Loop
        for step in range(1, self.model_data.num_steps + 1):
            instant = step * self.model_data.dt
            
            if self.verbose:
                print('Time ' + str(round(instant, 5)))
        
            if self.solver_data.step_output_allowed:
                sol.append( nonlinear_solver.solve(sol[-1], instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_exporter) )
                step_exporter.export(sol[-1])
                self.__export_solution_csv(sol[-1], str(step))

                if self.solver_data.prepare_plots:
                    self.__export_plot(sol[-1], str(step))
            else:
                sol = nonlinear_solver.solve(sol, instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_exporter)

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

        if self.solver_data.prepare_plots:
            self.plotter = Plot_Exporter(self.solver_data.output_directory)

        if self.solver_data.step_output_allowed:
            step_exporter = Step_Exporter(self.solver_data.mdg, "sol", self.solver_data.output_directory, primal=self.primal)
            sol = [self.solver_data.initial_solution]
            step_exporter.export( sol[-1] )
            self.__export_solution_csv(sol[-1], '0')

            if self.solver_data.prepare_plots:
                self.__export_plot(sol[-1], '0')
        else:
            sol = self.solver_data.initial_solution

            if self.solver_data.prepare_plots:
                self.__export_plot(sol[-1], '0')

        csv_exporters = list()
        
        name_schemes = []
        nonlinear_solvers = []
        final_nonlinear_solver = self.__get_nonlinear_solver(self.solver_data.scheme)

        for scheme in schemes:
            name_schemes.append( scheme.name )
            nonlinear_solvers.append(self.__get_nonlinear_solver(scheme))

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
            for scheme, nonlinear_solver, iteration, abs_tol, rel_tol, exporter in zip(schemes, nonlinear_solvers, iterations, 
                                                                                       abs_tolerances, rel_tolerances, csv_exporters):
                print(Solver_Enum(scheme).name)

                self.solver_data.max_iterations_per_step = iteration
                tmp_sol = nonlinear_solver.solve(sol[-1], instant, abs_tol, rel_tol, exporter, id_solver, prev=tmp_sol)
                id_solver = id_solver + 1

            self.solver_data.max_iterations_per_step = backup_step
            
            if self.verbose:
                print(Solver_Enum(self.solver_data.scheme).name)

            if self.solver_data.step_output_allowed:
                sol.append( final_nonlinear_solver.solve(sol[-1], instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_final_exporter, id_solver, prev=tmp_sol) )
                step_exporter.export(sol[-1])
                self.__export_solution_csv(sol[-1], str(step))

                if self.solver_data.prepare_plots:
                    self.__export_plot(sol[-1], str(step))
            else:
                sol = final_nonlinear_solver.solve(sol[-1], instant, self.solver_data.eps_psi_abs, self.solver_data.eps_psi_rel, csv_final_exporter, id_solver, prev=tmp_sol)
                

                if self.solver_data.prepare_plots:
                    self.__export_plot(sol, str(step))

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