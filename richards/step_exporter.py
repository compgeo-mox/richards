from richards.matrix_computer import Matrix_Computer

import porepy as pp
import pygeon as pg



class Step_Exporter:

    def __init__(self, mdg, sol_name, output_directory, primal=False):

        self.num_step = 0

        self.save = pp.Exporter(mdg, sol_name, folder_name=output_directory)
        self.computer = Matrix_Computer(mdg)

        self.primal = primal

        self.subdomain = mdg.subdomains(return_data=False)[0]


    
    def export(self, solution):
        if self.primal:
            self._export_primal(solution)
        else:
            self._export_dual(solution)

    def export_final_pvd(self, times):
        self.save.write_pvd(times)


    
    def _export_dual(self, solution):
        q_dofs = solution[:self.computer.dof_RT0]
        h_dofs = solution[-self.computer.dof_P0:]
        
        ins = list()

        ins.append((self.subdomain, "cell_q", ( self.computer.project_RT0_to_solution(q_dofs) ).reshape((3, -1), order="F")))
        ins.append((self.subdomain, "cell_h", self.computer.project_P0_to_solution(h_dofs) ))
        ins.append((self.subdomain, "cell_p", self.computer.project_P0_to_solution(h_dofs) - self.subdomain.cell_centers[1, :] ))
    
        self.save.write_vtu(ins, time_step=self.num_step)
        
        self.num_step = self.num_step + 1

    
    def _export_primal(self, solution):
        ins = list()

        ins.append((self.subdomain, "cell_h", self.computer.project_P1_to_solution(solution) ))
        ins.append((self.subdomain, "cell_p", self.computer.project_P1_to_solution(solution - self.subdomain.nodes[1, :]) ))
    
        self.save.write_vtu(ins, time_step=self.num_step)
        
        self.num_step = self.num_step + 1


    