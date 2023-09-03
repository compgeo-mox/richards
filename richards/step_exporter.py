from richards.matrix_computer import Matrix_Computer

import porepy as pp
import pygeon as pg



class Step_Exporter:

    def __init__(self, mdg, sol_name, output_directory, key='flow'):
        self.subdomains = []

        self.dof_q = []
        self.dof_psi = []

        self.num_step = 0

        self.save = pp.Exporter(mdg, sol_name, folder_name=output_directory)
        self.computer = Matrix_Computer(mdg)


        for subdomain in mdg.subdomains(return_data=False):
            rt0 = pg.RT0(key)
            p0 = pg.PwConstants(key)

            self.subdomains.append ( subdomain )

            self.dof_q.append( rt0.ndof(subdomain) )
            self.dof_psi.append( p0.ndof(subdomain) )


    
    def export(self, solutions):
        proj_q = self.computer.compute_proj_q_mat()
        proj_psi = self.computer.compute_proj_psi_mat()

        for i in range(len(solutions)):
            solution = solutions[i]
            q   = solution[:self.dof_q[i]]
            psi = solution[-self.dof_psi[i]:]

            ins = list()

            ins.append((self.subdomains[i], "cell_q", ( proj_q[i] @ q).reshape((3, -1), order="F")))
            ins.append((self.subdomains[i], "cell_p", proj_psi[i] @ psi))
    
            self.save.write_vtu(ins, time_step=self.num_step)
        
        self.num_step = self.num_step + 1


    
    def export_final_pvd(self, times):
        self.save.write_pvd(times)