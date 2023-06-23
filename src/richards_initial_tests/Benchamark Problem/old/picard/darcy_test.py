import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


N = 4
sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)

mdg = pg.as_mdg(sd)


pg.convert_from_pp(sd)
sd.compute_geometry()


key = "flow"
bc_val = []
bc_ess = []

RT0 = pg.RT0(key)
P0 = pg.PwConstants(key)

for sd, data in mdg.subdomains(return_data=True):
    perm = pp.SecondOrderTensor(np.ones(sd.num_cells))
    parameters = {
        "second_order_tensor": perm,
    }
    pp.initialize_data(sd, data, key, parameters)

    left_right = np.logical_or(sd.face_centers[0, :] == 0, sd.face_centers[0, :] == 1)
    top_bottom = np.logical_or(sd.face_centers[1, :] == 0, sd.face_centers[1, :] == 1)
    ess_p_dofs = np.zeros(P0.ndof(sd), dtype=bool)

    def p_bc(x):
        return x[1]

    bc_val.append(-RT0.assemble_nat_bc(sd, p_bc, top_bottom))
    bc_ess.append(np.hstack((left_right, ess_p_dofs)))


face_mass = pg.face_mass(mdg)
cell_mass = pg.cell_mass(mdg)
div = cell_mass @ pg.div(mdg)

dt = 0.1
beta = 0.1

spp = sps.bmat([[face_mass,                 -div.T], 
                [div,        beta * cell_mass / dt]], format="csc")


dof_p, dof_q = div.shape


rhs = np.zeros(dof_p + dof_q)
rhs[:dof_q] += np.hstack(bc_val)


p = np.zeros(dof_p)
for n in np.arange(2):
    rhs_loop = rhs.copy()
    rhs_loop[-dof_p:] += cell_mass @ p / dt

    for m in np.arange(100):
        rhs_picard = rhs_loop.copy()
        rhs_picard[-dof_p:] += (beta - 1) * cell_mass @ p / dt

        ls = pg.LinearSystem(spp, rhs_picard)
        ls.flag_ess_bc(np.hstack(bc_ess), np.zeros(dof_q + dof_p))
        x = ls.solve()

        q = x[:dof_q]
        p = x[-dof_p:]

        print(p)

    print("=====")

proj_q = RT0.eval_at_cell_centers(sd)
cell_q = (proj_q * q).reshape((3, -1), order="F")
cell_p = P0.eval_at_cell_centers(sd) * p

for _, data in mdg.subdomains(return_data=True):
    pp.set_solution_values("cell_q", cell_q, data, 0)
    pp.set_solution_values("cell_p", cell_p, data, 0)

save = pp.Exporter(mdg, "sol")
save.write_vtu(["cell_q", "cell_p"])
