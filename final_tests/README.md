This directory contains the collection of tests we performed and decided to insert into the thesis, namely:
- Richards: The tests related to Richards's problem are contained in this directory. In particular:
    - 'primal_L_test.py' and 'dual_L_test.py' will reproduce the tests on the benchmark problem with the L-scheme (variable L, variable N, variable time step length)
    - 'primal_mesh_test.py' and 'dual_mesh_test.py' will reproduce the tests on the benchmark problem with the modified Picard method, the Newton method, and the L-scheme (with $L_1=3.5 \cdot 10^{-2}$ and $L_2=4.5 \cdot 10^{-2}$)
    - 'primal_single_stage.ipynb' and 'dual_single_stage.ipynb' are the notebook files that can be used to quickly launch a Richards's problem simulation with a specific nonlinear solver
    - 'primal_multi_stage.ipynb' and 'dual_multi_stage.ipynb' are the notebook files that can be used to quickly launch a Richards's problem simulation with a combination of nonlinear solvers (i.e., the first N iterations or up to a particular error threshold are performed with a specific scheme, while the remaining iterations are performed with a different scheme)
- Coupling Darcy-Richards: In this directory, it's contained the notebook 'lagrange_multipler.ipynb', which can be run to solve the fixed-domain Darcy-Richards's problem with the $L$-scheme
- Moving Darcy: This directory contains the notebook 'primal. ipynb', which can be used to run the moving domain Darcy's problem in both the original paper's formulation and the corrected version.
