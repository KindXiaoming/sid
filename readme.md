# Sparse Invariant Detector (SID)
This is the code repo for the paper: ["Discovering New Interpretable Conservation Laws as Sparse Invariants"](https://arxiv.org/abs/2305.19525). We propose Sparse Invariant Detector, SID, which is able to discover all conserved quantities, given the knowledge of differential equations and the basis functions. Each example is made self-consistent in jupyter notebooks. 

|Examples| Notebook |
|--|--|
|System Biology|toy.ipynb|
|Fluid Mechanics|  fluid.ipynb (2D), fluid_3D.ipynb (3D) |
|Atmospheric Chemistry |  chem.ipynb (w/o H2O), chem_H2O.ipynb (w/ H2O) 

If you want to try your own differential equations $\dot{x}=f(x)$, the only two things you need to modify to toy.ipynb are (1) specify $f(x)$; (2) specify basis functions, by creating basis.txt. If you're dealing with polynomial bases, "create_ploy_basis_file" can help you create the file. If you're working with other functions, you will need to create your own basis.txt.


