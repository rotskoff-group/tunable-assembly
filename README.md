# Microscopic origin of tunable assembly forces in chiral active environments

Official implementation of:

**Microscopic origin of tunable assembly forces in chiral active environments**

Clay H. Batton and Grant M. Rotskoff

<https://arxiv.org/abs/2310.17763>

**Abstract**: The fluctuations of a nonequilibrium bath enable dynamics inaccessible to any equilibrium system.
Exploiting the driven dynamics of active matter in order to do useful work has become a topic of significant experimental and theoretical interest. 
Due to the unique modalities controlling self-assembly, the interplay between passive solutes and the particles in an active bath has been studied as a potential driving force to guide assembly of otherwise non-interacting objects.
Here, we investigate and characterize the microscopic origins of the attractive and repulsive interactions between passive solutes in an active bath.
We show that, while assembly does not occur dynamically for achiral active baths, chiral active particles can produce stable and robust assembly forces.
We both explain the observed oscillatory force profile for active Brownian particles and demonstrate that chiral active motion leads to fluxes consistent with an odd diffusion tensor that, when appropriately tuned, produces long-ranged assembly forces. 

This folder contains three subfolders

1. [hoomd-blue](https://github.com/rotskoff-group/tunable-assembly/tree/main/hoomd-blue): a modified implementation of the Brownian dynamics integrator of HOOMD-blue v3.11.0 that outputs the fluxes as velocities.
2. [odd-diffusion](https://github.com/rotskoff-group/tunable-assembly/tree/main/odd-diffusion): code for obtaining the numerical solution for the odd diffusion system.
3. [simulation-scripts](https://github.com/rotskoff-group/tunable-assembly/tree/main/simulation-scripts): scripts for the simulation studies in the paper, with the general workflow of using `calculate_rg.py`, `generate.py`, `equilibriate.py`, `equilibriate_active.py`, and then `run.py` in that order.
