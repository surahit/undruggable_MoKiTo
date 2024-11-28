### Preamble
This repository contains Python code and Jupyter Notebooks for reproducing the results presented in the manuscript *Topological analysis reveals multiple pathways in molecular dynamics*.

The same code and processed data are also available on Zenodo: [https://doi.org/10.5281/zenodo.14229803](https://doi.org/10.5281/zenodo.14229803)

# MoKiTo: Molecular Kinetics via Topology
MoKiTo (Molecular Kinetics via Topology) is a Python-based toolkit designed for analyzing and extracting topological insights from Molecular Dynamics (MD) simulations.   
It enables researchers to study conformational transitions and kinetic pathways in molecular systems by generating Molecular Kinetics Maps (MKMs).

## Installation
git clone https://github.com/donatiluca/MoKiTo.git  
pip install -r requirements.txt


## How to Use MoKiTo
The workflow depends on whether you're working with toy model systems (typically one- or two-dimensional systems driven by overdamped Langevin dynamics) or all-atom molecular systems.  
The directory `examples/` contains sample workflows for both use cases. Modify the scripts as needed for your specific data and analysis.

### Low-Dimensional Systems

For low-dimensional systems, follow these steps:

1. Use `generate_trajectories.ipynb` to create the initial trajectory and the short trajectories.
2. Use `isokann.ipynb` to learn the $\chi$-function.
3. Use `mokito.ipynb` to load the trajectories and the $\chi$-function to generate the MKM and the energy landscape.

### Molecular systems

For MD simulations, ensure the relevant `.pdb` file is placed into the `input/` directory. 
Then follow these steps:

1. Run `generate_initial_trajectory.ipynb` to create the initial trajectory.
2. Use `generate_short_trajectories.ipynb` to sample the system's dynamics.
4. Run `calculate_PWDs.ipynb` to convert the MD trajectories saved as `.dcd` files into `.pt` arrays that contain the pairwise distance matrices.
5. Use `isokann.ipynb` to learn the $\chi$-function.
6. Use `mokito.ipynb` to load the pairwise distance matrices and the $\chi$-function to generate the MKM and the energy landscape.