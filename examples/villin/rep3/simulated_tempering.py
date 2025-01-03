import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
import os

# Add the project root
sys.path.append(os.path.abspath('../../../'))

from src.openmm.simulated_tempering_module import generate_initial_trajectory

# For reproducibility
np.random.seed(3)

inp_dir = '../input/'
out_dir = 'output/'

pdbfile_solute = 'pdbfile_solute.pdb'
pdbfile_water = 'pdbfile_water.pdb'

file_traj_water  = "trajectory_water.dcd"
file_traj_solute = "trajectory_solute.dcd"

generate_initial_trajectory(
                                inp_dir  =  inp_dir,
                                out_dir  =  out_dir,
                                pdbfile_solute = pdbfile_solute, 
                                file_traj_water  = file_traj_water,
                                file_traj_solute = file_traj_solute,
                                dt       = 0.002,
                                Nsteps   = 500000000,
                                Nframes_solute    = 2000,
                                Nframes_water     = 2000,
                                solvent = 'water',
                                timesteps_equilibration = 1,
                                save_vels = False,
                                gamma = 1,
                                Ntemps     = 20,
                                minT       = 273,
                                maxT       = 500,
                                water_padding = None,
                                water_box = [5.4, 5.4, 5.4],
                                platform = 'CUDA'
                            )
