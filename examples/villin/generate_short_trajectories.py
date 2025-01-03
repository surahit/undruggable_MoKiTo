import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
import os

# Add the project root
sys.path.append(os.path.abspath('../../'))

from src.useful_functions import *
from src.openmm.short_trajectories_module import generate_short_trajectories

# For reproducibility
np.random.seed(0)

# For matplotlib
font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

# Read directory paths
read_dirs_paths('dir_paths.txt', globals())
check_directories(out_trajectories1 ,out_trajectories2 ,out_trajectories3 ,out_trajectories4)


pdbfile_solute = 'pdbfile_solute.pdb'
pdbfile_water = 'pdbfile_water.pdb'

file_traj_water  = "trajectory_water.dcd"
file_traj_solute = "trajectory_solute.dcd"

generate_short_trajectories(
                                inp_dir  =  inp_dir,
                                out_dir  =  out_trajectories2,
                                out_dir2 =  out_trajectories4,
                                pdbfile_solute = 'pdbfile_solute.pdb', 
                                pdbfile_water  = 'pdbfile_water.pdb',
                                file_traj_water  = "trajectory_water.dcd",
                                file_traj_solute = "trajectory_solute.dcd",
                                dt       = 0.002,
                                Nsteps   = 500,
                                Nframes_solute    = 10,
                                Nframes_water     = 0,
                                solvent = 'water',
                                NfinPoints = 10,
                                gamma = 1,
                                T     = 300,
                                use_initial_vels = False,
                                integrator = 'langevin'
                                )




