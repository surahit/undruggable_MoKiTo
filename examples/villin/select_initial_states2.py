# %%
from openmm import *
from openmm.app import *
from openmm.unit import *
from copy import copy

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import shutil

def copy_rename_paste_file(source_file_path, destination_dir, new_file_name):
    # Check if source file exists
    if not os.path.exists(source_file_path):
        print(f"Source file '{source_file_path}' does not exist.")
        return
    
    # Check if destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory '{destination_dir}'")

    # Get the full path of the new file
    new_file_path = os.path.join(destination_dir, new_file_name)

    # Copy the file to the destination directory
    shutil.copy(source_file_path, new_file_path)


timesteps_equilibration = 50000
traj = None

inp_dir       =  'input/'
out_dir       =  'output/trajectories/openmm_files/'


pdbfile_water    = 'pdbfile_water.pdb'
file_traj_water  = 'trajectory_water.dcd'

pdb_water       = PDBFile(inp_dir + pdbfile_water) # this file is used to count atoms of the molecule

forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")

system = forcefield.createSystem(pdb_water.topology, 
                                nonbondedMethod=PME, 
                                nonbondedCutoff=1.0*nanometer, 
                                constraints=HBonds)

platform = Platform.getPlatformByName('CUDA')   
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
mdtraj_top = md.load(inp_dir + pdbfile_water).topology  # Or any MDTraj-compatible topology

k = -1
for r in range(6):

    out_dir2      =  'rep' + str(r) + '/output/'
    traj_rep      =  md.load(out_dir2 + file_traj_water, top = inp_dir + pdbfile_water)

    for i in tqdm(np.arange(0,2000,2)):

    
        integrator_ = copy(integrator)
        simulation = Simulation(pdb_water.topology, system, integrator_, platform)
        
        # Get the positions of the atoms from the current frame
        positions = traj_rep[i].xyz[0]
        
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(300)
        simulation.step(timesteps_equilibration)
        state = simulation.context.getState(getPositions=True)
        openmm_positions = state.getPositions().value_in_unit(nanometer)  # OpenMM positions (Quantity object)

        new_x0 = md.Trajectory(openmm_positions, topology=mdtraj_top)

        if traj is None:
            traj = new_x0
        else:
            traj = traj + new_x0

        k = k + 1
        new_filename  = 'x0_' + str(k) + '.xml'
                        
        simulation.saveState(out_dir + 'initial_states/' + new_filename)

        del simulation
        del integrator_

file_traj_water  = "trajectory_water.dcd"
file_traj_solute = "trajectory_solute.dcd"

traj.save_dcd(out_dir + file_traj_water)
traj.remove_solvent().save_dcd(out_dir + file_traj_solute)

# %%



