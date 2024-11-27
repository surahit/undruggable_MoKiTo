from openmm import *
from openmm.app import *
from openmm.unit import *
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from sys import stdout
import mdtraj.reporters
import numpy as np
from time import gmtime, strftime
import mdtraj as md
from tqdm import tqdm
from copy import copy

def generate_short_trajectories(
                                inp_dir   =  'input/',
                                out_dir   =  'output/',
                                out_dir2  =  'output/final_states/',
                                pdbfile_solute = 'pdbfile_solute.pdb', 
                                pdbfile_water  = 'pdbfile_water.pdb',
                                smiles = None,
                                file_traj_water  = "trajectory_water.dcd",
                                file_traj_solute = "trajectory_solute.dcd",
                                dt       = 0.002,
                                Nsteps   = 1000,
                                Nframes_solute    = 10,
                                Nframes_water     = 0,
                                solvent = 'water',
                                NfinPoints = 10,
                                gamma = 10,
                                T     = 300,
                                use_initial_vels = False,
                                integrator = 'langevin'
                                ):

    print(">> Working directories:")
    print("Input files:", inp_dir)
    print("Output files:", out_dir)
    print("Output files2:", out_dir2)   
    print(" ")
    
    # Starting points (number of files in input/initial_states)
    traj_water       = md.load_dcd(out_dir + file_traj_water, top=inp_dir + pdbfile_water)
    pdb_solute       = PDBFile(inp_dir + pdbfile_solute) # this file is used to count atoms of the molecule
    Natoms           = pdb_solute.topology.getNumAtoms()
    Npoints          = traj_water.n_frames
    print("Number of initial states:", Npoints)
    
    print("Number of final states:", NfinPoints)
    print(" ")
    
    
    # Rate at which frames are saved
    Nout     = int(Nsteps / Nframes_solute) 
    
    print(">> Integrator parameters:")
    print("Integrator timestep:", str(dt), "ps")
    print("Number of timesteps:", str(Nsteps))
    print('Real time:', str(Nsteps * dt / 1000), "ns")
    print("Saved frames (only molecule):", str(Nframes_solute))
    print(" ")
    
    
    # System parameters
    kB    = 0.008314  # kJ mol-1 K-1 
    
    print(">> System parameters:")
    print("Temperature:", str(T), "K")
    print("Friction:", str(gamma), "ps-1\n")
    
    # LOG-file
    log = open(out_dir2                                 + "log.txt", 'w')
    log.write("Number of initial states: "             + str(Npoints) + "\n" )
    log.write("Number of final states: "               + str(NfinPoints) + "\n" )
    log.write('Timestep: '                             + str(dt)     + " ps\n")
    log.write("nsteps: "                               + str(Nsteps) + "\n" )
    log.write("Saved frames (only molecule): "         + str(Nframes_solute)   + "\n")
    log.write("Temperature: "                          + str(T)      + " K\n")
    log.write("Collision rate: "                       + str(gamma)  + " ps-1\n")
    log.write("Boltzmann const.: "                     + str(kB)     + " kJ mol-1 K-1 \n")
    log.write("Simulation start: "                     + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n" )
    log.close();
    
    forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")
    
    if smiles is not None:
        print("smiles = ", smiles)
    
        molecule = Molecule.from_smiles(smiles)
        gaff_template = GAFFTemplateGenerator(molecules=[molecule], forcefield='gaff-2.11')
        forcefield.registerTemplateGenerator(gaff_template.generator)
    
    pdb_water = PDBFile(inp_dir + pdbfile_water)
    system = forcefield.createSystem(pdb_water.topology, 
                                     nonbondedMethod=PME, 
                                     nonbondedCutoff=1.0*nanometer, 
                                     constraints=HBonds)
    
    
    print(">> Molecule parameters:")
    print("Number of atoms (no water):", pdb_solute.topology.getNumAtoms() )
    print("Number of water molecules:", int((pdb_water.topology.getNumAtoms() - pdb_solute.topology.getNumAtoms()) / 3))
    print("Total number of atoms:",  pdb_water.topology.getNumAtoms())
    print(" ")

    #integrator_ = LangevinMiddleIntegrator(T*kelvin, gamma/picosecond, dt*picoseconds)
    platform = Platform.getPlatformByName('CUDA')   
    
    for i in tqdm(range(Npoints)):
          
        # Get the positions of the atoms from the current frame
        positions = traj_water.xyz[i]
                
        # Num replicas per initial state
        for r in range(NfinPoints):
            
            # set-up simulation 
        
            if integrator == 'langevin':
                integrator_ = LangevinIntegrator( T * kelvin, gamma/picosecond, dt*picoseconds)
            elif integrator == 'nose_hoover':
                integrator_ = NoseHooverIntegrator( T * kelvin, 0.5/picosecond, dt*picoseconds)
                    
            simulation = Simulation(pdb_water.topology, system, integrator_, platform)
            
            if use_initial_vels == False:
                simulation.context.setPositions(positions)
                simulation.context.setVelocitiesToTemperature(T)
            elif use_initial_vels == True:
                simulation.loadState(out_dir + 'initial_states/x0_' +str(i) + '.xml')
                    
            simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dir2 + 'xt_' + str(i) + '_r' + str(r) + '.dcd', Nout, atomSubset=range(pdb_solute.topology.getNumAtoms())))
            
            # repeat procedure for nsteps
            simulation.step(Nsteps)
    
    # add total calculation time to LOG-file
    log = open(out_dir + 'final_states/log.txt', 'a')
    log.write("Simulation end: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )
    log.close();
    
    # end
    print('\n\n****** SIMULATION COMPLETED *****************************\n\n')
