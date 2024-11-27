from openmm import *
from openmm.app import *
from openmm.unit import *
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from sys import stdout
import mdtraj.reporters
import numpy as np
from time import gmtime, strftime
import copy

def generate_initial_trajectory(
                                inp_dir  =  'input/',
                                out_dir  =  'output/',
                                pdbfile_solute = 'pdbfile_solute.pdb', 
                                smiles = None,
                                file_traj_water  = "trajectory_water.dcd",
                                file_traj_solute = "trajectory_solute.dcd",
                                dt       = 0.002,
                                Nsteps   = 5000,
                                Nframes_solute    = 500,
                                Nframes_water     = 5,
                                solvent = 'water',
                                timesteps_equilibration = 10000,
                                save_vels = False,
                                gamma = 10,
                                Ntemps     = 5,
                                minT       = 300,
                                maxT       = 450,
                                water_padding = 1.3,
                                water_box = None, #[9, 9, 9]
                                platform = 'CUDA',
                                platformProperties  = {'Precision': 'single', 'DeviceIndex':'0'},
                                integrator = 'langevin',
                                NPT = False
                                ):

    # directories
    print(">> Working directories:")
    print("Input files:", inp_dir)
    print("Output files:", out_dir)
    print(" ")
    
    # Integrator parameters
    dt       = 0.002      # ps
    
    # Rates at which frames are saved
    Nout_solute     = int(Nsteps / Nframes_solute) 
    Nout_water      = int(Nsteps / Nframes_water) 
    
    print(">> Integrator parameters:")
    print("Integrator timestep:", str(dt), "ps")
    print("Number of timesteps:", str(Nsteps))
    print('Real time:', str(Nsteps * dt / 1000), "ns")
    print("Saved frames (only molecule):", str(Nframes_solute))
    print("Saved frames (molecule+water):", str(Nframes_water))
    print(" ")
    
    # System parameters
    kB    = 0.008314  # kJ mol-1 K-1 
    T     = minT
    print(">> System parameters:")
    print("Lowest temperature:", str(T), "K")
    print("Friction:", str(gamma), "ps-1\n")
    
    
    # Generate log-file
    log = open(out_dir                                 + "log.txt", 'w')
    log.write('Timestep: '                             + str(dt)     + " ps\n")
    log.write("nsteps: "                               + str(Nsteps) + "\n" )
    log.write("Saved frames (only molecule): "         + str(Nframes_solute)   + "\n")
    log.write("Saved frames (molecule + water): "      + str(Nframes_water)   + "\n")
    log.write("nstxout: "                              + str(Nout_solute)   + "\n")
    log.write("Temperature: "                          + str(T)      + " K\n")
    log.write("Collision rate: "                       + str(gamma)  + " ps-1\n")
    log.write("Solvent: "                              + solvent                 )
    log.write("Boltzmann const.: "                     + str(kB)     + " kJ mol-1 K-1 \n")
    log.write("Simulation start: "                     + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n" )
    log.close();
    
    # Load molecule files
    pdb_solute    =  PDBFile(inp_dir + 'pdbfile_solute.pdb')    


    if integrator == 'langevin':
        integrator = LangevinIntegrator( T * kelvin, gamma/picosecond, dt*picoseconds)
    elif integrator == 'nose_hoover':
        integrator = NoseHooverIntegrator( T * kelvin, 0.5/picosecond, dt*picoseconds)

    print(integrator)
    
    platform   = Platform.getPlatformByName(platform)
    
    if solvent == 'water':
        
        forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")

        if smiles is not None:
            print("smiles = ", smiles)
        
            molecule = Molecule.from_smiles(smiles)
            gaff_template = GAFFTemplateGenerator(molecules=[molecule], forcefield='gaff-2.11')
            forcefield.registerTemplateGenerator(gaff_template.generator)

        # Solvation
        pdb_water = copy.copy(pdb_solute)
        print("Solvation ... \n")
        
        modeller = Modeller(pdb_water.topology, pdb_water.positions)
        
        # cubic water box with a minimum distance of 1 nm to the box boarders
        modeller.addSolvent(forcefield, 
                            boxSize = water_box,
                            padding = water_padding, 
                            neutralize = True)
        
        pdb_water.positions = modeller.getPositions()
        pdb_water.topology  = modeller.getTopology()
        
        with open(inp_dir + "pdbfile_water.pdb", "w") as file_:
            pdb_water.writeFile(
                pdb_water.topology, pdb_water.positions,
                file=file_
            )
            
        system = forcefield.createSystem(pdb_water.topology, 
                                         nonbondedMethod = PME, 
                                         nonbondedCutoff = 1.0*nanometer,
                                         constraints = HBonds)
        
        simulation = Simulation(pdb_water.topology, system, integrator, platform, platformProperties)
        simulation.context.setPositions(pdb_water.positions)
        
        print(">> Molecule parameters:")
        print("Number of atoms (no water):", pdb_solute.topology.getNumAtoms())
        print("Number of water molecules:", int((pdb_water.topology.getNumAtoms() - pdb_solute.topology.getNumAtoms())/3))
        print("Total number of atoms:", pdb_water.topology.getNumAtoms())
    
    elif solvent == 'implicit':
        
        forcefield = ForceField("amber14/protein.ff14SB.xml", "implicit/obc2.xml")
        
        if smiles is not None:
            print("smiles = ", smiles)
        
            molecule = Molecule.from_smiles(smiles)
            gaff_template = GAFFTemplateGenerator(molecules=[molecule], forcefield='gaff-2.11')
            forcefield.registerTemplateGenerator(gaff_template.generator)
            
        system = forcefield.createSystem(pdb_solute.topology, 
                                         nonbondedMethod = CutoffNonPeriodic,
                                         nonbondedCutoff = 1.0 * nanometer,
                                         constraints = None)
        if NPT == True:
            system.addForce(MonteCarloBarostat( 1 * bar, T * kelvin ))

        simulation = Simulation(pdb_solute.topology, system, integrator, platform)
        simulation.context.setPositions(pdb_solute.positions)
        
        print(">> Molecule parameters:")
        print("Number of atoms (no water):", pdb_solute.topology.getNumAtoms())
            

    print(" ")

    
    # minimization
    print('\n\n*** Minimizing ...')
    simulation.minimizeEnergy()
    print('*** Minimization completed ***') 
    
    # equilibration
    simulation.context.setVelocitiesToTemperature(T)
    print('\n\n*** Equilibrating...')

    """
    if solvent == 'water':
            simulation.reporters.append(
            StateDataReporter(
            out_dir + "equilibration.log", 1, step=True,
            potentialEnergy=True, totalEnergy=True,
            temperature=True, progress=False,
            remainingTime=False, speed=False,
            totalSteps=timesteps_equilibration,
            separator='\t')
            )
    """     
    simulation.step(timesteps_equilibration)
    print('*** Equilibration completed ***')
        
    # print on screen
    simulation.reporters.append(StateDataReporter(stdout, 
                                                  10000, 
                                                  speed = True, 
                                                  step=True, 
                                                  potentialEnergy=True, 
                                                  temperature=True))
    
    # save trajectory
    #if solvent == 'water':
    simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dir + file_traj_water, Nout_water))
    
    simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dir + file_traj_solute, Nout_solute, 
                                atomSubset=range(pdb_solute.topology.getNumAtoms())))


    simulation_st = simulatedtempering.SimulatedTempering(simulation, 
                                                          numTemperatures=Ntemps, 
                                                          minTemperature=minT, 
                                                          maxTemperature=maxT,
                                                          reportInterval =Nframes_solute,
                                                          reportFile  = out_dir + 'log_SimulatedTempering.txt' )
    
    # Save velocities
    if save_vels == 1:
        
        i = 0
        for step in range(1, Nsteps + 1):
            simulation_st.step(1)
            if step % (Nout_water) == 0:
                simulation.saveState(out_dir + 'initial_states/x0_' +str(i) + '.xml')
                i=i+1
                
    elif save_vels == 0:
        simulation_st.step(Nsteps)

    
    
        
    
    # add total calculation time to LOG-file
    log = open(out_dir + "log.txt", 'a')
    log.write("Simulation end: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )
    log.close();
    
    # end
    print('\n\n****** SIMULATION COMPLETED *****************************\n\n')
