from openmm import *
from openmm.app import *
from openmm.unit import *
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from sys import stdout
import mdtraj.reporters
import numpy as np
from time import gmtime, strftime
import os
import mdtraj as md
# directories

name_project = '33dichlorisobutene/water4/'

inp_dir      =  '/data/numerik/people/ldonati/' + name_project + 'input/'
out_dir      =  '/scratch/htc/ldonati/' + name_project + 'output/'

forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")

smiles ='C=C(C)C(Cl)Cl'

if smiles is not None:
    print("smiles = ", smiles)

    molecule = Molecule.from_smiles(smiles)
    gaff_template = GAFFTemplateGenerator(molecules=[molecule], forcefield='gaff-2.11')
    forcefield.registerTemplateGenerator(gaff_template.generator)


#traj_solute = md.load_dcd(out_dir + 'trajectory.dcd', top=inp_dir + 'pdbfile_no_water.pdb')
#traj_water  = md.load_dcd(out_dir + 'trajectory_water.dcd', top=inp_dir + 'pdbfile_water.pdb')

pdb = PDBFile(inp_dir + 'pdbfile_solute.pdb')

#system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)

import parmed

# Create a modeller instance
modeller = Modeller(pdb.topology, pdb.positions)

# Optionally, modify the modeller, e.g., add missing hydrogens
# modeller.addHydrogens(forcefield)

# Create the system
system = forcefield.createSystem(modeller.topology)

# Convert to a ParmEd structure
structure = parmed.openmm.load_topology(modeller.topology, system)

# Save the ParmEd structure to a .prmtop file
structure.save(inp_dir + 'prmtopfile_solute.prmtop', format='amber')