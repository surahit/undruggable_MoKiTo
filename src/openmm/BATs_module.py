import MDAnalysis as mda
import os
import mdtraj as md
import numpy as np
import sympy as sp
import torch as pt
from tqdm import tqdm
import scipy
import itertools
import matplotlib.pyplot as plt
import glob
from openmm import *
from openmm.app import *
from openmm.unit import *

font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

np.random.seed(0)
pt.manual_seed(0)


print(" ")
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


def generate_pairs(N):
    #return np.c_[np.array(np.meshgrid(np.arange(N), np.arange(N))).T.reshape(-1,2)]
    t = np.arange(0,N,1)
    return np.array(list(set(itertools.combinations(t, 2))))

def generate_BAT_torch(
                        inp_dir  =  'input/',
                        out_dir  =  'output/',
                        iso_dir  =  'ISOKANN_files/',
                        pdbfile_solute    = 'pdbfile_no_water.pdb', 
                        pdbfile_water     = 'pdbfile_water.pdb', 
                        prmtopfile_solute = "prmtopfile_no_water.prmtop", 
                        file_traj_water   = "trajectory_water.dcd",
                        file_traj_solute  = "trajectory.dcd",
                        frames     = np.array([0,1,2,3,4,5,6,7,8,9]),
                        rel_coords = np.array([[0, 70]]),
                        periodic   = False,
                        BB=False):
    
        
    if BB==True:
        print(" ")
        print("This will create torch files containing BAT coordinates using only BACKBONE atoms")
        print(" ")
    else:
        print(" ")
        print("This will create torch files containing BAT coordinates using ALL atoms")
        print(" ")
    
    # Starting points (number of files in input/initial_states)
    traj     = md.load(out_dir + file_traj_water, top = inp_dir + pdbfile_water)   
    Npoints  = traj.n_frames
    
    # Load the trajectory with MDA to find bonda angles and dihedrals
    mda_traj     = mda.Universe(inp_dir + prmtopfile_solute, out_dir + file_traj_solute)
    
    
    print("Number of initial states:", Npoints)
    
    _, _, files = next(os.walk(out_dir + 'final_states/'))
    Nfinpoints = int(( len(files) - 1 ) / Npoints)
    print("Number of final states:", Nfinpoints)
    
    #from MDAnalysis.analysis.bat import BAT
    #R = BAT(traj)
    
    
    pdb = mda.Universe(inp_dir + pdbfile_solute)
    Natoms = pdb.atoms.n_atoms
    print("Number of atoms (no water):", Natoms)
    
    
    # Calculate relevant coordinate
    print("I am generating the relevant coordinate...")
    if len(rel_coords[0])==2:
        print("len(rel_coords)==2, then the relevant coordinate is a DISTANCE between 2 atoms")
        r        = np.squeeze(md.compute_distances(traj, rel_coords, periodic=False))
    elif len(rel_coords[0])==3:
        print("len(rel_coords)==3, then the relevant coordinate is an ANGLE between 3 atoms")
        r        = np.squeeze(md.compute_angles(traj, rel_coords, periodic=True))
    elif len(rel_coords[0])==4:
        print("len(rel_coords)==4, then the relevant coordinate is a DIHEDRAL between 4 atoms")
        r        = np.squeeze(md.compute_dihedrals(traj, rel_coords, periodic=True))
    
    print(" ")
    np.savetxt(iso_dir + 'R0.txt', r)
    
    # Select backbone atoms 
    if BB == True:
    
        bb            = traj.topology.select("backbone")
        mda_bb        = mda_traj.select_atoms("backbone")
           
        print("Number of backbone atoms:", mda_bb.n_atoms)
        
    
        # The indices correspond to the full trajectory, then in the computation of bonds, angles, dihedral, I use the full trajectory
        indices_bonds     = mda_bb.bonds.indices
        indices_angles    = mda_bb.angles.indices
        indices_torsions  = mda_bb.dihedrals.indices
    
        Ndims     =   indices_bonds.shape[0] + indices_angles.shape[0] + indices_torsions.shape[0]
        print("Number of BAT dimensions:", Ndims )
        
        
            
        # Load initial states
        print("I am creating the tensor with the initial states...")
        b0  =  md.compute_distances(traj, indices_bonds, periodic=False)
        a0  =  md.compute_angles(traj, indices_angles, periodic=True)
        t0  =  md.compute_dihedrals(traj, indices_torsions, periodic=True)
        
        B0  =  pt.tensor(b0, dtype=pt.float32, device=device)
        A0  =  pt.tensor(a0, dtype=pt.float32, device=device)
        T0  =  pt.tensor(t0, dtype=pt.float32, device=device)
        
    else:
        
        indices_bonds     = mda_traj.bonds.indices
        indices_angles    = mda_traj.angles.indices
        indices_torsions  = mda_traj.dihedrals.indices
    
        Ndims     =   indices_bonds.shape[0] + indices_angles.shape[0] + indices_torsions.shape[0]
    
        print("Number of BAT dimensions:", Ndims )
            
        # Load initial states
        print("I am creating the tensor with the initial states...")
        b0  =  md.compute_distances(traj, indices_bonds, periodic=False)
        a0  =  md.compute_angles(traj, indices_angles, periodic=True)
        t0  =  md.compute_dihedrals(traj, indices_torsions, periodic=True)
    
        B0  =  pt.tensor(b0, dtype=pt.float32, device=device)
        A0  =  pt.tensor(a0, dtype=pt.float32, device=device)
        T0  =  pt.tensor(t0, dtype=pt.float32, device=device)
        
    
    
    
    
    #pt.save(B0, iso_dir + 'Bonds_0.pt')
    #pt.save(A0, iso_dir + 'Angles_0.pt')
    #pt.save(T0, iso_dir + 'Torsions_0.pt')
    
    BAT0 = pt.cat([B0, A0, T0], axis=1)
    
    print('Shape of BAT0?')
    print('Npoints, Ndims')
    print(BAT0.shape)
    print(" ")
    
    pt.save(BAT0, iso_dir + 'BAT_0.pt')
    
    
    # Load one trajectory to calculate number of frames
    print("I am creating the tensor with the final states...")
    xt         =  md.load(out_dir + "final_states/xt_0_r0.dcd", 
                                  top = inp_dir + pdbfile_solute)
    print("The shape of a file xf_i_rj.dcd is", xt.xyz.shape)
    Ntimesteps = xt.n_frames
    
    
    Ntimesteps = len(frames)
    
    
    Bt = pt.zeros((Ntimesteps, Npoints, Nfinpoints, indices_bonds.shape[0]    ), dtype = pt.float32, device=device)
    At = pt.zeros((Ntimesteps, Npoints, Nfinpoints, indices_angles.shape[0]   ), dtype = pt.float32, device=device)
    Tt = pt.zeros((Ntimesteps, Npoints, Nfinpoints, indices_torsions.shape[0] ), dtype = pt.float32, device=device)
    
    for i in tqdm(range(Npoints)):
        
        for j in range(Nfinpoints):
            xt         =  md.load(out_dir + "final_states/xt_" + str(i) + "_r" + str(j) + ".dcd", 
                                  top = inp_dir + pdbfile_solute)
    
            
            for k in range(Ntimesteps):
                frame = frames[k]
                if BB == True:
                    bt  =  md.compute_distances(xt[frame], indices_bonds, periodic=False)
                    at  =  md.compute_angles(xt[frame], indices_angles, periodic=True)
                    tt  =  md.compute_dihedrals(xt[frame], indices_torsions, periodic=True)
    
                else:
                    bt  =  md.compute_distances(xt[frame], indices_bonds, periodic=False)
                    at  =  md.compute_angles(xt[frame], indices_angles, periodic=True)
                    tt  =  md.compute_dihedrals(xt[frame], indices_torsions, periodic=True)
            
                Bt[k,i,j,:]  =  pt.tensor(bt, dtype=pt.float32, device=device)
                At[k,i,j,:]  =  pt.tensor(at, dtype=pt.float32, device=device)
                Tt[k,i,j,:]  =  pt.tensor(tt, dtype=pt.float32, device=device)
    
    
    #pt.save(Bt, iso_dir + 'Bonds_t.pt')
    #pt.save(At, iso_dir + 'Angles_t.pt')
    #pt.save(Tt, iso_dir + 'Torsions_t.pt')
    
    BATt = pt.cat([Bt, At, Tt], axis=3)
    
    print(" ")
    print('Shape of BATt?')
    print('Nframes, Npoints, Nfinpoints, Ndims')
    print(BATt.shape)
    
    pt.save(BATt, iso_dir + 'BAT_t.pt')
