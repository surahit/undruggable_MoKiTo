from copy import deepcopy
import numpy as np
import openmm as mm
from openmm import unit
from openmmtools.states import ThermodynamicState, SamplerState
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeSampler
from openmmtools.mcmc import*
from openmm.app import PDBFile


def make_rest2_system(system_ref: mm.System, solute_idx: set, s: float) -> mm.System:
    """
    Return a *new* System with REST2 scaling baked into parameters:
    """
    sys = deepcopy(system_ref)
    sqrt_s = np.sqrt(s)
    
    #get the nonbonded force
    for f in sys.getForces():
        if isinstance(f, mm.NonbondedForce):
            nb = f
    """
    nb_forces = [f for f in sys.getForces() if isinstance(f, mm.NonbondedForce)]
    if len(nb_forces) != 1:
        raise RuntimeError("This template assumes exactly one NonbondedForce.")
    nb = nb_forces[0]
    """
    # scale the charge of the solute atoms by sqrt(s) and the attraction term of the LJ potential by s
    for i in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        if i in solute_idx:
            nb.setParticleParameters(i, q*sqrt_s, sig, eps*s)
    
    # Scaling the forces between exceptional pairs

    """
    Exception pairs:
    
    1. By default, OpenMM automatically generates exceptions for 1–2 (direct bond) and 1–3 (bonded via one other atom) bonded pairs (already set to 0):
    
    2. 1–4 pairs (dihedrals): atoms separated by three bonds usually do interact, but force fields often scale their nonbonded interactions.
    These are stored explicitly as exceptions in OpenMM.
    """


    # The proper dihedral potentials for which the first and fourth atoms are in the Hot region (solute atoms) is scaled by a factor s.
    # The proper dihedral potentials for which either the first or the fourth atom is in the Hot region is scaled by a factor sqrt(s). (If some residues of the protein are defined as hot region, not the whole chain)

    # Note: Bonds/angles left untouched (as in PLUMED partial_tempering guidance). 
    # PME long-range is handled automatically by the scaled charges in NonbondedForce.
    
    
    for k in range(nb.getNumExceptions()):
        i, j, qij, sig, eps = nb.getExceptionParameters(k)
        # solute-solute 1-4 nb interactions
        if i in solute_idx and j in solute_idx:
            nb.setExceptionParameters(k, i, j, qij*s, sig, eps*s)
        # solute-solvent 1-4 nb interactions
        elif i in solute_idx or j in solute_idx:
            nb.setExceptionParameters(k, i, j, qij*sqrt_s, sig, eps*sqrt_s)
        # solvent-solvent terms
        else:
            pass 


    # Only 
    # scale torsions: PeriodicTorsionForce and CMAP/Improper if present
    # E = k*(1 + cos(n*theta - phase_offset))
    for f in sys.getForces():
        if isinstance(f, mm.PeriodicTorsionForce):
            for idx in range(f.getNumTorsions()):
                # if all 4 atoms of the dihedral is in the Hot regions, scale the potential by s
                a, b, c, d, n, k, phase = f.getTorsionParameters(idx)
                if a in solute_idx and d in solute_idx:
                    f.setTorsionParameters(idx, a, b, c, d, n, k*s, phase)
                
                elif a in solute_idx and d not in solute_idx:
                    f.setTorsionParameters(idx, a, b, c, d, n, k*sqrt_s, phase)

                elif d in solute_idx and a not in solute_idx:
                    f.setTorsionParameters(idx, a, b, c, d, n, k*sqrt_s, phase)

                else:
                    pass

    return sys

def build_rest2_ladder(system_ref, solute_idx, s_list):
    """Return list of (s, System) with baked-in scaling for each replica."""
    replicas = []
    for s in s_list:
        replicas.append((s, make_rest2_system(system_ref, solute_idx, s)))
    return replicas



def run_rest2_hrex(pdb, system_ref, solute_indices, temperatures, s_list,
                   nsteps=500000, collision_rate=1.0/unit.picosecond,
                   timestep=2.0*unit.femtoseconds, swap_interval=100,
                   platform_name="CUDA"):

    assert len(temperatures) == len(s_list)
    solute_idx = set(solute_indices)

    ladder = build_rest2_ladder(system_ref, solute_idx, s_list)

    # 1) thermodynamic states (all same T for REST2)
    # NVT ensemble is assumed for each state
    thermo_states = [ThermodynamicState(system=sys_s, temperature=T)
                     for (s, sys_s), T in zip(ladder, temperatures)]

    # 2) MCMC move
    move = LangevinDynamicsMove(timestep = timestep,
                                collision_rate = collision_rate,
                                n_steps = swap_interval)

    # 3) reporter + sampler
    reporter = MultiStateReporter("rest2_hrex_4.nc", checkpoint_interval=100)
    sampler = ReplicaExchangeSampler(mcmc_moves=move,
                                     number_of_iterations=int(nsteps//swap_interval),
                                     online_analysis_interval=None,
                                     replica_mixing_scheme='swap-neighbors')

    
    # 4) initial SamplerStates (same coords for all replicas, same box)
    pdb_in = PDBFile(pdb)
    positions = pdb_in.getPositions()
    
    a_vec, b_vec, c_vec = system_ref.getDefaultPeriodicBoxVectors()
   
    sampler_states = [
        SamplerState(positions=positions, box_vectors=(a_vec, b_vec, c_vec))
        for _ in range(len(thermo_states))
    ]
    
    # 5) create + run
    sampler.create(
        thermodynamic_states=thermo_states,
        sampler_states=sampler_states,   
        storage=reporter
    )
    sampler.run()
    reporter.close()
