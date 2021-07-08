import numpy as np
import pandas as pd
import fire

from simtk import openmm

import polychrom
import polychrom.hdf5_format
import polychrom.simulation
import polychrom.forces
import polychrom.forcekits
import polychrom.starting_conformations
from simtk.openmm.openmm import Platform
import wiggin.starting_mitotic_conformations


def run_simulation(out_dir, chain_length, mt_len, int_region, int_r, int_n, fix_tel, rep_e, timesteps, gap_p):
    PLATFORM = 'CUDA'
    NUMBER_CHAINS = 1
    WIGGLE_DISTANCE = 0.05
    COLLISION_RATE = 0.003
    ERROR_TOLERANCE = 0.005
    CEN_TETHER_R_MIN = 0.2
    CEN_TETHER_R_MAX = 0.25
    BOND_LENGTH = 1.0 / (gap_p + 1)
    NUCLEUS_R = 100

    assert openmm.Platform_getPlatformByName(PLATFORM)

    reporter = polychrom.hdf5_format.HDF5Reporter(folder=out_dir, max_data_length=5, overwrite=True)

    # create chromosome and microtubule chains
    chrom_number_particles = chain_length * NUMBER_CHAINS
    total_number_particles = chrom_number_particles + mt_len * NUMBER_CHAINS * 2 # two microtubules per chromosome

    chroms = pd.DataFrame([[i, i + chain_length, (2 * i + chain_length) // 2] for i in range(0, chrom_number_particles, chain_length)], columns=['start', 'end', 'cen'])
    mts = pd.DataFrame([[i, i + mt_len] for i in range(chrom_number_particles, total_number_particles, mt_len)], columns=['start', 'end'])

    chrom_chains = list(zip(chroms['start'], chroms['end'], [False] * NUMBER_CHAINS))
    mt_chains = list(zip(mts['start'], mts['end'], [False] * NUMBER_CHAINS * 2))

    sim = polychrom.simulation.Simulation(
        platform=PLATFORM, 
        integrator='variableLangevin',
        error_tol=ERROR_TOLERANCE,
        GPU='0',
        collision_rate=COLLISION_RATE,
        N=total_number_particles,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
    )

    # create chromosome and microtubule chains
    chrom_number_particles = chain_length * NUMBER_CHAINS
    total_number_particles = chrom_number_particles + mt_len * NUMBER_CHAINS * 2 # two microtubules per chromosome

    chroms = pd.DataFrame([[i, i + chain_length, (2 * i + chain_length) // 2] for i in range(0, chrom_number_particles, chain_length)], columns=['start', 'end', 'cen'])
    mts = pd.DataFrame([[i, i + mt_len] for i in range(chrom_number_particles, total_number_particles, mt_len)], columns=['start', 'end'])

    chrom_chains = list(zip(chroms['start'], chroms['end'], [False] * NUMBER_CHAINS))
    mt_chains = list(zip(mts['start'], mts['end'], [False] * NUMBER_CHAINS * 2))
    # create centromere-microtubule bonds, 2 microtubules per centromere

    cen_mt_bonds = []
    for i in range(NUMBER_CHAINS):
        cen = chroms.iloc[i]['cen']
        cen_mt_bonds.append((cen, mts.iloc[i]['end'] - 1))
        cen_mt_bonds.append((cen, mts.iloc[(i + len(mts)) // 2]['end'] - 1))

    sim.add_force(
        polychrom.forcekits.polymer_chains(
            sim,
            chains=chrom_chains + mt_chains,
            bond_force_func=polychrom.forces.harmonic_bonds,
            bond_force_kwargs={
                'bondLength': BOND_LENGTH,
                'bondWiggleDistance': WIGGLE_DISTANCE
            },
            angle_force_func=None,
            nonbonded_force_func=polychrom.forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                'trunc': rep_e
            },
            except_bonds=gap_p + 1,
            extra_bonds=cen_mt_bonds,
            override_checks=True
        )
    )

    # angle forces for microtubules stiffness
    sim.add_force(
        polychrom.forcekits.polymer_chains(
            sim,
            chains=mt_chains,
            bond_force_func=None,
            angle_force_func=polychrom.forces.angle_force,
            angle_force_kwargs={
                'k': 10
            },
            nonbonded_force_func=None,
            override_checks=True
        )
    )

    # confine microtubule starts to nuclear envelope
    select_finite = lambda arr: arr[np.isfinite(arr)]
    sim.add_force(
        polychrom.forces.spherical_confinement(
            sim,
            r=CEN_TETHER_R_MIN,
            particles=select_finite(mts['start'].values),
            invert=True,
            k=5.0,  
            center=[0, 0, -NUCLEUS_R],
            name='spherical_exclusion_microtubules',
        )
    )

    sim.add_force(
        polychrom.forces.spherical_confinement(
            sim,
            r=CEN_TETHER_R_MAX,
            particles=select_finite(mts['start'].values),
            invert=False,
            k=5.0,  
            center=[0, 0, -NUCLEUS_R],
            name='spherical_confinement_microtubules',
        )
    )

    sim.add_force(
        polychrom.forces.spherical_confinement(
            sim,
            r=NUCLEUS_R,
            k=5.0,  # How steep the walls are
            center=[0,0,0],
            name='spherical_confinement_nucleus',
        )
    )

    left_telomeres = chroms['start']
    right_telomeres = chroms['end'] - 1
    all_telomeres = np.r_[left_telomeres, right_telomeres]

    sim.add_force(
        polychrom.forces.spherical_confinement(
            sim,
            r=NUCLEUS_R - 1,
            particles=all_telomeres,
            invert=True,
            k=5.0,  
            center=[0, 0, 0],
            name='telomere_lamina_attraction',
        )
    )

    polychrom.starting_conformations.create_spiral

    # choose random microtubule end points
    init_mts = []
    for _ in range(len(mts)):
        x = y = z = np.nan
        while ~(np.linalg.norm([x, y, z]) < NUCLEUS_R):
            theta, u = polychrom.starting_conformations._random_points_sphere(1).T
            x = (mt_len / 2) * np.sqrt(1.0 - u * u) * np.cos(theta)
            y = (mt_len / 2) * np.sqrt(1.0 - u * u) * np.sin(theta)
            z = (mt_len / 2) * u - NUCLEUS_R
        
        # create microtubule positions, exclude first coordinate to avoid microtubules having same positions
        init_mts.append(np.linspace([0, 0, -NUCLEUS_R], [*x, *y, *z], num=mt_len + 1)[1:])

    init_chroms = []
    telomere_positions = []

    constraint = lambda p: np.linalg.norm(p) < NUCLEUS_R
    step_size = BOND_LENGTH # half bond length, double number of particles for gap particles

    if not int_region:
        # intertwine entire chain
        int_region = chain_length
        int_len = int_region // 2
    else:
        int_len = (int_region * (gap_p + 1)) // 2 # scale intertwined region with extra chain length from gap particles

    for idx, chrom in chroms.iterrows():
        # intertwined arms from centromere to int_region
        cen_region = polychrom.starting_conformations.create_constrained_random_walk(
            int_len,
            constraint_f=constraint,
            starting_point=init_mts[idx][-1], # end of first attached microtubule is starting point for centromere random walk
            step_size=step_size,
            segment_size=round((chain_length // (gap_p + 1)) * 0.01)
        )

        left_intertwined, right_intertwined = wiggin.starting_mitotic_conformations.make_catenated_pair(
            core_conformation=cen_region,
            linking_number=int_n,
            radius=int_r
        )

        if int_region < chain_length:
            # non-intertwined rest of arms
            left_not_intertwined = polychrom.starting_conformations.create_constrained_random_walk(
                chain_length // 2 - int_len,
                constraint_f=constraint,
                starting_point=left_intertwined[0],
                step_size=step_size
            )
            right_not_intertwined = polychrom.starting_conformations.create_constrained_random_walk(
                chain_length // 2 - int_len,
                constraint_f=constraint,
                starting_point=right_intertwined[0],
                step_size=step_size
            )

            init_chroms.append(np.vstack((
                left_not_intertwined,
                left_intertwined[::-1],
                right_intertwined,
                right_not_intertwined
            )))
        else:
            init_chroms.append(np.vstack((
                left_intertwined[::-1],
                right_intertwined
            )))

        if fix_tel:
            # select random points on the nucleus to tether telomeres to
            theta, u = polychrom.starting_conformations._random_points_sphere(2).T
            x = NUCLEUS_R * np.sqrt(1.0 - u * u) * np.cos(theta)
            y = NUCLEUS_R * np.sqrt(1.0 - u * u) * np.sin(theta)
            z = NUCLEUS_R * u
            telomere_positions.append((x[0], y[0], z[0]))
            telomere_positions.append((x[1], y[1], z[1]))

    d0 = np.vstack(init_chroms + init_mts)

    sim.set_data(d0, center=False)  # loads a polymer, puts a center of mass at zero

    if fix_tel:
        # fix telomeres to prevent intertwines unravelling
        sim.add_force(
            polychrom.forces.tether_particles(
                sim,
                particles=all_telomeres,
                positions=telomere_positions,
                name='telomere_fixing'
            )
        )
        
    # fix microtubule starts to prevent CENs from spinning
    sim.add_force(
        polychrom.forces.tether_particles(
            sim, 
            particles=list(mts['start']),
            positions=[coords[0] for coords in init_mts],
            name='microtubule_fixing',
        )
    )

    sim.local_energy_minimization(
        maxIterations=1000,
        tolerance=1,
        random_offset=0.1
    )

    for _ in range(timesteps // 10_000):
        sim.do_block(1)
        
    sim.print_stats()

    reporter.dump_data()

if __name__ == '__main__':
    fire.Fire(run_simulation)