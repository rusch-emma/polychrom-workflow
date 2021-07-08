import itertools
import multiprocessing as mp
import os
import sys
import fire

import cooltools.lib.plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import EngFormatter
from numpy.core.shape_base import block
from polychrom.contactmaps import binnedContactMap
from polychrom.hdf5_format import list_URIs, load_URI


def get_chains(start, length, n=1, double=False):
    if double:
        length *= 2
    edges = list(range(start, start + length * (n + 1), length))
    chains = pd.DataFrame(list(zip(edges[:-1], edges[1:])), columns=['start_p', 'end_p'])
    chains['length'] = chains['end_p'] - chains['start_p']

    return chains

def create_map(in_dirs, out_dir, chain_length, mt_len, int_region, int_r, int_n, fix_tel, rep_e, timesteps, gap_p, assembly, bin_size, blocks):
    """
    Creates a binned contact map from a polychrom simulation output folder and stores it as a binary .npy file.
    """

    exclude_particles = mt_len * 2
    number_particles = chain_length * (gap_p + 1)

    sims = list(itertools.chain.from_iterable([list_URIs(d)[-200:] for d in in_dirs.split(' ')]))

    print(f'Creating contact map for {os.path.basename(in_dirs)} ...')

    contigs = get_chains(0, number_particles)

    chains = list(zip(contigs['start_p'], contigs['end_p'], [False] * len(contigs)))

    hmap, starts = binnedContactMap(filenames=sims, chains=chains, binSize=bin_size, n=mp.cpu_count(), loadFunction=lambda x : load_URI(x)['pos'][:-exclude_particles])

    np.save(out_dir, hmap)


if __name__ == '__main__':
    fire.Fire(create_map)
