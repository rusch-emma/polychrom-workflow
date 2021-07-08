import itertools
import multiprocessing as mp
import os
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


def plot_map(hmap, bin_size, out_path, contigs, title, plot_labels):
    fig, ax = plt.subplots(
        figsize=(14, 10),
        ncols=1
    )
    bp_formatter = EngFormatter('b')
    norm = LogNorm(vmin=1, vmax=hmap.max())

    img = ax.matshow(hmap, norm=norm, cmap='fall')

    if plot_labels:
        chrom_positions = contigs['start_p'] // bin_size
        chrom_labels = contigs['chrom']
        ax.set_xticks(chrom_positions)
        ax.set_yticks(chrom_positions)
        ax.set_xticklabels(chrom_labels)
        ax.set_yticklabels(chrom_labels)
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.grid(ls='dashed', lw=0.25, alpha=0.75)

    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label='counts (log)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)


def create_map(in_dirs, out_dir, name, chain_length, mt_len, int_region, int_r, int_n, fix_tel, rep_e, timesteps, gap_p, assembly, bin_size, blocks):
    """
    Creates and plots a binned contact map from a polychrom simulation output folder.
    """

    exclude_particles = mt_len * 2
    number_particles = chain_length * (gap_p + 1)
    title = f"{name}: rep_e={rep_e}, n_chains=1, n_particles={number_particles}, mt_len={mt_len}, int_n={int_n}, int_r={int_r}, int_region={int_region}, gap_p={gap_p}, fix_tel={fix_tel}, blocks={blocks}, bin_size={bin_size}"

    dirs = [os.path.join(in_dirs, subdir) for subdir in os.listdir(in_dirs) if os.path.isdir(os.path.join(in_dirs, subdir))]
    if not dirs:
        dirs.append(in_dirs)
    sims = []
    for d in dirs:
        try:
            sims.append(list_URIs(d)[-blocks:])
        except:
            # skip directories without data
            pass
    sims = list(itertools.chain.from_iterable(sims))

    print(f'Creating contact map for {os.path.basename(in_dirs)} ...')

    contigs = get_chains(0, number_particles)

    chains = list(zip(contigs['start_p'], contigs['end_p'], [False] * len(contigs)))

    hmap, starts = binnedContactMap(filenames=sims, chains=chains, binSize=bin_size, n=mp.cpu_count(), loadFunction=lambda x : load_URI(x)['pos'][:-exclude_particles])

    np.save(out_dir, hmap)
    #plot_map(hmap, bin_size, plots_dir + '.png', contigs, title, True)


if __name__ == '__main__':
    fire.Fire(create_map)
