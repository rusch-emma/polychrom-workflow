import os
import cooltools.lib.plotting # includes 'fall' cmap
import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_map(in_file, out_file, name):
    hmap = np.load(in_file)

    fig, ax = plt.subplots(figsize=(15, 15))

    img = ax.matshow(hmap, norm=LogNorm(vmin=1, vmax=hmap.max()), cmap='fall')

    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label='counts (log)')
    plt.title(name + ': ' + os.path.basename(os.path.splitext(out_file)[0]))
    plt.tight_layout()

    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)


if __name__ == '__main__':
    fire.Fire(plot_map)
