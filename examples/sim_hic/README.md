# polychrom-workflow example: Simulation + Contact Maps

This example workflow performs a parameter sweep running the Polychrom simulation defined in `scripts/simulation.py` and then creates contact maps from the simulation results using `scripts/create_maps.py`.

## Usage

```
snakemake --profile profiles/slurm
```