# polychrom-workflow
A Snakemake workflow for running and analysing parameter sweeps of polymer simulations using [Polychrom](https://github.com/open2c/polychrom).

## Configuration

### Workflow Configuration

Configure the workflow by modifying `config.yaml`. An example template is provided in `configs/config.yaml`.

The following entries are expected in `config.yaml`:

* `logs`: Directory for storing log files.
* `sim_script`: Python script for running the Polychrom simulation. See the [Polychrom documentation](https://polychrom.readthedocs.io/en/latest/) for how to set up simulations.
* `sims_dir`: Directory for storing the simulation result files. Results will be stored here in directories with names generated from a combination of used parameter values and 
* `replicates`: The number of replicate simulations per parameter combination to run.
* `params`: A list of parameters with a list of values each. Simulations will be run for each combination of parameter values.

### Cluster Configuration

A profile for the Slurm cluster management software is provided in `profiles/slurm`. Add additional profiles here for other cluster management systems.

* `config.yaml`: Adjust job related settings, e.g. global resource limits, job submissions settings or job-restart settings.
* `cluster_config.yaml`: Adjust rule specific settings, e.g. resources, output/error paths or walltime. For example the `run_simulation` rule has been configured to use GPU nodes.


### Conda

By default the workflow will use the currently activated Conda environment and its installed packages. Alternatively you can use the `--use-conda` flag to create and use a dedicated environment which will be installed in the workflow directory. This will use `environment.yaml` to download and install all required dependencies.

## Usage

Run the workflow with a specific profile with `Snakefile` in the current directory:

```
snakemake --profile profiles/slurm
```

You can also first test the workflow by performing a dry-run with the `-n` flag:

```
snakemake -n --profile profiles/slurm
```
