# polychrom-workflow
A Snakemake workflow for running and analysing parameter sweeps of polymer simulations using [Polychrom](https://github.com/open2c/polychrom).

## Configuration

### Workflow Configuration

Configure the workflow by modifying `config.yaml`. An example template is provided in `configs/config.yaml`.

The following entries are expected in `config.yaml`:

* `logs_dir`: Directory for storing log files.
* `simulation`: Contains simulation configuration.
  * `script`: Python script for running the Polychrom simulation. See the [Polychrom documentation](https://polychrom.readthedocs.io/en/latest/) for how to set up simulations. Scripts should take all arguments provided by the `parameter` directives in `config.yaml` as CLI arguments. This can be automated, e.g. see [Python Fire](https://github.com/google/python-fire) and `examples` for an example use case.
  * `out_dir`: Directory for storing the simulation result files. Results will be stored here in directories with names generated from a combination of used parameter values and 
  * `replicates`: The number of replicate simulations per parameter combination to run.
  * `parameters`: A list of parameters with a list of values each. Simulations will be run for each combination of parameter values.

### Adding additional rules

You can add additional rules, e.g. for analysing simulation results, by adding a new rule in `workflow/Snakefile` and providing it with the correct input files/directories (see [Snakemake documentation: `expand`](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#the-expand-function) for how to expand patterns into lists of files) and a pattern for the resulting output files/directories. Then add `shell` to call your script with the necessary parameters which you should add in `config.yaml`. Input and output wildcards can be accessed using `{input}` and `{output}` respectively in strings. (Note: in Python f-string braces have to be escaped by using double braces, so `{input}` becomes `{{input}}` for example.)

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

## Output

### Results

Results are configured to be stored in `workflow/results` with subdirectories for simulation results as well as analyses/post-processing of simulation results.

### Logs

Log and error files will be stored under `workflow/logs` with subdirectories for each rule that produced the corresponding log files.

Note: Output directories are configured in `config.yaml` and can be changed there.

## Examples

`examples/` contains examples on how to use this workflow.

* `examples/sim_hic`: Example workflow performing a parameter sweep running a Polychrom simulation and then creating contact maps from the simulation results.
