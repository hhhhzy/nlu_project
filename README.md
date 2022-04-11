# nlu_project

## Usuage

### Run jupyter notebook in the singularity container
For first time usage:

1. Clone this repo to greene scratch.

2. In the main directory of nlu_project, run `bash ./scripts/create_base_overlay.sh` and `bash ./scripts/create_package_overlay.sh`.

Start from here if you have done step 1 and 2 before:

3. Run `sbatch ./scripts/run_jupyter.sbatch`.

4. Follow the instructions printed on the slurm output.