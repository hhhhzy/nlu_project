# nlu_project

## Usuage

### Run jupyter notebook with the singularity environment
For first time usage:

1. Clone this repo to greene scratch.

2. In the main directory of nlu_project, run `bash ./scripts/create_base_overlay.sh` and `bash ./scripts/create_package_overlay.sh`.

Start from here if you have done step 1 and 2 before:

3. Go to https://ood-3.hpc.nyu.edu/ -> Interactive Apps -> Jupyter Notebook

4. Follow the instructions under 'How to use your singularity+conda environment in jupyterhub'. The python wrapper for step 2 should look like this:

```
singularity exec $nv \
  --overlay /scratch/zh2095/nlu_project/overlay-base.ext3:ro \
  --overlay /scratch/zh2095/nlu_project/overlay-packages.ext3:ro \
  /scratch/wz2247/singularity/images/pytorch_21.06-py3.sif \
  /bin/bash -c "source ~/.bashrc; conda activate /ext3/conda/bootcamp; $cmd $args"
```
