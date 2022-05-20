# nlu_project
The paper is [here](./NLU_paper_Group18.pdf).

## Usuage

### Run jupyter notebook with the singularity environment
For first time usage, create the two overlay files that contains all the packages needed:

1. Clone this repo to greene scratch.

2. In the main directory of nlu_project, run `bash ./scripts/create_base_overlay.sh` and `bash ./scripts/create_package_overlay.sh`.

If you already have the overlay files:

3. Connect to NYU vpn.

4. Go to https://ood-3.hpc.nyu.edu/ -> Interactive Apps -> Jupyter Notebook

5. Follow the instructions under 'How to use your singularity+conda environment in jupyterhub'. The python wrapper for step 2 should look like this:

```
singularity exec $nv \
  --overlay /scratch/zh2095/nlu_project/overlay-base.ext3:ro \
  --overlay /scratch/zh2095/nlu_project/overlay-packages.ext3:ro \
  /scratch/wz2247/singularity/images/pytorch_21.06-py3.sif \
  /bin/bash -c "source ~/.bashrc; conda activate /ext3/conda/bootcamp; $cmd $args"
```
6. Launch the notebook, and select the kernel 'my_env'.
