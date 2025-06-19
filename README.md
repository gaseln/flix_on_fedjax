# Explicit Mixture of Local and Global models

This repository contains the code to run large-scale experiments presented in the paper ["FLIX: A Simple and Communication-Efficient Alternative to Local Methods in Federated Learning"](https://arxiv.org/abs/2111.11556). For small-scale experiments, look at [this repository](https://github.com/gaseln/FLIX_small_scale_experiments).

## Building the Conda environment

```bash
./bin/create-conda-env.sh
```

Once the new environment has been created, you can activate the environment with the following 
command.

```bash
conda activate $ENV_PREFIX
```
## Running experiments

Run one of the notebooks in the folder "notebooks" to run a needed experiment -- EMNIST_general.ipynb for the experiment with the EMNIST dataset, Shakespeare_general.ipynb for the experiment with the Shakespeare dataset.
