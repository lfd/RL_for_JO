# Quantum Reinforcement Learning for Join Ordering

This repository provides the implementation and artifacts accompanying the following article:
```
@article{franz:23:qrl_jo,
    author = {Maja Franz AND
              Tobias Winker AND
              Sven Groppe AND
              Wolfgang Mauerer},
    title  = {Hype or Heuristic? Quantum Reinforcement Learning for Join Order Optimisation},
    note   = {under review},
}
```

## Setup

### Python packages
The code can be executed with Python version 3.9.
For managing multiple python versions it is recommended to use [pyenv](https://github.com/pyenv/pyenv).
After installing Python 3.9 using pyenv with
```
pyenv install 3.9
```
It can be used in the local directory:
```
pyenv local 3.9
```
Setting up a new virtual environment using
```
python -m venv .venv
source .venv/bin/activate
```
all required python packages can be installed with:
```
pip install -r requirements.txt
```

### PostgreSQL and the Join Order Benchmark

For the setup of PostgreSQL-V16.0 and the Join Order Benchmark, we provide two setup scripts:
```
source install_postgres.sh
source setup_JOB.sh
```

## Training

``python main.py <configuration_name>``

e.g. ``python main.py example``

Available configurations can be found in ``configs/``

A training process can be tracked with TensorBoard:

``tensorboard --logdir logs``

## Additional Material and Hyperparameter Search

Information on the classical baseline replication and on specific hyperparameters can be found in [`info/supplementary.pdf`](info/supplementary.pdf)

## Result Sets

All result sets from the paper are stored as CSV files in logs.
The tikz plots from the paper can be reproduced with the [`plots/plot.r`](plots/plot.r) script using
```
Rscript plots/plot.r
```
