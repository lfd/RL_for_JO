# Quantum Reinforcement Learning for Join Ordering

This repository provides the implementation and artifacts accompanying the following article:
```
@inproceedings{franz:24:qce24,
 title     = {Hype or Heuristic? Quantum Reinforcement Learning for Join Order Optimisation},
 author    = {Maja Franz and Tobias Winker and Sven Groppe and Wolfgang Mauerer},
 booktitle = {IEEE International Conference on Quantum Computing and Engineering (QCE)},
 year      = {2024},
 month     = {09},
 userd     = {IEEE QCE '24},
}
```
A preprint is available on [arXiv](https://arxiv.org/abs/2405.07770).

## Content of this Repository

- [`experimental_analysis/`](experimental_analysis/):
This directory contains all necessary scripts to run a training and evaluation of the classical and multi-step QRL approaches presented in the publication.
For running experiments with the single-step QML approach, we refer to the [source code](https://github.com/TobiasWinker/QC4DB_QO) accompanying the [related publication](https://doi.org/10.1145/3579142.3594299).
All result sets from the paper are stored as CSV files in [`experimental_analysis/logs/paper_results`](experimental_analysis/logs/paper_results).
The tikz plots from the paper can be reproduced with the script [`scripts/generate_plots.sh`](scripts/generate_plots.sh).

- [`info/`](info/):
This directory contains supplementary material on the classical baseline replication and on specific hyperparameters.

- [`paper/`](paper/):
This directory contains the article in PDF and its source code in LaTeX.

- [`plots/`](plots/):
This directory contains the plots from the paper and the source code to generate them in R.

- [`scalability/`](scalability/):
In this directory the CSV files, which are used to generate the scalability Figs. 9 and 10, are stored, together with the python scripts, which were used to generate those.

- [`scripts/`](scripts/):
This directory contains bash scripts, which can either be used as endpoints for the Docker-image, or executed locally.
Furthermore the setup scripts for PostgreSQL-V16 and the JOB can be found there.


## Setup

### Docker

#### Get docker image
Build image:

```docker build -t qce24_repro .```

or pull image:

```docker pull ghcr.io/lfd/rl_for_jo/qce24_repro:latest```

The image does contain an instance of PostgreSQL-V16.
However the dataset for the Join order benchmark is only installed, once a container is run with the corresponding options for the endpoint.

#### Create Container

```docker run --name qce24_repro -it qce24_repro [<-flags>] [<option>]```

The `<option>` specifies which operations are performed on container start.

Available options are:
* `experiments_classic`: performs the trainings with a classical NN\*
* `experiments_quantum`: performs the trainings involving a VQC\*
* `experiments_noise`: performs a evaluation in simulated noisy environments\*
* `plot`: generates the the plots for the paper using R
* `paper`: generates the full paper from LaTeX
* `plot_paper`: generates both, plots and the paper
* `all`: performs all of the above\*
* `bash`(default): does not perform any operation, but launches interactive shell, default

Feel free to define additional `<-flags>`, e.g.:
* Volume, to keep track of generated files on the host system: `-v $PWD:/home/repro/qce24_repro`
* Port forwarding to launch TensorBoard on the container to track the training process for RL on the host: `-p 6006:6006`. TensorBoard can be started in the Container with: `tensorboard --logdir experimental_analysis/logs --host 0.0.0.0`

\*Please note the long runtimes for RL trainings (hours to days).
Additionally, the Join Order Benchmark gets set up for these options, which additionally takes up to 1-2 hours.
For quickly inspecting our reproduction package, we recommend to use the option `bash`.

### Local Setup

#### Python packages
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

#### PostgreSQL and the Join Order Benchmark

For the setup of PostgreSQL-V16.0 and the Join Order Benchmark, we provide two setup scripts:
```
source scripts/install_postgres.sh
source scripts/setup_JOB.sh
```

The `scripts/setup_JOB.sh` script, and the Docker setup make use of the scripts from this [GitHub repository](https://github.com/danolivo/jo-bench) to set up the JOB.

## Custom Trainings

``python main.py <configuration_name>``

e.g. ``python main.py example``

Available configurations can be found in ``configs/``

A training process can be tracked with TensorBoard:

``tensorboard --logdir logs``

