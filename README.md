# andes_gym


Andes environment for OpenAI Gym

* Free software: GPL v3+


## Installation
Create an empty conda environment with
```bash
conda create -n drl-andes python=3.8
```

### Install ANDES
The detailed instruction of Andes installation can be found [here](https://andes.readthedocs.io/en/stable/install.html).
```bash
conda config --add channels conda-forge
conda config --set channel_priority flexible
```
Use `conda install` instead of `pip install` for `andes` installation
```bash
conda install andes
```
or in the root directory of `andes`, run
```bash
conda install --file requirements.txt
conda install --file requirements-dev.txt
python3 -m pip install -e .
```
Test if `andes` has been successfully installed by
```bash
andes selftest
```

### Install GYM
Install `mpi4py` from Intel with

```bash
conda install -c intel mpi4py
```

Install `gym`
```bash
pip install gym
```

Install a recent version of ANDES with

### Stable-baselines supporting `tensorflow`

Install `tensorflow (cpu version)`
```bash
pip install andes==1.5.8
```

Install `keras`
```bash
pip install keras
```

Install `stable-baselines` or `stable-baselines3`
Refer to the [stable-baselines](https://github.com/hill-a/stable-baselines) website or [stable-baselines3](https://github.com/whoiszyc/stable-baselines3) for detailed instruction.

For `stable-baselines`

Option 1:
First `Homebrew` will be needed. Then, install the dependencies of `stable-baselines` as
```bash
brew install cmake openmpi
```
Then, open the `stable-baselines` repo, cd to the directory and 
```bash
pip install -e .
```
Option 2:
```bash
pip install stable-baselines
```

### Stable-baselines supporting `pytorch`
For `stable-baselines3` （uses Pytorch 1.7）
```bash
pip install stable-baselines3
```
or in the root directory
```bash
pip install -e .
```

### Install `andes_gym`
In the root directory of `andes_gym` , install `andes_gym` in the development mode with
```bash
pip install -e .
```

Examples
--------

Try out examples in the folder `examples` .

Edit the number of time steps in `train_freq_ddpg.py` to a proper value. The larger the value, the longer it takes to simulate and train.

Run the example with

```bash
python train_freq_ddpg.py
```

Some deprecation warnings can be safely ignored.

When the training is completed, replay the trained model with

```bash
python play_freq_ddpg.py
```

You may need to install additional packages, such as matplotlib, to show the visualization.
The matplotlib window might appear frozen, but is actually refreshed after each run.

Performance Tips
----------------

While most of the computation time was spent in ANDES, tests show that Linux perform consistently better than Windows, especially with `cvxoptklu` installed.

Currently, it is difficult to install `cvxoptklu` on Windows. Therefore, a Linux box is recommended whenever possible.

Some algorithms can take advantage of multi-core processors but most cannot. Please check the algorithm documentation from stable-baselines to verify. A quick way to check if an algorithm is taking advantage of your multi-core processor is to check the utilization of CPU (in Windows Task Manager or in `htop` of Linux).

Version Control
---------------

When working on the source code, please branch from `master` and work on your own branch.

You can either use GitHub for Desktop or learn the commands. The following are some commands for quick reference.

Branching can be done the collowing command

```bash
git checkout -b YOUR_BRANCH_NAME
```

where `YOUR_BRANCH_NAME` is the branch name of your choice.

To stage changes, use

```bash
git add PATH_TO_FILE
```

To commit changes, use

```bash
git commit
```

To push to a not-yet-exising branch, use

```bash
git push -u origin YOUR_BRANCH_NAME
```

After the first push, your local git will memorize the tracking branch. Next time, you can simply push with

```bash
git push
```

Features
--------

* TODO
