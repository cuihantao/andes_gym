===============================
andes_gym
===============================

.. image:: https://img.shields.io/travis/cuihantao/andes_gym.svg
        :target: https://travis-ci.org/cuihantao/andes_gym

.. image:: https://img.shields.io/pypi/v/andes_gym.svg
        :target: https://pypi.python.org/pypi/andes_gym


Andes environment for OpenAI Gym

* Free software: GPL v3+
 
Installation
------------

Create an empty conda environment with

```
git clone https://github.com/cuihantao/andes_gym
```


Clone the repository with

```
conda create -n andes_gym --yes
```

Install the previous version of ANDES with 

```
pip install andes==0.6.9
```

Install `mpi4py` from Intel with

```
conda install -c intel mpi4py
```

In the root directory of `andes_gym`, install `andes_gym` in the development mode with

```
pip install -e .
```

Examples
--------
Try out examples in the folder `examples`.

Edit the number of time steps in `train_freq_ddpg.py` to a proper value. The larger the value, the longer it takes to simulate and train.

Run the example with 

```
python train_freq_ddpg.py
```
Some deprecation warnings can be safely ignored.

When the training is completed, replay the trained model with 

```
python play_freq_ddpg.py
```

You may need to install additional packages, such as matplotlib, to show the visualization.


Performance Tips
----------------

While most of the computation time was spent in ANDES, tests show that Linux perform consistently better than Windows, especially with `cvxoptklu` installed.

Currently, it is difficult to install `cvxoptklu` on Windows. Therefore, a Linux box is recommended whenever possible.

Some algorithms can take advantage of multi-core processors but most cannot. Please check the algorithm documentation from stable-baselines to verify. A quick way to check if an algorithm is taking advantage of your multi-core processor is to check the utilization of CPU (in Windows Task Manager or in `htop` of Linux).


Features
--------

* TODO
