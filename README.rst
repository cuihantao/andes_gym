andes_gym
=========

Andes environment for OpenAI Gym

-  Free software: GPL v3+

Installation
------------

Create an empty conda environment with

.. code:: bash

   git clone https://github.com/cuihantao/andes_gym

Clone the repository with

.. code:: bash

   conda create -n andes_gym --yes

Install the previous version of ANDES with

.. code:: bash

   pip install andes==0.6.9

Install ``mpi4py`` from Intel with

.. code:: bash

   conda install -c intel mpi4py

In the root directory of ``andes_gym`` , install ``andes_gym`` in the
development mode with

.. code:: bash

   pip install -e .

Examples
--------

Try out examples in the folder ``examples`` .

Edit the number of time steps in ``train_freq_ddpg.py`` to a proper
value. The larger the value, the longer it takes to simulate and train.

Run the example with

.. code:: bash

   python train_freq_ddpg.py

Some deprecation warnings can be safely ignored.

When the training is completed, replay the trained model with

.. code:: bash

   python play_freq_ddpg.py

You may need to install additional packages, such as matplotlib, to show
the visualization. The matplotlib window might appear frozen, but is
actually refreshed after each run.

Performance Tips
----------------

While most of the computation time was spent in ANDES, tests show that
Linux perform consistently better than Windows, especially with
``cvxoptklu`` installed.

Currently, it is difficult to install ``cvxoptklu`` on Windows.
Therefore, a Linux box is recommended whenever possible.

Some algorithms can take advantage of multi-core processors but most
cannot. Please check the algorithm documentation from stable-baselines
to verify. A quick way to check if an algorithm is taking advantage of
your multi-core processor is to check the utilization of CPU (in Windows
Task Manager or in ``htop`` of Linux).

Version Control
---------------

When working on the source code, please branch from ``master`` and work
on your own branch.

You can either use GitHub for Desktop or learn the commands. The
following are some commands for quick reference.

Branching can be done the collowing command

.. code:: bash

   git checkout -b YOUR_BRANCH_NAME

where ``YOUR_BRANCH_NAME`` is the branch name of your choice.

To stage changes, use

.. code:: bash

   git add PATH_TO_FILE

To commit changes, use

.. code:: bash

   git commit

To push to a not-yet-exising branch, use

.. code:: bash

   git push -u origin YOUR_BRANCH_NAME

After the first push, your local git will memorize the tracking branch.
Next time, you can simply push with

.. code:: bash

   git push

Features
--------

-  TODO
