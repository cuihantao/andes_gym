import gym
import pandas as pd
import numpy as np
import andes_gym
import os
import matplotlib.pyplot as plt
import time
import torch
# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
# That is dangerous, since it can degrade performance or cause incorrect results.
# The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library.
# As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results.
# For more information, please see http://www.intel.com/software/products/support/.
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3 import DDPG

plot_episode = True
save_dir = "delay_learning_200_action_40/"


# setup environment and model
env = gym.make('AndesFreqControl-v0')
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
model = DDPG(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, learning_starts=200)  #

# start training
time_start = time.time()
model.learn(total_timesteps=4000)  # we need to change the total steps with action numbers
print("training {} completed using {}".format(id, time.time() - time_start))

obs = env.reset()
done = False
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done is True:
        env.render()

