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
save_dir = "delay_learning_400_action_20/"

# to ensure 200 total training episodes, we need to pair the actions with the total_timesteps
# action=20, total_timesteps=4000
# action=40, total_timesteps=8000
# action=60, total_timesteps=12000

for id in range(1, 11):
    # setup environment and model
    env = gym.make('AndesFreqControl-v0')
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
    model = DDPG(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, learning_starts=400)  #

    # start training
    time_start = time.time()
    # This total_timesteps is not the total training episode but the numbers of actions applied
    # hence, we need to change the total steps with action numbers
    model.learn(total_timesteps=4000)
    print("training {} completed using {}".format(id, time.time() - time_start))

    # save model
    model.save(save_dir + "andes_secfreq_ddpg_fix_{}.pkl".format(id))

    # save data
    freq = pd.DataFrame(env.final_freq)
    freq.to_csv(save_dir + "andes_secfreq_ddpg_fix_{}.csv".format(id), index=False)

    # plot the results
    if plot_episode == True:
        plt.rcParams.update({'font.family': 'Arial'})
        plt.figure(figsize=(9, 7))
        plt.plot(env.final_freq, color='blue', alpha=1, linewidth=2)
        plt.xlabel("Episode", fontsize=20)
        plt.ylabel("Frequency (Hz)", fontsize=20)
        plt.grid()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title("Restored frequency under fixed disturbance via DRL secondary control", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir + "andes_secfreq_ddpg_fix_{}.png".format(id))

