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
model.learn(total_timesteps=2000)  # we need to change the total steps with action numbers
print("training {} completed using {}".format(id, time.time() - time_start))

obs = env.reset()
done = False
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done is True:
        break

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
# plt.savefig(save_dir + "andes_secfreq_ddpg_fix_{}.png".format(id))

# plot the frequency
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(left=0, right=np.max(env.t_render))
ax.set_ylim(auto=True)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.set_xlabel("Time [s]", fontsize=16)
ax.set_ylabel("Bus Frequency [Hz]", fontsize=16)
ax.ticklabel_format(useOffset=False)
for i in range(env.N_Bus):
    ax.plot(env.t_render, env.final_obs_render[:, i] * 60)
plt.savefig("fig_freq_dynamics.pdf")

