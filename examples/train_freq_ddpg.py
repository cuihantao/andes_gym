import gym
import numpy as np
import andes_gym
import os

# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
# That is dangerous, since it can degrade performance or cause incorrect results.
# The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library.
# As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results.
# For more information, please see http://www.intel.com/software/products/support/.
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG

env = gym.make('AndesFreqControl-v0')
# env = DummyVecEnv([lambda: env])
#
# # the noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# param_noise = None

model = DDPG(MlpPolicy, env, verbose=1, learning_starts=10)
model.learn(total_timesteps=2000)
print("training completed")
model.save("andes_freq_ddpg.pkl")
