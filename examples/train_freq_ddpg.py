import gym
import numpy as np
import andes_gym

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG

import multiprocessing

env = gym.make('AndesFreqControl-v0')
env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None

model = DDPG(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2000)
model.save("andes_freq_ddpg.pkl")
