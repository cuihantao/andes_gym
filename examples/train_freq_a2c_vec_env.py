import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C

env = make_vec_env('AndesFreqControl-v0', n_envs=4)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)
model.save("andes_freq_a2c.pkl")

