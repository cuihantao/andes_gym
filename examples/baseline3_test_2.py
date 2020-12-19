import gym
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

env = gym.make('CartPole-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100, log_interval=4)
model.save("CartPole_dqn.pkl")

del model # remove to demonstrate saving and loading

model = DQN.load("CartPole_dqn.pkl")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()