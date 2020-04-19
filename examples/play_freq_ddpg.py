import gym
import andes_gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG


def main():
    env = gym.make('AndesFreqControl-v0')
    env = DummyVecEnv([lambda: env])

    model = DDPG.load("andes_freq_ddpg.pkl")

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if int(dones[0]) == 1:
            env.render()


if __name__ == '__main__':
    main()
