import gym
import andes_gym

from stable_baselines import deepq, PPO1
from stable_baselines.common.policies import MlpPolicy


def main():
    env = gym.make("AndesFreqControl-v0")
    env.path = "../andes_gym/envs/CL_2machine_mpc.dm"

    model = PPO1(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=100)

    model.save("andes_freq_model.pkl")
    print("Saving model to andes_freq_model.pkl")


if __name__ == '__main__':
    main()
