import gym
import andes_gym
from stable_baselines import deepq, PPO1
from stable_baselines.common.policies import MlpPolicy


def main():
    env = gym.make("AndesFreqControl-v0")
    env.path = "../envs/CL_2machine_mpc.dm"
    model = PPO1(MlpPolicy, env, verbose=1)
    model.load(load_path="andes_freq_model.pkl")

    obs = env.reset()
    done = False

    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if done is True:
            env.render()
            obs = env.reset()
            done = False


if __name__ == '__main__':
    main()
