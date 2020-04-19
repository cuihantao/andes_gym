import gym
import andes_gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C
import multiprocessing


def main():
    n_cpu = multiprocessing.cpu_count();
    env = SubprocVecEnv([lambda: gym.make('AndesFreqControl-v0') for i in range(n_cpu)])
    model = A2C.load("andes_freq_a2c.pkl")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # Note: VecEnv calls reset automatically
        if int(dones[0]) == 1:
            env.render()


if __name__ == '__main__':
    main()
