import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C
import multiprocessing

n_cpu = multiprocessing.cpu_count()

env = SubprocVecEnv([lambda: gym.make('AndesFreqControl-v0') for i in range(n_cpu)])

model = A2C(MlpLstmPolicy, env, verbose=1)
model.learn(total_timesteps=100)
model.save("andes_freq_a2c.pkl")

