import gym
import time
import torch
import matplotlib.pyplot as plt
import andes_gym
from stable_baselines3 import A2C

# setup environment and model
env = gym.make('AndesFreqControl-v0')
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
model = A2C("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

# start training
time_start = time.time()
model.learn(total_timesteps=2000)
print("training completed using {}".format(time.time() - time_start))

# plot the results
plt.rcParams.update({'font.family': 'Arial'})
plt.figure(figsize=(9, 7))
plt.plot(env.final_freq, color='blue', alpha=1, linewidth=2)
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Frequency (Hz)", fontsize=20)
plt.grid()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Restored frequency under fixed disturbance via DRL secondary control", fontsize=16)
plt.show()
plt.tight_layout()
plt.savefig("restored_frequency_fix.png")
