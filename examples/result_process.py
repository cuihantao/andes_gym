import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


# label
setting_label = [
    "Delay 0, total 2000, act 20",
    "Delay 0, total 2000, act 40",
    "Delay 500, total 2000, act 20",
    "Delay 500, total 4000, act 20",
]

for i in range(1, 11):
    setting_1 = pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_{}.csv".format(i)).drop(['Unnamed: 0'], axis=1)
    setting_1.to_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_{}.csv".format(i), index=False)


setting_4 = pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_1.csv").T
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_2.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_3.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_4.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_5.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_6.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_7.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_8.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_9.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_10.csv").T, ignore_index=True)

Iteration_step = range(setting_1.shape[1])

plt.rcParams.update({'font.family': 'Arial'})
plt.figure(figsize=(12, 6))
# Random
plt.plot(Iteration_step, setting_1.mean(), label=setting_label[0], color='k', linewidth=2, alpha=1)
plt.fill_between(Iteration_step, setting_1.mean() - setting_1.std(), setting_1.mean() + setting_1.std(), facecolor='k', alpha=0.3)
# LeastConfidence
plt.plot(Iteration_step, setting_2.mean(), label=setting_label[1], color='b', linewidth=2, alpha=1)
plt.fill_between(Iteration_step, setting_2.mean() - setting_2.std(), setting_2.mean() + setting_2.std(),
         color='b', alpha=0.3)
# # Margin
# plt.plot(Iteration_step, Margin.mean(), label='Margin Sampling', color='r', linewidth=2, alpha=1)
# plt.fill_between(Iteration_step, Margin.mean() - Margin.std(), Margin.mean() + Margin.std(),
#          color='r', alpha=0.3)
# # Entropy
# plt.plot(Iteration_step, Entropy.mean(), label='Entropy Sampling', color='g', linewidth=2, alpha=1)
# plt.fill_between(Iteration_step, Entropy.mean() - Entropy.std(), Entropy.mean() + Entropy.std(),
#          color='g', alpha=0.3)
# x and y axis label
plt.xlabel('Iteration Steps', fontsize=20)
plt.ylabel('Final Frequency', fontsize=20)
plt.legend(fontsize=20)
plt.grid(color='0.8')
plt.xticks(fontsize=20)  # Iteration_step, range(0, 11),
plt.yticks(fontsize=20)
# plt.axis([2, 11, 0.75, 0.95])
# plt.title('Active Learning for Power Flow Solvability', fontsize=16)
plt.show()
# plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
plt.tight_layout()
plt.savefig("fig_secfreq.pdf")
