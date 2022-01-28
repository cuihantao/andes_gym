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

setting_label = [
    "Delay 0", "Delay 100", "Delay 200", "Delay 300", "Delay 400", "Delay 500"
]
# hyper-parameter set 1
setting_1 = pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_1.csv").T
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_2.csv").T, ignore_index=True)
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_3.csv").T, ignore_index=True)
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_4.csv").T, ignore_index=True)
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_5.csv").T, ignore_index=True)
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_6.csv").T, ignore_index=True)
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_7.csv").T, ignore_index=True)
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_8.csv").T, ignore_index=True)
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_9.csv").T, ignore_index=True)
setting_1 = setting_1.append(pd.read_csv("delay_learning_0_action_20/andes_secfreq_ddpg_fix_10.csv").T, ignore_index=True)

setting_2 = pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_1.csv").T
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_2.csv").T, ignore_index=True)
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_3.csv").T, ignore_index=True)
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_4.csv").T, ignore_index=True)
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_5.csv").T, ignore_index=True)
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_6.csv").T, ignore_index=True)
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_7.csv").T, ignore_index=True)
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_8.csv").T, ignore_index=True)
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_9.csv").T, ignore_index=True)
setting_2 = setting_2.append(pd.read_csv("delay_learning_100_action_20/andes_secfreq_ddpg_fix_10.csv").T, ignore_index=True)

setting_3 = pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_1.csv").T
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_2.csv").T, ignore_index=True)
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_3.csv").T, ignore_index=True)
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_4.csv").T, ignore_index=True)
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_5.csv").T, ignore_index=True)
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_6.csv").T, ignore_index=True)
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_7.csv").T, ignore_index=True)
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_8.csv").T, ignore_index=True)
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_9.csv").T, ignore_index=True)
setting_3 = setting_3.append(pd.read_csv("delay_learning_200_action_20/andes_secfreq_ddpg_fix_10.csv").T, ignore_index=True)

setting_4 = pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_1.csv").T
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_2.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_3.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_4.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_5.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_6.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_7.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_8.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_9.csv").T, ignore_index=True)
setting_4 = setting_4.append(pd.read_csv("delay_learning_300_action_20/andes_secfreq_ddpg_fix_10.csv").T, ignore_index=True)

setting_5 = pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_1.csv").T
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_2.csv").T, ignore_index=True)
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_3.csv").T, ignore_index=True)
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_4.csv").T, ignore_index=True)
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_5.csv").T, ignore_index=True)
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_6.csv").T, ignore_index=True)
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_7.csv").T, ignore_index=True)
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_8.csv").T, ignore_index=True)
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_9.csv").T, ignore_index=True)
setting_5 = setting_5.append(pd.read_csv("delay_learning_400_action_20/andes_secfreq_ddpg_fix_10.csv").T, ignore_index=True)

setting_6 = pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_1.csv").T
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_2.csv").T, ignore_index=True)
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_3.csv").T, ignore_index=True)
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_4.csv").T, ignore_index=True)
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_5.csv").T, ignore_index=True)
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_6.csv").T, ignore_index=True)
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_7.csv").T, ignore_index=True)
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_8.csv").T, ignore_index=True)
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_9.csv").T, ignore_index=True)
setting_6 = setting_6.append(pd.read_csv("delay_learning_500_action_20/andes_secfreq_ddpg_fix_10.csv").T, ignore_index=True)


plt.rcParams.update({'font.family': 'Arial'})
plt.figure(figsize=(12, 6))
# setting 1: delay 0
Iteration_step_1 = range(setting_1.shape[1])
plt.plot(Iteration_step_1, setting_1.mean(), label=setting_label[0], color='k', linewidth=2, alpha=1)
plt.fill_between(Iteration_step_1, setting_1.mean() - setting_1.std(), setting_1.mean() + setting_1.std(), facecolor='k', alpha=0.3)
# setting 2: delay 100
Iteration_step_2 = range(setting_2.shape[1])
plt.plot(Iteration_step_2, setting_2.mean(), label=setting_label[1], color='b', linewidth=2, alpha=1)
plt.fill_between(Iteration_step_2, setting_2.mean() - setting_2.std(), setting_2.mean() + setting_2.std(),
         color='b', alpha=0.3)
# setting 3: delay 200
Iteration_step_3 = range(setting_3.shape[1])
plt.plot(Iteration_step_3, setting_3.mean(), label=setting_label[2], color='r', linewidth=2, alpha=1)
plt.fill_between(Iteration_step_3, setting_3.mean() - setting_3.std(), setting_3.mean() + setting_3.std(),
         color='r', alpha=0.3)
# setting 4: delay 300
Iteration_step_4 = range(setting_4.shape[1])
plt.plot(Iteration_step_4, setting_4.mean(), label=setting_label[3], color='m', linewidth=2, alpha=1)
plt.fill_between(Iteration_step_4, setting_4.mean() - setting_4.std(), setting_4.mean() + setting_4.std(),
         color='m', alpha=0.3)
# setting 5: delay 400
Iteration_step_5 = range(setting_5.shape[1])
plt.plot(Iteration_step_5, setting_5.mean(), label=setting_label[4], color='y', linewidth=2, alpha=1)
plt.fill_between(Iteration_step_5, setting_5.mean() - setting_5.std(), setting_5.mean() + setting_5.std(),
         color='y', alpha=0.3)
# setting 6: delay 500
Iteration_step_6 = range(setting_6.shape[1])
plt.plot(Iteration_step_6, setting_6.mean(), label=setting_label[5], color='g', linewidth=2, alpha=1)
plt.fill_between(Iteration_step_6, setting_6.mean() - setting_6.std(), setting_6.mean() + setting_6.std(),
         color='g', alpha=0.3)
# x and y axis label
plt.xlabel('Episode Indices', fontsize=20)
plt.ylabel('Final Frequency', fontsize=20)
plt.legend(fontsize=20, loc="lower right")
plt.grid(color='0.8')
plt.xticks(fontsize=20)  # Iteration_step, range(0, 11),
plt.yticks(fontsize=20)
plt.axis([0, 200, 59.4, 60.3])
# plt.title('Active Learning for Power Flow Solvability', fontsize=16)
plt.show()
plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
plt.tight_layout()
plt.savefig("fig_secfreq_delay_1.pdf")

mean_set_1 = abs(setting_1.mean().iloc[150:200]-60).mean()
mean_set_2 = abs(setting_2.mean().iloc[150:200]-60).mean()
mean_set_3 = abs(setting_3.mean().iloc[150:200]-60).mean()
mean_set_4 = abs(setting_4.mean().iloc[150:200]-60).mean()
mean_set_5 = abs(setting_5.mean().iloc[150:200]-60).mean()
mean_set_6 = abs(setting_6.mean().iloc[150:200]-60).mean()

