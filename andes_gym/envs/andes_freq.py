"""
Load frequency control environment using ANDES.

This file was part of gym-power and is now part of andes_gym.

Authors:
Hantao Cui (cuihantao@gmail.com)
Yichen Zhang (whoiszyc@live.com)

Modification and redistribution of this file is subject to a collaboration agreement.
Derived source code should be made available to all authors.
"""
import os
import gym
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import andes

from gym import error, spaces, utils
from gym.utils import seeding


class AndesFreqControl(gym.Env):
    """
    Load frequency control environment using ANDES.

    This environment simulates the 2-machine, 5-bus system in ANDES
    with random load ramp disturbance. The duration of simulation is 50s.

    Observation:
        Bus Frequency
        Bus Frequency ROCOF

    Action:
        Discrete action every T seconds.
        Activation of the action will adjust the `pin` of `TG1` at action instants

    Reward:
        Based on the frequency at the action instants

    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Environment initialization
        """
        path = pathlib.Path(__file__).parent.absolute()
        self.path = os.path.join(path, "ieee14_alter_pq.xlsx")

        self.tf = 10.0     # end of simulation time
        self.tstep = 1/30  # simulation time step
        self.fixt = True   # if we do fixed step integration
        self.no_pbar = True

        # we need to let the agent to observe the disturbed trajectory before any actions taken,
        # therefore the following instant sequence is not correct: np.array([0.1, 5, 10]).
        # Instead, we will use this instant sequence: np.array([5,..., 10])
        self.action_instants = np.linspace(5, 10, 10)

        self.N = len(self.action_instants)  # number of actions
        self.N_TG = 5  # number of TG1 models
        self.N_Bus = 5  # let it be the number of generators for now

        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(self.N_TG,))
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.N_TG,))

        self.i = 0  # index of the current action

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.fig = None
        self.ax = None
        self.action_last = None

        self.t_render = None
        self.final_obs_render = None

        self.freq_print = []
        self.action_print = []
        self.reward_print = []

        # record the final frequency
        self.final_freq = []

    def seed(self, seed=None):
        """
        Generate the amount of load disturbance
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def initialize(self):
        """
        Initialize the andes simulation
        """
        self.i = 0
        self.sim_case = andes.run(self.path, no_output=True)
        self.sim_case.PQ.config.p2p = 1
        self.sim_case.PQ.config.p2z = 0
        self.sim_case.PQ.config.p2i = 0
        self.sim_case.PQ.config.q2q = 1
        self.sim_case.PQ.config.q2z = 0
        self.sim_case.PQ.config.q2i = 0
        self.sim_case.TDS.init()

        # random or fixed disturbance
        self.disturbance = 0.6  ## random.uniform(0.2, 0.5)
        # self.disturbance = random.uniform(0.2, 0.5)
        self.sim_case.Alter.amount.v[0] = self.disturbance

        # configurations
        self.sim_case.TDS.config.fixt = self.fixt

        # sensed signals
        self.w = np.array(self.sim_case.GENROU.omega.a)
        # self.dwdt = np.array(self.sim_case.BusFreq.dwdt)
        self.tg_idx = [i for i in self.sim_case.TurbineGov._idx2model.keys()]

        self.action_last = np.zeros(self.N_TG)
        # TODO: add the load disturbance model

        # Step to the first action instant
        assert self.sim_to_next(), "First simulation step failed"

        self.freq_print = []
        self.action_0_print = []
        self.action_1_print = []
        self.action_2_print = []
        self.action_3_print = []
        self.action_4_print = []
        self.reward_print = []

    def sim_to_next(self):
        """
        Simulate to the next action time instance.
        Increase the counter i each time
        """
        next_time = self.tf
        if self.i < len(self.action_instants):
            next_time = float(self.action_instants[self.i])

        self.sim_case.TDS.config.tf = next_time
        self.i += 1

        return self.sim_case.TDS.run(self.no_pbar)

    def reset(self):
        print("Env reset.")
        self.initialize()
        freq = self.sim_case.dae.x[self.w]
        self.freq_print.append(freq[0])
        return freq

    def step(self, action):
        """
        Stepping function for RL
        """
        reward = 0.0  # reward for the current step
        done = False

        # Get the next action time in the list
        if self.i >= len(self.action_instants) - 1:  # the learning ends before the last instant
            # all actions have been taken. wrap up the simulation
            done = True

        # apply control for current step
        self.sim_case.TurbineGov.set(
            src='paux0', idx=self.tg_idx, value=action, attr='v')

        # Run andes TDS to the next time and increment self.i by 1
        sim_crashed = not self.sim_to_next()

        # get frequency and ROCOF data
        freq = self.sim_case.dae.x[self.w]

        # --- Temporarily disable ROCOF ---
        # rocof = np.array(self.sim_case.dae.y[self.dwdt]).reshape((-1, ))
        # obs = np.append(freq, rocof)

        obs = freq

        if sim_crashed:
            reward -= 9999
            done = True

        # reward functions
        # reward -= np.sum(np.abs(2 * 100 * action))

        if not sim_crashed and done:
            reward -= np.sum(np.abs(3000 * (freq - 1)))
        else:
            reward -= np.sum(np.abs(100 * (freq - 1)))

        # store last action
        self.action_last = action

        # add the first frequency value to `self.freq_print`
        self.freq_print.append(freq[0])
        self.action_0_print.append(action[0])
        self.action_1_print.append(action[1])
        self.action_2_print.append(action[2])
        self.action_3_print.append(action[3])
        self.action_4_print.append(action[4])
        self.reward_print.append(reward)

        if done:
            self.action_total_print = []
            for i in range(len(self.action_0_print)):
                self.action_total_print.append(self.action_0_print[i]
                                               + self.action_1_print[i]
                                               + self.action_2_print[i]
                                               + self.action_3_print[i]
                                               + self.action_4_print[i])
            print("Action #0: {}".format(self.action_0_print))
            print("Action #1: {}".format(self.action_1_print))
            print("Action #2: {}".format(self.action_2_print))
            print("Action #3: {}".format(self.action_3_print))
            print("Action #4: {}".format(self.action_4_print))
            print("Action Total: {}".format(self.action_total_print))
            print("Disturbance: {}".format(self.disturbance))
            print("Freq on #0: {}".format(self.freq_print))
            print("Freq postfault: {}".format(self.freq_print[0] * 60))
            print("Freq after control: {}".format(self.freq_print[-1] * 60))
            print("Rewards: {}".format(self.reward_print))
            print("Total Rewards: {}".format(sum(self.reward_print)))

            # record the final frequency
            self.final_freq.append(self.freq_print[-1] * 60)

            # store data for rendering. To workwround automatic resetting by VecEnv
            widx = self.w

            self.sim_case.dae.ts.unpack()
            xdata = self.sim_case.dae.ts.t
            ydata = self.sim_case.dae.ts.x[:, widx]

            self.t_render = np.array(xdata)
            self.final_obs_render = np.array(ydata)


        return obs, reward, done, {}

    def render(self, mode='human'):

        print("Entering render...")

        if self.fig is None:
            self.fig = plt.figure(figsize=(9, 6))

            self.ax = self.fig.add_subplot(1, 1, 1)

            self.ax.set_xlim(left=0, right=np.max(self.t_render))
            self.ax.set_ylim(auto=True)
            self.ax.xaxis.set_tick_params(labelsize=16)
            self.ax.yaxis.set_tick_params(labelsize=16)
            self.ax.set_xlabel("Time [s]", fontsize=16)
            self.ax.set_ylabel("Bus Frequency [Hz]", fontsize=16)
            self.ax.ticklabel_format(useOffset=False)

            plt.ion()
        else:
            self.ax.clear()
            self.ax.set_xlim(left=0, right=np.max(self.t_render))
            self.ax.set_ylim(auto=True)
            self.ax.set_xlabel("Time [s]", fontsize=16)
            self.ax.set_ylabel("Bus Frequency [Hz]", fontsize=16)
            self.ax.ticklabel_format(useOffset=False)

        for i in range(self.N_Bus):
            self.ax.plot(self.t_render, self.final_obs_render[:, i]*60)

        self.fig.canvas.draw()

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        else:
            raise NotImplementedError

    def close(self):
        pass
