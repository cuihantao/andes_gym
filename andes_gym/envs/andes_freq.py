"""
Load frequency control environment using ANDES.

This file was part of gym-power and is now part of andes_gym.

Authors:
Hantao Cui (cuihantao@gmail.com)
Yichen Zhang (whoiszyc@hotmail.com)

Modification and redistribution of this file is subject to a collaboration agreement.
Derived source code should be made available to all authors.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pathlib

# Import general modules
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from cvxopt import matrix

import andes


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
        self.path = os.path.join(path, "CL_2machine_mpc.dm")

        self.tf = 10.0  # end of simulation time
        self.tstep = 1/60  # simulation time step
        self.fixt = False  # if we do fixed step integration

        self.action_instants = np.array([0.1, 0.2, 0.5, 0.9, 1.5, 3.5, 5.5])

        self.N = len(self.action_instants)  # number of actions
        self.N_TG = 2  # number of TG1 models
        self.N_Bus = 5

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.N_TG,))
        self.observation_space = spaces.Box(low=-5, high=5, shape=(2 * self.N_Bus,))

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

        self.sim_case = andes.main.run(self.path, routine="pflow", no_output=True)

        # TODO: some configurations
        self.sim_case.tds.config.fixt = self.fixt

        # TODO: fix the hardcode
        self.pm0 = matrix(self.sim_case.TG1.pm0)  # original pm0
        self.w = matrix(self.sim_case.BusFreq.w)
        self.dwdt = matrix(self.sim_case.BusFreq.dwdt)

        self.action_last = np.zeros(self.N_TG)
        # TODO: add the load disturbance model

        # Step to the first action instant
        assert self.sim_to_next(), "First simulation step failed"

        self.freq_print = []
        self.action_print = []
        self.reward_print = []

    def sim_to_next(self):
        """
        Simulate to the next action time instance.
        Increase the counter i each time
        """
        next_time = self.tf
        if self.i < len(self.action_instants):
            next_time = float(self.action_instants[self.i])

        self.sim_case.tds.config.tf = next_time
        self.i += 1

        return self.sim_case.tds.run()

    def reset(self):
        print("Env reset.")
        self.initialize()
        return np.append(np.ones(shape=(self.N_Bus, )) ,np.zeros(shape=(self.N_Bus, ) ))

    def step(self, action):
        """
        Stepping function for RL
        """
        reward = 0.0  # reward for the current step
        done = False

        # Get the next action time in the list
        if self.i >= len(self.action_instants):
            # all actions have been taken. wrap up the simulation
            done = True

        # apply control for current step
        # NOTE: the negative value of action will contribute to the "increase" of generator's pm
        self.sim_case.TG1.pm0 = self.pm0 - matrix(action.astype(float))

        # Run andes tds to the next time and increment self.i by 1
        sim_crashed = not self.sim_to_next()

        # get frequency and ROCOF data
        freq = np.array(self.sim_case.dae.x[self.w]).reshape((-1, ))
        rocof = np.array(self.sim_case.dae.y[self.dwdt]).reshape((-1, ))

        obs = np.append(freq, rocof)

        if sim_crashed:
            reward -= 9999
            done = True

        # reward functions
        reward -= np.sum(np.abs(2 * 100 * action))

        if not sim_crashed and done:
            reward -= np.sum(np.abs(60 * 15000 * (freq - 1)))
        else:
            reward -= np.sum(np.abs(50 * 15000 * (freq - 1)))

        # store last action
        self.action_last = action

        # add the first frequency value to `self.freq_print`
        self.freq_print.append(freq[0])
        self.action_print.append(action[0])
        self.reward_print.append(reward)

        if done:
            print("Action #0: {}".format(self.action_print))
            print("Freq on #0: {}".format(self.freq_print))
            print("Rewards: {}".format(self.reward_print))
            print("Total Rewards: {}".format(sum(self.reward_print)))

            # store data for rendering. To workwround automatic resetting by VecEnv
            widx = self.w + 1
            xdata, ydata = self.sim_case.varout.get_xy(widx)

            self.t_render = np.array(xdata).reshape((-1, 1))
            self.final_obs_render = np.array(ydata).reshape((self.N_Bus, -1))

        return obs, reward, done, {}

    def render(self, mode='human'):

        print("Entering render...")

        if self.fig is None:
            self.fig = plt.figure(figsize=(4, 3))

            self.ax = self.fig.add_subplot(1, 1, 1)

            self.ax.set_xlim(left=0, right=np.max(self.t_render))
            self.ax.set_ylim(auto=True)
            self.ax.set_xlabel("Time [s]")
            self.ax.set_ylabel("Bus Frequency [pu]")
            self.ax.ticklabel_format(useOffset=False)

            plt.ion()
        else:
            self.ax.clear()
            self.ax.set_xlim(left=0, right=np.max(self.t_render))
            self.ax.set_ylim(auto=True)
            self.ax.set_xlabel("Time [s]")
            self.ax.set_ylabel("Bus Frequency [pu]")
            self.ax.ticklabel_format(useOffset=False)

        for i in range(self.N_Bus):
            self.ax.plot(self.t_render, self.final_obs_render[i, :])

        plt.draw()
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