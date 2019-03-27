import random
import math
import numpy as np
import tensorflow as tf
import collections
import sys

Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_eps_greedy_action(session, policy_model, obs, step, num_actions, exploit, explore, EPS_START=1.0, EPS_END=0.01, EPS_DECAY=10000):
    """
    Decides whether the agent should exploit or explore

    :param policy_model: (callable): mapping of `obs` to q-values
    :param obs: (np.array): current state observation
    :param step: (int): training step count
    :param num_actions: (int): number of actions available to the agent
    :param EPS_START:
    :param EPS_END:
    :param EPS_DECAY:
    :return: action
    """

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)
    # exploit
    if random.random() > eps_threshold:
        exploit += 1
        action = policy_model.take_action(session, obs)
    # explore
    else:
        explore += 1
        action = np.random.rand(num_actions)
      

    return action, explore, exploit
