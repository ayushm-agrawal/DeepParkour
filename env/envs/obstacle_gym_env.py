import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import math
import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet

import random
from pybullet_utils import bullet_client
import pybullet_data
from pkg_resources import parse_version

class ObstacleGym(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    
    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=50,
                 isEnableSelfCollision=True,
                 isDiscrete=False,
                 renders=False):
        print('init')
        self._timeStep = 0.01
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._ballUniqueId = -1
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        if self._renders:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        self.seed()
        observationDim = 2
    def reset(self):
        pass
    def __del__(self):
        pass
    def seed(self):
        pass
    def getExtendedObservation(self):
        pass
    def step(self):
        pass
    def render(self):
        pass
    def _reward(self):
        pass
    def _termination(self):
        pass
    
    
