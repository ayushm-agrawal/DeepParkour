"""Init file for environment"""
import gym
from gym.envs.registration import register
register(
	   id='ObstacleEnv-v0',
	   entry_point='env.envs.gym_locomotion_env:ObstacleBulletEnv')
