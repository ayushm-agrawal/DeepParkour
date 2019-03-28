import gym
import numpy as np
import os.path, sys, io
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import pybullet_envs
import time
from baselines import deepq
import numpy as np

def main():
    # create and render environment
    env = gym.make("HumanoidBulletEnv-v0")
    # env.render(mode="human")
    # get shape of actions
    NUM_ACTIONS = env.action_space.shape[0]
    # get shape of observations
    OBS_SHAPE = env.observation_space.shape[0]

    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to humanoid_model.pkl")
    act.save("humanoid_model.pkl")

if __name__ == '__main__':
    main()
