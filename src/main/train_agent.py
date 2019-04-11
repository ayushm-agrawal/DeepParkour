import argparse
import os.path, sys, io
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import gym
import pybullet as p
import pybullet_envs
from baselines import deepq
from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.common import tf_util as U
from baselines import logger
import time
from model.ppo_policy import PPO_AGENT


def train(num_timesteps, model_path):
    # create environment
    env = gym.make("HumanoidBulletEnv-v0")
    # create session
    U.make_session(num_cpu=1).__enter__()
    # scale rewards by a factor of 10
    env = RewScale(env, 0.1)
    
    ppo_agent = PPO_AGENT(env, total_timesteps=num_timesteps)
    pi = ppo_agent.policy()
          
    env.close()
    # save model
    if model_path:
        U.save_state(model_path)
    return pi

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale

def main():
    # setup parser
    parser = argparse.ArgumentParser(description='Train Humanoid Agent.')
    parser.add_argument('--model-path', default=os.path.join('../../agents/humanoid_10M/', 'humanoid_policy'))
    parser.add_argument('--timesteps', type=int, default=5e7, help='number of training steps to take')
    args = parser.parse_args()


    print('Training the humanoid agent')
    # train the agent
    train(num_timesteps=args.timesteps, model_path=args.model_path)

if __name__ == '__main__':
    main()
