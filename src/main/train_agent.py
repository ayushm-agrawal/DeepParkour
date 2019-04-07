import argparse
import os
import gym
import pybullet as p
import pybullet_envs
from baselines import deepq
from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.common import tf_util as U
from baselines import logger
import time

def policy(env, num_timesteps):
    # create the policy function
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=2)
    
    # train the agent using learning algorithm
    pi = pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_actorbatch=2048,
        clip_param=0.1, entcoeff=0.0,
        optim_epochs=10,
        optim_stepsize=1e-4,
        optim_batchsize=64,
        gamma=0.99,
        lam=0.95,
        schedule='constant',
    )
    return pi

def train(num_timesteps, model_path):
    # create environment
    env = gym.make("HumanoidBulletEnv-v0")
    # create session
    U.make_session(num_cpu=1).__enter__()
    # scale rewards by a factor of 10
    env = RewScale(env, 0.1)
    
    pi = policy(env, num_timesteps)
          
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
    parser.add_argument('--model-path', default=os.path.join('../../agents/', 'humanoid_policy'))
    parser.set_defaults(num_timesteps=int(5e6))
    args = parser.parse_args()

    print('Training the humanoid agent')
    # train the agent
    train(num_timesteps=args.num_timesteps, model_path=args.model_path)

if __name__ == '__main__':
    main()
