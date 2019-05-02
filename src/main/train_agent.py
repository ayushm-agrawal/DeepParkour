""" This file trains the agent """
import argparse
import os.path
import gym
from baselines.common import tf_util as U
from src.model.ppo_policy import PpoAgent


def train(train_env, num_timesteps, model_path):
    """
    This function trains the model
    params:
        - train_env: training environment name
        - num_timesteps: timesteps to train
        - model_path: dir to save agent
    returns:
    """
    # create environment
    env = gym.make(train_env)
    # create session
    U.make_session(num_cpu=2).__enter__()
    # scale rewards by a factor of 10
    env = RewScale(env, 0.1)

    ppo_agent = PpoAgent(env, total_timesteps=num_timesteps)
    p_i = ppo_agent.policy()

    env.close()
    # save model
    if model_path:
        U.save_state(model_path)
    return p_i


class RewScale(gym.RewardWrapper):
    """ This scales the environment rewards by 1/10 """
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


def main():
    """ This function sets up the parser and trains the agent """
    # setup parser
    parser = argparse.ArgumentParser(description='Train Humanoid Agent.')
    parser.add_argument('--model-path', default=os.path.join('../agents/', 'humanoid_policy'))
    parser.add_argument('--timesteps', type=int, default=5e7, help='number of training steps')
    parser.add_argument('--env', type=str, default="ObstacleEnv-v0", help='Environment for train')
    args = parser.parse_args()

    print('Training the humanoid agent')
    # train the agent
    print("=================================================")
    print("Environment: {}".format(args.env))
    print("=================================================")
    train(train_env=args.env, num_timesteps=args.timesteps, model_path=args.model_path)

if __name__ == '__main__':
    main()
