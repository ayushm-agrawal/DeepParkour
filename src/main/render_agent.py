import argparse
import os

import gym
import pybullet_envs
import tensorflow as tf
import env

from baselines import deepq, logger
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple



def load_policy(model_path):
    session = tf.Session()
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(session, model_path)
    graph = session.graph
    ob = graph.get_tensor_by_name('pi/ob:0')
    actions = graph.get_tensor_by_name('pi/pol/final/BiasAdd:0')
    return session, ob, actions


def test(render_env, model_path, render, episodes):
    # test agent
    env = gym.make(render_env)
    session, ob, actions = load_policy(model_path)

    if render:
        env.render(mode="human")
    U.make_session(num_cpu=2).__enter__()
    env.reset()
    for episode in range(episodes):
        obs = env.reset()
        obs = obs.reshape(1, 44)
        total_reward = 0
        done = False
        while not done:
            action = session.run(actions, feed_dict={ob: obs})[0]
            # pi.act(stochastic=False, ob=ob)[0]
            obs, reward, done, _ = env.step(action)
            obs = obs.reshape(1, 44)

            total_reward += reward
            if done:
                print('Episode {}: Total Reward {}'.format(
                    episode+1, total_reward))
    env.close()


def main():
    # setup parser
    parser = argparse.ArgumentParser(description='Train Humanoid Agent.')
    parser.add_argument('--model-path', default=os.path.join('../agents/10M_AGENT', 'humanoid_policy'))
    parser.add_argument('--render', type=int, default=1, help="CPU render boolean")
    parser.add_argument('--num_episodes', type=int, default=10, help='number of episodes to render')
    parser.add_argument('--env', type=str, default="ObstacleEnv-v0", help='Environment which is used for training/testing')
    args = parser.parse_args()

    test(render_env=args.env, model_path=args.model_path, render=args.render, episodes=args.num_episodes)


if __name__ == '__main__':
    main()
