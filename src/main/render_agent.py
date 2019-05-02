""" This file renders trained agents """
import argparse
import os
import gym
import tensorflow as tf
from baselines.common import tf_util as U



def load_policy(model_path):
    """
    This function loads the agent
    params:
        - model_path: dir where the agent is stored
    returns:
        - session: tensorflow session
        - ob: tensorflow placeholder for observations
        - actions: tensorflow placeholder for actions
    """
    session = tf.Session()
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(session, model_path)
    graph = session.graph
    o_b = graph.get_tensor_by_name('pi/ob:0')
    actions = graph.get_tensor_by_name('pi/pol/final/BiasAdd:0')
    return session, o_b, actions


def test(render_env, model_path, render, episodes):
    """
    This function renders trained agents
    params:
        - render_env: name for the environment
        - model_path: dir where the agent is stored
        - render: GUI render flag
        - episodes: number of times the agents is rerun
    returns:
    """
    # test agent
    env = gym.make(render_env)
    session, o_b, actions = load_policy(model_path)

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
            action = session.run(actions, feed_dict={o_b: obs})[0]
            obs, reward, done, _ = env.step(action)
            obs = obs.reshape(1, 44)

            total_reward += reward
            if done:
                print('Episode {}: Total Reward {}'.format(
                    episode+1, total_reward))
    env.close()


def main():
    """ This function sets up the parser and renders the agent """
    # setup parser
    parser = argparse.ArgumentParser(description='Train Humanoid Agent.')
    parser.add_argument('--model-path', default=os.path.join('../agents/', 'humanoid_policy'))
    parser.add_argument('--render', type=int, default=1, help="CPU render boolean")
    parser.add_argument('--num_episodes', type=int, default=10, help='number of episodes to render')
    parser.add_argument('--env', type=str, default="ObstacleEnv-v0", help='Environment for test')
    args = parser.parse_args()

    test(render_env=args.env, model_path=args.model_path,
         render=args.render, episodes=args.num_episodes)


if __name__ == '__main__':
    main()
