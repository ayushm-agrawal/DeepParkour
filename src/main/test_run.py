import argparse
import gym
import pybullet_envs
from baselines import deepq


# setup parser
parser = argparse.ArgumentParser(description='Train Humanoid Agent.')
parser.add_argument(
    '--model_dir',
    type=str,
    default='/home/cse496dl/atendle/Final_Project/DeepParkour/agents',
    help='directory where agents are saved')
parser.add_argument('--batch_size', type=int, default=64, help='mini batch size for training')
parser.add_argument('--network', type=int, default='mlp', help='model/agent to use')
parser.add_argument('--total_timesteps', type=int, default=100000, help='number of timesteps to run')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for optimizer')
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--print_frequency', type=float, default=100, help='Frequency of prints')
parser.add_argument('--exp_frac', type=float, default = 0.1, help='fraction of entire training period over which the exploration rate is annealed')
parser.add_argument('--exp_final', type=float, default = 0.02, help='final value of random action probability')
parser.add_argument('--buffer', type=int, default=50000, help='size of the replay buffer')
parser.add_argument('--render', type=int, default=0, help='flag to render(1 == render) (0 == do not render)')
# setup parser arguments
args = parser.parse_args()

def main():
    # create environment
    env = gym.make("HumanoidBulletEnv-v0")
    # render environment
    if args.render == 1:
        env.render(mode="human")
        
    # log necessary hyperparameters
    print('Number of observations collected by the Environment: {}'.format(env.observation_space.shape[0]))
    print('Number of actions an agent can take: {}'.format(env.action_space.shape[0]))
    print('Batch Size: {}'.format(args.batch_size))
    print('Discount Factor: {}'.format(args.gamma))
    print('Type of Network: {}'.format(args.network))
    print('Learning Rate: {}'.format(args.learning_rate)
    print('Total Timesteps: {}'.format(args.total_timesteps))
    print('Buffer Size: {}'.format(args.buffer))
    print('Exploration Fraction: {}'.format(args.exp_frac))
    print('Exploration Final Episode: {}'.format(args.exp_final))
      
   
    
    # train the agent using learning algorithm
    act = deepq.learn(
        env,
        batch_size=args.batch_size,
        gamma=args.gamma,
        network=args.network,
        lr=args.learning_rate,
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer,
        exploration_fraction=args.exp_frac,
        exploration_final_eps=args.exp_final,
        print_freq=args.print_frequency
    )
          
    print("Saving model")
    act.save("humanoid_model.pkl")

if __name__ == '__main__':
    main()
