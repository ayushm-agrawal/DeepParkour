import gym
import numpy as np
import os.path, sys, io
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import pybullet_envs
import time
import tensorflow as tf
from model.simple_policy import SimplePolicy
import numpy as np
import random
from util.util import select_eps_greedy_action, ReplayMemory, Transition

# create and render environment
env = gym.make("HumanoidBulletEnv-v0")
# env.render(mode="human")
# get shape of actions
NUM_ACTIONS = env.action_space.shape[0]
# get shape of observations
OBS_SHAPE = env.observation_space.shape[0]

# # setup hyperparameters
# TARGET_UPDATE_STEP_FREQ = args.target_update
# BATCH_SIZE = args.batch_size
# EPS_START = args.eps_start
# EPS_END = args.eps_end
# EPS_DECAY = args.eps_decay
# EPISODE_NUM = args.ep_num
# REPLAY_BUFFER_SIZE = 50000
# max_steps = args.max_steps
# LEARNING_RATE = args.learning_rate
# GAMMA = 0.99

# policy
pi = SimplePolicy(OBS_SHAPE, NUM_ACTIONS)
replay_memory = ReplayMemory(50000)

# print("Batch Size: {}".format(BATCH_SIZE))
# print("Episodes: {}".format(EPISODE_NUM))
# print("Target Update Freq: {}".format(TARGET_UPDATE_STEP_FREQ))
# print("========================\n")
# print("Epsilon Start: {}".format(EPS_START))
# print("Epsilon End: {}".format(EPS_END))
# print("Epsilon Decay: {}".format(EPS_DECAY))
# print("========================\n")

with tf.Session() as session:
    # initialize variables
    session.run(tf.global_variables_initializer())
    step = 0
    score_list = [] 
    exploit= 0 
    explore = 0
    
    while len(replay_memory) < 64:
        prev_observation = env.reset()
        observation, reward, done, _ = env.step(np.random.rand(NUM_ACTIONS))
        
        prepped_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
        action, count_explore, count_exploit = select_eps_greedy_action(session, pi, prepped_obs, step, NUM_ACTIONS, exploit, explore)
        observation, reward, done, info = env.step(action)
        # add to memory
        print("Filling Replay Memory.", end='\r')
        replay_memory.push(prev_observation, action, observation, reward)
    
    print("\n====================\n")
    print("Training Start\n")
    for episode in range(10):
        print("------------------")
        print("| Episode: {}".format(episode))
        # initialize environment
        prev_observation = env.reset()
        observation, reward, done, _ = env.step(np.random.rand(NUM_ACTIONS))
        done = False
        ep_score = 0
        steps = 0
        exploit= 0 
        explore = 0
        while steps < 100: # until the episode ends
            steps += 1
            
            # select and perform an action
            prepped_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
            action, count_explore, count_exploit = select_eps_greedy_action(session, pi, prepped_obs, step, NUM_ACTIONS, exploit, explore)
            observation, reward, done, info = env.step(action)
            # add to memory
            replay_memory.push(prev_observation, action, observation, reward)
            prev_observation = observation

            # before enough transitions are collected to form a batch
            if len(replay_memory) < 64:
                break

            # prepare training batch
            transitions = replay_memory.sample(64)
            batch = Transition(*zip(*transitions))
            next_states = np.array(batch.next_state, dtype=np.float32)
            state_batch = np.array(batch.state, dtype=np.float32)
            action_batch = np.array(batch.action, dtype=np.int64)
            reward_batch = np.array(batch.reward)

            # state values
            state_output = pi.take_action(session, state_batch)

            # calculate best value at next state
            next_state_output = pi.take_action(session, next_states)
            # compute the expected Q values
            expected_state_action_values = (next_state_output * 0.99) + reward_batch
            # optimize
            loss = pi.loss_optimize(session, state_batch, expected_state_action_values, state_output)

            ep_score += reward
            step += 1
            exploit= count_exploit 
            explore = count_explore

            
        # #update the target network, copying all variables in DQN
        # if episode % TARGET_UPDATE_STEP_FREQ == 0:
        #     # get trainable variables
        #     trainable_vars = tf.trainable_variables()
        #     # length of trainable variables
        #     total_vars = len(trainable_vars)
        #     # list to hold all operators
        #     ops = []

        #     # iterate through policy model weights
        #     policy_model_weights = trainable_vars[0:total_vars//2]
        #     for idx, var in enumerate(policy_model_weights):
        #         # get target model weights
        #         target_model_weights = trainable_vars[idx + total_vars//2]
        #         # assign policy model weights to target model weights
        #         ops.append(target_model_weights.assign((var.value())))

        #     # run session to transfer weights
        #     for op in ops:
        #         session.run(op)
                
        print("| Steps: {}".format(steps))
        print("| Score: {}".format(ep_score))
        print("| Expore: {}".format(count_explore))
        print("| Exploit: {}".format(count_exploit))
        print("------------------")
        # print("Episode {} achieved score {} at {} training steps\n".format(episode, ep_score, step))
        score_list.append(ep_score)

    avg_score = (sum(score_list))/(len(score_list))
    print("\nAverage episode score: {}".format(avg_score))
    print("Top score for all episodes: {}".format(max(score_list)))
    print("Total steps taken: {}".format(step))

# def main():
#     # policy
#     pi = SimplePolicy(env.observation_space, action_space)
#     # reset
#     env.reset()
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         for episode in range(10):
#             frame, score, restart_delay = 0, 0, 0
#             obs = env.reset()
#             while True:
#                 action = pi.take_action(session, obs)
#                 observation, reward, done, info = env.step(action)
