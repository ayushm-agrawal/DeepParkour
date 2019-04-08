import gym
import numpy as np 
import pybullet_envs
import time
import tensorflow as tf

def relu(x):
    return np.maximum(x, 0)

# class small_reactive_policy:
#     """Simple multi-layer perceptron policy, no internal state"""
#     def __init__(self, observation_space, action_space):
#         assert weights_dense1_w.shape == (observation_space.shape[0], 256)
#         assert weights_dense2_w.shape == (256, 128)
#         assert weights_final_w.shape == (128, action_space.shape[0])
    
#     def act(self, ob):
#         ob[0] += -1.4 + 0.8
#         x = ob
#         x = relu(np.dot(x, weights_dense1_w) + weights_dense1_b)
#         x = relu(np.dot(x, weights_dense2_w) + weights_dense2_b)
#         x = np.dot(x, weights_final_w) + weights_final_b
        # return x

def main():
    session = tf.Session()
    saver = tf.train.import_meta_graph('./test_agent.meta')
    saver.restore(session,'./test_agent')
    graph = session.graph
    obsv = graph.get_tensor_by_name('SimplePolicy/observations:0')
    actions = graph.get_tensor_by_name('output:0')
    # print(session.graph.get_operations())
    env = gym.make("HumanoidBulletEnv-v0")
    # env.render(mode="human")
    # print("Obs: {}".format(env.observation_space.shape[0]))
    # print("Action: {}".format(env.action_space))
    # pi = small_reactive_policy(env.observation_space, env.action_space)
    env.reset()
    while True:
        frame, score, restart_delay = 0, 0, 0
        obs = env.reset()
        obs = obs.reshape(1,44)
        while True:
            a = session.run(actions, feed_dict = {obsv: obs})
            obs, r, done, _ = env.step(a[0])
            obs = obs.reshape(1,44)
            # print("Observations: {}".format(obs))
            # print("Reward: {}".format(r))
            # print("Action: {}".format(a))            
            score += r
            frame += 1
            time.sleep(1./60.)

            still_open = env.render("human")
            if still_open==False:
                return 
            if not done: 
                continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60*2 # 2 secs at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0: break


if __name__=="__main__":
    main()

