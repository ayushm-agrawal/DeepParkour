""" This file creates the PPO agent class """
from baselines.ppo1 import mlp_policy, pposgd_simple

class PpoAgent:
    """ This class creates a PPO agent """
    def __init__(self, env, batch_size=64, optim_epochs=10, optim_stepsize=1e-4,
                 total_timesteps=1e6, gamma=0.99, lam=0.95, tsteps_per_act=2048,
                 ent_coeff=0.0, clip_param=0.1):
        self.env = env
        self.total_timesteps = total_timesteps
        self.discount = gamma
        self.lam = lam
        self.timesteps_per_actorbatch = tsteps_per_act
        self.ent_coeff = ent_coeff
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.opt_stepsize = optim_stepsize
        self.opt_epochs = optim_epochs
    # create the policy function
    def mlp(self, name, ob_space, ac_space):
        """ Creates the Multilayer Perceptron Network for PPO function approximation """
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space,
                                    ac_space=ac_space, hid_size=64, num_hid_layers=2)
    # train the agent using learning algorithm
    def policy(self):
        """ This method creates the learning policy for the agent """
        p_i = pposgd_simple.learn(self.env,
                                  self.mlp,
                                  max_timesteps=self.total_timesteps,
                                  timesteps_per_actorbatch=self.timesteps_per_actorbatch,
                                  clip_param=self.clip_param,
                                  entcoeff=self.ent_coeff,
                                  optim_epochs=self.opt_epochs,
                                  optim_stepsize=self.opt_stepsize,
                                  optim_batchsize=self.batch_size,
                                  gamma=self.discount,
                                  lam=self.lam,
                                  schedule='constant'
                                 )
        return p_i
