import tensorflow as tf

class SimplePolicy(object):
    def __init__(self, obs_space, action_space, name='SimplePolicy'):
        self.name = name

        with tf.variable_scope(name):
            # observation placeholder
            self.obs = tf.placeholder(dtype=tf.float32, shape=(None, obs_space), name="observations")
            self.actions = tf.placeholder(tf.float32, [None, action_space], name='actions')
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            # dense 1
            self.dense_1 = tf.layers.Dense(256, activation=tf.nn.relu, name="hidden_1")(self.obs)
            # dense 2
            self.dense_2 = tf.layers.Dense(128, activation=tf.nn.relu, name="hidden_2")(self.dense_1)
            # # dense 3
            # self.dense_3 = tf.layers.Dense(128, activation=tf.nn.relu, name="hidden_3")(self.dense_2)
            # output
            self.output = tf.layers.Dense(action_space, name="output")(self.dense_2)
        
        # compute the loss
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
        self.loss = tf.losses.huber_loss(self.target_Q, self.Q)
        # pass it to optimizer
        self.optimizer = tf.train.RMSPropOptimizer(0.0005)
        self.train_op = self.optimizer.minimize(self.loss)

    def take_action(self, session, obs):
        """
        This function uses the agent to take an action
        params:
            session: the current tensorflow session
            obs: observation
        returns:
            action: next best action
        """
        # get actions from the model
        actions = session.run(self.output, feed_dict={self.obs: obs})

        return actions

    def loss_optimize(self, session, state_batch, expected_vals, predicted_vals):
        loss, _ = session.run([self.loss, self.train_op], feed_dict={self.obs: state_batch, self.target_Q: expected_vals, self.actions: predicted_vals})
        return loss
