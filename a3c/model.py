import numpy as np
import tensorflow as tf
import os

from unityagents.environment import UnityEnvironmentException

class AC_Network():
    def __init__(self, o_size_h, o_size_w, o_channels, s_size, a_size, trainer, scope, activation):
        with tf.variable_scope(scope):

            self.image_in = tf.placeholder(shape=[None, 128 * 128], dtype=tf.float32)

            self.inputs = tf.reshape(self.image_in, [-1, 128, 128, 1])

            self.dw_1 = tf.layers.conv2d(self.inputs, o_channels, kernel_size=[8, 8], strides=[4, 4], 
                                     use_bias=False, activation=None, name="dw_1")
            self.pw_1 = tf.layers.conv2d(self.dw_1, 16, kernel_size=[1, 1], strides=[1,1], 
                                         use_bias=False, activation=activation, name="pw_1")

            self.dw_2 = tf.layers.conv2d(self.pw_1, 16, kernel_size=[4, 4 ], strides=[2, 2], 
                                         use_bias=False, activation=None, name="dw_2")
            self.pw_2 = tf.layers.conv2d(self.dw_2, 32, kernel_size=[1, 1], strides=[1,1], 
                                         use_bias=False, activation=activation, name="pw_2")
            
            
            self.pw_2_flatten = tf.contrib.layers.flatten(self.pw_2)

            self.hidden_1 = tf.layers.dense(self.pw_2_flatten, 256, activation = activation, use_bias = True, name="hidden_1")

            #Output layers
            self.policy = tf.layers.dense(self.hidden_1, a_size, activation=tf.nn.softmax, use_bias=False, 
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.01), name="policy")

            self.value = tf.layers.dense(self.hidden_1, 1, activation = None, use_bias=False,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0), name="value")

            self.output = tf.multinomial(self.policy, 1)
            self.output = tf.identity(self.output, name="action")
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)

                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01
                self.loss = tf.clip_by_value(self.loss, 1e30000, -1e30000)
                #Get the gradients from local network using locaal losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                self.var_norms = tf.clip_by_value(self.var_norms, 1e30000, -1e30000)

                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                #Apply local gradients to global networks
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

    @staticmethod
    def save_global_weights(sess, file_name):
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        for var in global_vars:
            value = sess.run(var)
            #value.tofile(str(file_name + var.name + ".bin"))
            name = var.name.replace("/", "_")
            name = name.replace(":", "_")

            if len(var.shape) == 4:
                for last_index in range(value.shape[3]):
                    for third_index in range(value.shape[2]):
                        matrix = value[:, :, third_index, last_index]
                        matrix_name = name + "_{}_{}".format(third_index, last_index)
                        fn = str(file_name + matrix_name + ".txt")
                        np.savetxt(fn, matrix,  fmt="%10.5f", delimiter=",")

            elif len(value.shape) == 2 or len(value.shape) == 1:
                fn = str(file_name + name + ".txt")
                np.savetxt(fn, matrix,  fmt="%10.5f", delimiter=",")

                



#def create_agent_model(env, lr = 1e-4, h_size = 128, epsilon = 0.2, beta = 1e-3, max_step=5e6, normalize = False, num_layers = 2):
#    """
#    Takes a Unity environment and model-specific hyper-parameters and returns the
#    appropriate A3C agent model for the environment.
#    :param env: a Unity environment.
#    :param lr: Learning rate.
#    :param h_size: Size of hidden layers/
#    :param epsilon: Value for policy-divergence threshold.
#    :param beta: Strength of entropy regularization.
#    :return: a sub-class of PPOAgent tailored to the environment.
#    :param max_step: Total number of training steps.
#    """

#    if(num_layers < 1) : num_layers = 1

#    brain_name = env.brain_names[0]
#    brain = env.brains[brain_name]

#    if(brain.action_space_type == "discrete"):
#        return DiscreteControlModel(lr, brain, h_size, epsilon, beta, max_step, normalize, num_layers)
#    else:
#        return None


#class A3CModel(object):
#    def __init__(self):
#        self.normalize = False
#        self.observation_in = []

#    def create_global_step(self):
#        """Creates global TF op to track and increment global training step"""
#        self.global_step = tf.Variable(0, name='global_step', trainable = False, dtype = tf.int32)
#        self.increment_step = tf.assign(self.global_step, self.global_step + 1)

#    def create_reward_encoder(self):
#         """Creates TF ops to track and increment recent average cumulative reward."""
#         self.last_reward = tf.Variable(0, name = 'last_reward', trainable=False, dtype=tf.float32)
#         self.new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
#         self.update_reward = tf.assign(self.last_reward, self.new_reward)

#    def create_visual_encoder(self, o_size_h, o_size_w, bw, h_size, num_streams, activation, num_layers):
#        """
#        Builds a set of visual (CNN) encoders.
#        :param o_size_h: Height observation size.
#        :param o_size_w: Width observation size.
#        :param bw: Whether image is greyscale {True} or color {False}.
#        :param h_size: Hidden layer size.
#        :param num_streams: Number of visual streams to construct.
#        :param activation: What type of activation function to use for layers.
#        :return: List of hidden layer tensors.
#        """

#        if bw:
#            c_channels = 1
#        else:
#            c_channels = 3

#        self.observation_in.append(tf.placeholder(shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32, 
#                                                  name='observation_%d' % len(self.observation_in)))

#        streams = []
#        for i in range(num_streams):
#            self.dw_1 = tf.layers.conv2d(self.observation_in[-1], c_channels, kernel_size=[8, 8], strides=[4, 4],
#                                         use_bias=False, activation=None)
#            self.pw_1 = tf.layers.conv2d(self.dw_1, 16, kernel_size=[1, 1], strides=[1,1],
#                                       use_bias=False, activation=activation)

#            self.dw_2 = tf.layers.conv2d(self.pw_1, 16, kernel_size=[4, 4 ], strides=[2, 2],
#                                         use_bias=False, activation=None)
#            self.pw_2 = tf.layers.conv2d(self.dw_2, 32, kernel_size=[1, 1], strides=[1,1],
#                                       use_bias=False, activation=activation)
            
            
#            self.pw_2_flatten = tf.contrib.layers.flatten(self.pw_2)

#            self.hidden_1 = tf.layers.dense(self.pw_2_flatten, 256, activation = activation, use_bias = True)

#            streams.append(self.hidden_1)
#        return streams

#    def create_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
#        """
#        Builds a set of hidden state encoders.
#        :param s_size: state input size.
#        :param h_size: Hidden layer size.
#        :param num_streams: Number of state streams to construct.
#        :param activation: What type of activation function to use for layers.
#        :return: List of hidden layer tensors.
#        """
#        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name = 'state')
#        if(self.normalize):
#            self.running_mean = tf.get_variable("running_mean", [s_size], dtype=tf.float32, trainable=False, initializer=tf.ones_initializer())
#            self.running_variance = tf.get_variable("running_variance", [s_size], dtype=tf.float32, trainable=False, initializer=tf.ones_initializer())
#            self.normalized_state = tf.clip_by_value(
#                (self.state_in - self.running_mean) / tf.sqrt(self.running_variance / (tf.cast(self.global_step, tf.float32) + 1)),
#                -5, 
#                5, 
#                name='normalized_state')
#            self.new_mean = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_mean')
#            self.new_variance = tf.placeholder(shape=[s_size],dtype=tf.float32, name='new_variance')
#            self.update_mean = tf.assign(self.running_mean, self.new_mean)
#            self.update_variance = tf.assign(self.running_variance, self.new_variance)
#        else:
#            self.normalized_state = self.state_in
        
#        streams = []
#        for i in range(num_streams):
#            hidden = self.normalized_state
#            for j in range(num_layer):
#                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
#            streams.append(hidden)
#        return hidden

#    def create_a3c_optimizer(self, policy, value, entropy, lr, scope, max_step):
#        """
#        Creates training-specific Tensorflow ops for A3C model.
#        :param probs: Current policy probabilities
#        :param value: Current value estimate
#        :param beta: Entropy regularization strength
#        :param entropy: Current policy entropy
#        :param epsilon: Value for policy-divergence threshold
#        :param lr: Learning rate
#        :param max_step: Total number of training steps.
#        """

#        if scope != 'global':
           
#            self.returns_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='discounted_rewards')
#            self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')
#            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')

#            #Loss functions
        
#            self.value_loss = 0.5 * tf.reduce_sum(tf.square(target_v - tf.reshape(value,[-1])))
#            self.policy_loss = - tf.reduce_sum(tf.log(policy) * self.advantage)
#            self.loss = 0.5 * self.value_loss + self.policy_loss - entropy * 0.01

#            self.learning_rate = tf.train.polynomial_decay(lr, self.global_step, max_step, 1e-10, power=1.0)

#            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

#            #Get the gradients from local network
#            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
#            self.gradients = tf.gradients(self.loss, local_vars)
#            self.var_norms = tf.global_norm(local_vars)
#            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

#            #Apply local gradients to global network
#            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
#            self.apply_grads = self.optimizer.apply_gradients(zip(grads.global_vars))


#class DiscreteModel(A3CModel):
#    def __init__(self, brain, lr, h_size, epsilon, beta, max_step, normalize, num_layers, scope):
#        """
#        Creates Discrete Control Actor-Critic model.
#        :param brain: State-space size
#        :param h_size: Hidden layer size
#        """
#        with tf.variable_scope(scope):
#            super(DiscreteModel, self).__init__()
#            self.create_global_steps()
#            self.create_reward_encoder()
#            self.normalize = normalize

#            hidden_visual, hidden = None, None

#            if brain.number_observation > 0:
#                enconders = []
#                for i in range(brain.number_observation):
#                    height_size, width_size = brain.camera_resolution[i]['height'], brain.camera_resolution[i]['width']
#                    bw = brain.camera_resolution[i]['blackAndWhite']
#                    enconders.append(self.create_visual_encoder(height_size, width_size, bw, h_size, 1, tf.nn.relu, num_layers)[0])
#                hidden_visual = tf.concat(enconders, axis=1)
        
#            if hidden_visual is None:
#                raise Exception("No valid network configuration possible. "
#                                "There are no observations in this brain")
#            else:
#                hidden = hidden_visual

#            a_size= brain.action_space_size

#            self.batch_size = tf.placeholder(shape = None, dtype=tf.int32, name='batch_size')
#            self.policy = tf.layers.dense(hidden, a_size, activation=None, use_bias=False,
#                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.01))
#            self.probs = tf.nn.softmax(self.policy, name='action_probs')
#            self.output = tf.multinomial(self.policy, 1)
#            self.output = tf.identity(self.output, name='action')
#            self.value = tf.layers.dense(hidden, 1, activation = None, use_bias=False,
#                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0))
#            self.value = tf.identity(self.value, name = 'value_estimate')
#            self.entropy = - tf.reduce_sum(self.probs * tf.log(self.probs + 1e-10), axis = 1)

#            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
#            self.selected_actions = tf.contrib.layer.one_hot_encoding(self.action_holder, a_size)
#            self.responsible_probs = tf.reduce_sum(self.probs * selected_actions, axis = 1)

#            self.create_a3c_optimizer(self.respondible_probs, self.value, self.entropy, lr, scope)