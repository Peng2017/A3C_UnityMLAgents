import numpy as np
import tensorflow as tf

from A3C.history import *

class Trainer(object):
    def __init__(self, a3c_model, sess, info, training):
        """
        Responsible for collecting experiences and training PPO model.
        :param a3c_model: Tensorflow graph defining model.
        :param sess: Tensorflow session.
        :param info: Environment BrainInfo object.
        """

        self.model = a3c_model
        self.sess = sess
        stats = {'cumulative_reward': [], 'episode_length': [], 'value_estimate': [],
                 'entropy': [], 'value_loss': [], 'policy_loss': [], 'learning_rate': []}
        self.stats = stats
        self.is_training = training
        self.reset_buffers(info, total=True)
        self.training_buffer = vectorize_history(empty_local_history({}))

    def running_average(self, data, steps, running_mean, running_variance):
        """
        Computes new running mean and variances.
        :param data: New piece of data.
        :param steps: Total number of data so far.
        :param running_mean: TF op corresponding to stored running mean.
        :param running_variance: TF op corresponding to stored running variance.
        :return: New mean and variance values.
        """
        mean, var = self.sess.run([running_mean, running_variance])
        current_x = np.mean(data, axis=0)
        new_mean = mean + (current_x - mean) / (steps + 1)
        new_variance = var + (current_x - new_mean) * (current_x - new_mean)
        return new_mean, new_variance

    def take_action(self, info, env, brain_name, steps, normalize):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param info: Current BrainInfo from environment.
        :param env: Environment to take actions in.
        :param brain_name: Name of brain we are learning model for.
        :return: BrainInfo corresponding to new environment state.
        """
        epsi = None
        feed_dict = {self.model.batch_size: len(info.states)}
        run_list = [self.model.output, self.model.probs, self.model.value, self.model.entropy,
                    self.model.learning_rate]

        for i, _ in enumerate(info.observations):
            feed_dict[self.model.observation_in[i]] = info.observations[i]

        if self.is_training:
            actions, a_dist, value, ent, learn_rate = self.sess.run(run_list, feed_dict)
        self.stats['value_estimate'].append(value)
        self.stats['entropy'].append(ent)
        self.stats['learning_rate'].append(learn_rate)

        new_info = env.step(actions, value={brain_name: value})[brain_name]
        self.add_experiences(info, new_info, actions, a_dist, value)
        return new_info

    def add_experiences(self, info, next_info, actions, a_dist, value):
        """
        Adds experiences to each agent's experience history.
        :param info: Current BrainInfo.
        :param next_info: Next BrainInfo.
        :param actions: Chosen actions.
        :param a_dist: Action probabilities.
        :param value: Value estimates.
        """
        for (agent, history) in self.history_dict.items():
            if agent in info.agents:
                idx = info.agents.index(agent)
                if not info.local_done[idx]:
                    for i, _ in enumerate(info.observations):
                        history['observations%d' % i].append([info.observations[i][idx]])
                        history['actions'].append(actions[idx])
                        history['rewards'].append(next_info.rewards[idx])
                        history['action_probs'].append(a_dist[idx])
                        history['value_estimates'].append(value[idx][0])
                        history['cumulative_reward'] += next_info.rewards[idx]
                        history['episode_steps'] += 1
    
    def process_experiences(self, info, time_horizon, gamma, lambd):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param info: Current BrainInfo
        :param time_horizon: Max steps for individual agent history before processing.
        :param gamma: Discount factor.
        :param lambd: GAE factor.
        """
        for l in range(len(info.agents)):
            if(info.local_done[l] or len(self.history_dict[info.agents[l]]['actions']) > time_horizon) and len(
                self.history_dict[info.agents[l]]['actions']) > 0:

                if info.local_done[l]:
                    next_reward = 0.0
                else:
                    feed_dict = {self.model.batch_size: len(info.states)}
                    for i in range(self.info.observations):
                        feed_dict[self.moel.observations_in[i]] = info.observations[i]
                    value_next = self.sess.run([self.model.value, feed_dict])[l]

                history= vectorize_history(self.history_dict[info.agents[l]])
                history['advantages'] = get_advantage(rewars = history['rewards'],
                                                      value_estimates=history['value_estimates'],
                                                      value_next=value_next,
                                                      gamma=gamma,
                                                      lambd=lambd)
                history['discounted_returns'] = history['advantages'] + history['value_estimates']
                if(len(self.training_buffer['actions']) > 0):
                    append_history(global_buffer=self.training_buffer)


