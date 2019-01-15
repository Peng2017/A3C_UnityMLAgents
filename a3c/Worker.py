import tensorflow as tf
import numpy as np
import scipy.signal

from .model import AC_Network

def process_frame(frame):
    r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.reshape(gray, [np.prod(gray.shape)])
    #gray = np.reshape(gray, [128, 128, 1])
    return gray

def discount(r, gamma=0.99, value_next=0.0):
    """
    Computes discounted sum of future rewards for use in updating value estimate.
    :param r: List of rewards.
    :param gamma: Discount factor.
    :param value_next: T+1 value estimate for returns calculation.
    :return: discounted sum of future rewards as list.
    """

    discounted_r = np.zeros_like(r)
    running_add = value_next

    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

class Worker():
    def __init__(self, env, brain, worker_id, s_size, a_size, model_path, trainer, global_episodes):
        self.env = env
        self.name = "worker_" + str(worker_id)
        self.number = worker_id
        self.brain_name = brain.brain_name
        self.model_path = model_path
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.is_training = True
        o_height, o_width = brain.camera_resolutions[0]['height'], brain.camera_resolutions[0]['width']
        s_size = o_height * o_width
        self.local_AC = AC_Network(o_height, o_width, 1, s_size, brain.action_space_size, trainer, self.name, tf.nn.relu)
        self.update_local_ops = update_target_graph('global', self.name)
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
        self.trained_frames = 0

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        #update the global network using gradients from loss
        #Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.image_in: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages}

        v_loss, p_loss, entropy, grad_norms, var_norms, _ = sess.run(
            [
                self.local_AC.value_loss, 
                self.local_AC.policy_loss,
                self.local_AC.entropy,
                self.local_AC.grad_norms,
                self.local_AC.var_norms,
                self.local_AC.apply_grads
                ],
                feed_dict=feed_dict)
        return v_loss / len(rollout), p_loss / len(rollout), entropy /len(rollout), grad_norms, var_norms

    def work(self, max_episode_length, buffer_size, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + self.name)
        self.trained_frames = 0
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0.0
                episode_step_count = 0


                done = False
                #Collect state from env
                info = self.env.reset(train_mode=True)[self.brain_name]
                
                s = np.asarray(info.observations[:])

                s = np.reshape(s, [128, 128, 3])

                s = process_frame(s)
                episode_frames.append(s)

                while not done:
                    run_list = [self.local_AC.policy, self.local_AC.value, self.local_AC.output]
                    feed_dict = {self.local_AC.image_in: [s]}

                    probs, value, actions = sess.run(run_list, feed_dict=feed_dict)
                    selected_action = actions[0,0]
                    _dict = {self.brain_name: value}
                    new_info = self.env.step(actions, value=_dict)[self.brain_name]
                    done = new_info.local_done[0]

                    if done == False:
                        s1 = new_info.observations[:]
                        s1 = np.reshape(s1, [128, 128, 3])
                        s1 = process_frame(s1)
                        episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, selected_action, new_info.rewards[0], s1, done, value[0,0]])
                    episode_values.append(value[0,0])

                    episode_reward += new_info.rewards[0]
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
                    self.trained_frames += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.

                    if len(episode_buffer) == buffer_size and done == False:
                        # Since we don't know what the true final return is, we "bootstrap" 
                        # from our current value estimation

                        v1 = sess.run(self.local_AC.value, feed_dict={self.local_AC.image_in: [s]})[0,0]
                        v_loss, p_loss, entropy, grad_norms, var_norms = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if done:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0:
                    v_loss, p_loss, entropy, grad_norms, var_norms = self.train(episode_buffer, sess, gamma, 0.0)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

                if episode_count % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])

                    tf.summary.scalar('Perf/Reward', mean_reward)
                    tf.summary.scalar('Perf/Length', mean_length)
                    tf.summary.scalar('Perf/Value', mean_value)
                    tf.summary.scalar('Losses/Value_Loss', v_loss)
                    tf.summary.scalar('Losses/Policy_Loss', p_loss)
                    tf.summary.scalar('Losses/Entropy', entropy)
                    tf.summary.scalar('Losses/Grad_Norm', grad_norms)
                    tf.summary.scalar('Losses/Var_Norm', var_norms)

                    merged = tf.summary.merge_all()
                    summary = sess.run(merged)

                    self.summary_writer.add_summary(summary, episode_count)
                        
                    self.summary_writer.flush()

                    print(
                        "{}: Episode = {}; Mean reward = {}; Mean value = {}; Value loss = {}; Policy loss = {}; Frames = {}".format(self.name,
                                                                                                                        episode_count,
                                                                                                                        mean_reward,
                                                                                                                        mean_value,
                                                                                                                        v_loss,
                                                                                                                        p_loss,
                                                                                                                        self.trained_frames))
                    saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.cptk')
                    #AC_Network.save_global_weights(sess, self.model_path+"/saved_weights/")
                    #np.savetxt(self.model_path + "/frames_" + self.name + ".txt", [self.trained_frames])
                    print ("Saved Model")
                if episode_count > max_episode_length:
                    coord.request_stop()

        print(self.name + " trained " + str(self.trained_frames) + " frames")






