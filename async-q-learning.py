import threading
import tensorflow as tf
import cv2
import sys
import numpy as np
import random
import gym
import time

# environment name
game = "Pong-v0"

# number of learning agents
num_threads = 12

# agent explorativeness
initial_epsilon = 1
frames_anneal = 1000000.0

# how often to train model
model_update_tsteps = 32

# how often to update target network
update_target_tsteps = 10000

# reward decay
reward_decay = 0.99

# global thread counter
max_thread_step = 5000000

# learning rate
initial_learning_rate = 0.0001

class StepCounter(object):
    def __init__(self):
        self.step_counter = 0

    def increment(self):
        self.step_counter += 1

    def get(self):
        return self.step_counter

    def set(self, num):
        self.step_counter = num

class AgentQNetwork(object):
    def __init__(self, num_actions, network, prefix):
        with tf.name_scope(prefix):
            # input layer
            self.s = tf.placeholder(tf.float32, [None, 80, 80, 4])

            # layer parameters
            self.W_conv1 = self.weight_variable([8, 8, 4, 32])
            self.b_conv1 = self.bias_variable([32])

            self.W_conv2 = self.weight_variable([4, 4, 32, 64])
            self.b_conv2 = self.bias_variable([64])

            self.W_conv3 = self.weight_variable([3, 3, 64, 64])
            self.b_conv3 = self.bias_variable([64])

            self.W_fc1 = self.weight_variable([1024, 256])
            self.b_fc1 = self.bias_variable([256])

            self.W_fc2 = self.weight_variable([256, num_actions])
            self.b_fc2 = self.bias_variable([num_actions])

            # first conv-relu layer
            h_conv1 = self.max_pool(tf.nn.relu(self.conv2d(self.s, self.W_conv1, 4) + self.b_conv1))

            # second conv-relu layer
            h_conv2 = self.max_pool(tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2))

            # third conv-relu layer
            h_conv3 = self.max_pool(tf.nn.relu(self.conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3))

            # first fully connected layer
            h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_conv3, [-1, 1024]), self.W_fc1) + self.b_fc1)

            # last layer
            self.action_values = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

            # this is the target value
            self.y = tf.placeholder(tf.float32, [None])

            # this is the action index
            self.a = tf.placeholder(tf.float32, [None, num_actions])

            zeroed = tf.mul(self.action_values, self.a)
            # compute loss
            self.actual_reward = tf.reduce_sum(zeroed, reduction_indices=1)
            self.loss = tf.reduce_mean(tf.square(self.y - self.actual_reward))
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train = tf.train.AdamOptimizer(initial_learning_rate).minimize(self.loss, global_step=global_step)

            # summary ops only for q network
            if network == None:
                # summary ops for tensorboard
                episode_reward = tf.Variable(0.)
                tf.scalar_summary("Episode Reward", episode_reward)
                episode_ave_max_q = tf.Variable(0.)
                tf.scalar_summary("Max Q Value", episode_ave_max_q)
                logged_epsilon = tf.Variable(0.)
                tf.scalar_summary("Epsilon", logged_epsilon)
                summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
                self.summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
                self.summary_ops = [summary_vars[i].assign(self.summary_placeholders[i]) for i in range(len(summary_vars))]

            # param assignment ops only for target network
            if network != None:
                self.copy_ops = []
                self.copy_ops.append(self.W_conv1.assign(network.W_conv1))
                self.copy_ops.append(self.b_conv1.assign(network.b_conv1))
                self.copy_ops.append(self.W_conv2.assign(network.W_conv2))
                self.copy_ops.append(self.b_conv2.assign(network.b_conv2))
                self.copy_ops.append(self.W_conv3.assign(network.W_conv3))
                self.copy_ops.append(self.b_conv3.assign(network.b_conv3))
                self.copy_ops.append(self.W_fc1.assign(network.W_fc1))
                self.copy_ops.append(self.b_fc1.assign(network.b_fc1))
                self.copy_ops.append(self.W_fc2.assign(network.W_fc2))
                self.copy_ops.append(self.b_fc2.assign(network.b_fc2))

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding = "VALID")

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def evaluate(self, sess, state):
        return self.action_values.eval(session = sess, feed_dict = {self.s: [state]})

    def learn(self, sess, action_list, state_list, target_reward_list, writer, summaries, timestep):
        sess.run(self.loss, feed_dict = {self.a: action_list,
                                         self.s: state_list,
                                         self.y: target_reward_list})

    def copy(self, sess, network):
        sess.run(self.copy_ops)

        # making sure the weight assignment has taken effect
        q_w_fc1 = self.W_fc1.eval(session = sess)
        t_w_fc1 = network.W_fc1.eval(session = sess)
        if np.array_equal(q_w_fc1, t_w_fc1):
            print "Nice! target network inherited parameter correctly"

def resize_input(obs):
    return cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def actorLearner(index, sess, q_network, target_network, saver, writer, summaries, lock, step_counter, num_actions, action_offset, env):
    # parameters to try when resuming
    # step_counter.set(3007322)
    # epsilon = 0.6

    epsilon = initial_epsilon

    # intialize starting state
    observation = env.reset()
    resized_obs = resize_input(observation)
    state = np.stack((resized_obs,resized_obs,resized_obs,resized_obs), axis = 2)

    state_list = []
    target_reward_list = []
    action_list = []
    episode_score = 0.0
    t_thread_counter = 0.0
    episode_max_q_value = 0.0
    episode_t = 0.0
    episode_index = 0

    final_epsilon = sample_final_epsilon()

    print "starting agent#%d, with final epsilon %0.4f" % (index, final_epsilon)
    time.sleep(3*index)

    while step_counter.get() < max_thread_step:
        step_counter.increment()
        t_thread_counter += 1.0
        episode_t += 1.0

        # anneal explorativeness
        epsilon -= (initial_epsilon - final_epsilon) / frames_anneal
        epsilon = max(epsilon, final_epsilon)

        # save new observation to state ( a state consists of 4 consecutive observations )
        state_list.append(state)

        # evaluate all action values at current state
        action_values = q_network.evaluate(sess, state)

        action = action_offset
        random_action = True
        if random.random() < epsilon:
            action += random.randrange(num_actions)
        else:
            random_action = False
            action += np.argmax(action_values)

        action_index = np.zeros([num_actions])
        action_index[action - action_offset] = 1
        action_list.append(action_index)
        
        observation, reward, done, info = env.step(action)
        # clip reward
        reward = np.clip(reward, -1, 1)
        episode_score += reward
        if done:
            target_reward_list.append(reward)
        else:
            # evaluate the loss with the target network
            state = np.append(np.delete(state, 0, axis=2), np.reshape(resize_input(observation), (80,80,1)), axis=2)
            max_q = np.max(target_network.evaluate(sess, state))
            episode_max_q_value += max_q
            target_reward_list.append(reward + reward_decay * max_q)

        if done or (t_thread_counter % model_update_tsteps == 0):
            q_network.learn(sess, action_list, state_list, target_reward_list, writer, summaries, step_counter.get())
            state_list = []
            target_reward_list = []
            action_list = []

        if t_thread_counter % 500 == 0:
            if random_action:
                print "thread #%d takes random action #%d\n" % (index, action)
            else:
                print "thread #%d takes planned action #%d\n" % (index, action)
            print action_values

        # checkpoint
        if step_counter.get() % 500 == 0:
            saver.save(sess, "/tmp/qmodel/saved_network")

        if done:
            avg_max_q = episode_max_q_value / episode_t
            sess.run(q_network.summary_ops[0], feed_dict = {q_network.summary_placeholders[0]:float(episode_score)})
            sess.run(q_network.summary_ops[1], feed_dict = {q_network.summary_placeholders[1]:float(avg_max_q)})
            sess.run(q_network.summary_ops[2], feed_dict = {q_network.summary_placeholders[2]:float(epsilon)})
            # write summaries
            result = sess.run(summaries)
            writer.add_summary(result, step_counter.get())

            print "thread #%d\nepisode #%d\nepisode steps: #%d\nepisode score: %d\naverage max q:%s\ntimestep: %d\nthread timestep: %d\nepsilon: %s\n" % (index, episode_index, episode_t, episode_score, str(avg_max_q), step_counter.get(), t_thread_counter, str(epsilon))
            episode_t = 0.0
            episode_score = 0.0
            episode_max_q_value = 0.0
            episode_index += 1
            observation = env.reset()
    print "thread #%d has quit" % index
def main(): 
    # initialize game environments
    envs = []
    for i in range(num_threads):
        env = gym.make(game)
        envs.append(env)

    num_actions = envs[0].action_space.n
    action_offset = 0
    if (game == "Pong-v0" or game == "Breakout-v0"):
        # Gym currently specifies 6 actions for pong and breakout when only 3 are needed
        num_actions = 3
        action_offset = 1

    # initialize lock
    lock = threading.Lock()

    # intialize global step counter
    step_counter = StepCounter()

    # initialize tensor flow
    sess = tf.InteractiveSession()
    #sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

    # initialize agent network
    q_network = AgentQNetwork(num_actions, None, 'q')

    # merge all the summaries and write them out to local path
    summaries = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/tensorboard_logs", sess.graph)

    # initialize target network
    target_network = AgentQNetwork(num_actions, q_network, 't')

    # initalize tensorflow variables
    sess.run(tf.initialize_all_variables())

    # initialize a saver
    saver = tf.train.Saver()

    # load checkpoint if there is any
    checkpoint = tf.train.get_checkpoint_state("/tmp/qmodel")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "successfully loaded checkpoint"

    # initial sync of target network parameters
    target_network.copy(sess, q_network)

    # spawn agent threads
    threads = list()
    for i in range(num_threads):
        t = threading.Thread(target=actorLearner, args=(i, sess, q_network, target_network, saver, writer, summaries, lock, step_counter, num_actions, action_offset, envs[i]))
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()

    last_target_copy_time = 0
    while True:
        now = time.time()
        #for env in envs:
        #    env.render()
        if step_counter.get() % update_target_tsteps == 0:
            if now - last_target_copy_time > 10:
                last_target_copy_time = now
                print "step#%d updating target network" % (step_counter.get())
                target_network.copy(sess, q_network)


    # Wait for all of them to finish
    for t in threads:
        t.join()

    sess.close()

    print "Done!!!!"

if __name__ == "__main__":
    main()
