import threading
import tensorflow as tf
import cv2
import sys
import numpy as np
import gym

# environment name
game = "Pong-v0"

# number of learning agents
num_threads = 2

# agent explorativeness
inital_epsilon = 1
final_epsilon = 0.1
frames_anneal = 1000000.0

# how often to train model
model_update_tsteps = 100

# how often to update target network
update_target_tsteps = 10000

# reward decay
reward_decay = 0.99

# global thread counter
g_episode_counter = 0
g_thread_counter = 0
max_thread_step = 5000000


class AgentQNetwork(object):
    def __init__(self, num_actions, prefix):
        # input layer
        self.s = tf.placeholder(tf.float32, [None, 84, 84, 4])

        # layer parameters
        self.W_conv1 = self.weight_variable([8, 8, 4, 32])
        self.b_conv1 = self.bias_variable([32])

        self.W_conv2 = self.weight_variable([4, 4, 32, 64])
        self.b_conv2 = self.bias_variable([64])

        self.W_conv3 = self.weight_variable([3, 3, 64, 64])
        self.b_conv3 = self.bias_variable([64])

        self.W_fc1 = self.weight_variable([11*11*64, 512])
        self.b_fc1 = self.bias_variable([512])

        self.W_fc2 = self.weight_variable([512, num_actions])
        self.b_fc2 = self.bias_variable([num_actions])

        # first conv-relu layer
        h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, 4) + self.b_conv1)

        # second conv-relu layer
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

        # third conv-relu layer
        self.h_conv3 = tf.nn.relu(self.conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

        # first fully connected layer
        h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(self.h_conv3, [-1, 11*11*64]), self.W_fc1) + self.b_fc1)

        # last layer
        self.action_values = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

        # this is the target value
        self.y = tf.placeholder(tf.float32, [None])

        # this is the action index
        self.a = tf.placeholder(tf.float32, [None, num_actions])

        # compute loss
        self.actual_reward = tf.reduce_sum(tf.mul(self.a, self.action_values), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - self.actual_reward))
        self.train = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01).minimize(self.loss)

        conv1_w_hist = tf.histogram_summary(prefix + "_conv1_w", self.W_conv1)
        conv1_b_hist = tf.histogram_summary(prefix + "_conv1_b", self.b_conv1)
        conv2_w_hist = tf.histogram_summary(prefix + "_conv2_w", self.W_conv2)
        conv2_b_hist = tf.histogram_summary(prefix + "_conv2_b", self.b_conv2)
        conv3_w_hist = tf.histogram_summary(prefix + "_conv3_w", self.W_conv3)
        conv3_b_hist = tf.histogram_summary(prefix + "_conv3_b", self.b_conv3)
        fc1_w_hist = tf.histogram_summary(prefix + "_fc1_w", self.W_fc1)
        fc1_b_hist = tf.histogram_summary(prefix + "_fc1_b", self.b_fc1)
        fc2_w_hist = tf.histogram_summary(prefix + "_fc2_w", self.W_fc2)
        fc2_b_hist = tf.histogram_summary(prefix + "_fc2_b", self.b_fc2)
        loss_summary = tf.scalar_summary(prefix + "_cost", self.loss)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def evaluate(self, sess, state):
        return self.action_values.eval(session = sess, feed_dict = {self.s: np.expand_dims(state, axis=0)})

    def learn(self, sess, action_list, state_list, target_reward_list, writer, summaries, timestep):
        self.train.run(session = sess, feed_dict = {self.a: action_list,
                                                    self.s: state_list,
                                                    self.y: target_reward_list})
        result = sess.run([summaries], feed_dict = {self.a: action_list,
                                                    self.s: state_list,
                                                    self.y: target_reward_list})
        writer.add_summary(result[0], timestep)

        loss = self.loss.eval(session = sess, feed_dict = {self.a: action_list,
                                                    self.s: state_list,
                                                    self.y: target_reward_list})
        print "loss: " + str(loss)

    def copy(self, sess, network):
        sess.run(self.W_conv1.assign(network.W_conv1))
        sess.run(self.b_conv1.assign(network.b_conv1))
        sess.run(self.W_conv2.assign(network.W_conv2))
        sess.run(self.b_conv2.assign(network.b_conv2))
        sess.run(self.W_conv3.assign(network.W_conv3))
        sess.run(self.b_conv3.assign(network.b_conv3))
        sess.run(self.W_fc1.assign(network.W_fc1))
        sess.run(self.b_fc1.assign(network.b_fc1))
        sess.run(self.W_fc2.assign(network.W_fc2))
        sess.run(self.b_fc2.assign(network.b_fc2))

        print "successfully updated target network parameters"
        # making sure the weight assignment has taken effect
        q_w_fc1 = self.W_fc1.eval(session = sess)
        t_w_fc1 = network.W_fc1.eval(session = sess)
        assert np.array_equal(q_w_fc1, t_w_fc1), "target network did not inherit parameter correctly"

def resize_input(obs):
    return cv2.cvtColor(cv2.resize(obs, (84, 84)), cv2.COLOR_BGR2BGRA)

def actorLearner(index, sess, q_network, target_network, saver, writer, summaries, lock):
    # create game environment
    env = gym.make(game)

    global g_thread_counter
    global g_episode_counter

    # initialize explorativeness
    epsilon = inital_epsilon

    # intialize starting state
    observation = env.reset()
    state_list = []
    target_reward_list = []
    action_list = []
    all_score = 0
    t_thread_counter = 0.0

    while g_thread_counter < max_thread_step:
        # increment global thread step counter
        g_thread_counter += 1
        t_thread_counter += 1.0

        # display
        env.render()

        # anneal explorativeness
        epsilon_decay = (t_thread_counter / frames_anneal) * (inital_epsilon - final_epsilon)
        epsilon -= epsilon_decay
        epsilon = max(epsilon, final_epsilon)

        # resize image to 84 by 84
        resized_obs = resize_input(observation)
        state_list.append(resized_obs)

        # evaluate all action values at current state
        action_values = q_network.evaluate(sess, resized_obs)
        # take a random action under epsilon prob
        action = env.action_space.sample()

        # take action with biggest reward
        rand = np.random.uniform(0,1)
        if rand > epsilon:
            action = np.argmax(action_values)

        action_index = np.zeros([env.action_space.n])
        action_index[action] = 1
        action_list.append(action_index)
        
        observation, reward, done, info = env.step(action)
        if reward != 0:
            print "immediate reward: %d" % reward
        all_score += reward
        if done:
            target_reward_list.append(reward)
        else:
            # evaluate the loss with the target network
            target_reward_list.append(reward + reward_decay * np.max(target_network.evaluate(sess, resize_input(observation))))

        if done or (t_thread_counter % model_update_tsteps == 0):
            q_network.learn(sess, action_list, state_list, target_reward_list, writer, summaries, g_thread_counter)
            state_list = []
            target_reward_list = []
            action_list = []

        if g_thread_counter % update_target_tsteps == 0:
            target_network.copy(sess, q_network)

        # checkpoint
        if g_thread_counter % 500 == 0:
            saver.save(sess, "/tmp/qmodel/saved_network")
            print "thread #%d successfully saved" % index

        # print diagnostics
        if g_thread_counter % 500 == 0:
            print "timestep: %d" % g_thread_counter
            print "episode: %d" % g_episode_counter
            print "thread #%d, epsilon: %s" % (index, str(epsilon))

        if done:
            g_episode_counter += 1
            print "thread #%d, episode#%d, total score: %d" % (index, g_episode_counter, all_score)
            all_score = 0
            observation = env.reset()
def main(): 
    # initialize game environment
    env = gym.make(game)

    # initialize lock
    lock = threading.Lock()

    # initialize tensor flow
    sess = tf.InteractiveSession()

    # initialize agent network
    q_network = AgentQNetwork(env.action_space.n, 'q')

    # merge all the summaries and write them out to local path
    summaries = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/tensorboard_logs", sess.graph)

    # initialize target network
    target_network = AgentQNetwork(env.action_space.n, 't')

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
        t = threading.Thread(target=actorLearner, args=(i, sess, q_network, target_network, saver, writer, summaries, lock))
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all of them to finish
    for t in threads:
        t.join()

    sess.close()

    print "Done!!!!"

if __name__ == "__main__":
    main()