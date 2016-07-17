import threading
import tensorflow as tf
import cv2
import sys
import numpy as np
import random
import gym
import time
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
from skimage.transform import resize
from skimage.color import rgb2gray

# environment name
game = "Pong-v0"

# number of learning agents
num_threads = 1

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
max_thread_step = 80000000

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

def resize_input(obs):
    return resize(rgb2gray(obs), (84, 84))

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def actorLearner(index, sess, graph_ops, saver, writer, summary_ops, step_counter, num_actions, action_offset, env):
    # parameters to try when resuming
    # step_counter.set(3007322)
    # epsilon = 0.6

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]
    summary_placeholders, update_ops, summary_op = summary_ops

    epsilon = initial_epsilon

    # intialize starting state
    observation = env.reset()
    resized_obs = resize_input(observation)
    state = np.stack((resized_obs,resized_obs,resized_obs,resized_obs))
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
        action_values = q_values.eval(session = sess, feed_dict = {s : [state]})
        
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
            state = np.append(np.delete(state, 0, axis=0), np.expand_dims(resize_input(observation), axis=0), axis = 0)
            max_q = np.max(target_q_values.eval(session = sess, feed_dict = {st : [state]}))
            episode_max_q_value += max_q
            target_reward_list.append(reward + reward_decay * max_q)

        if done or (t_thread_counter % model_update_tsteps == 0):
            sess.run(grad_update, feed_dict = {y : target_reward_list,
                                                  a : action_list,
                                                  s : state_list})
            state_list = []
            target_reward_list = []
            action_list = []

        # checkpoint
        if step_counter.get() % 500 == 0:
            saver.save(sess, "/tmp/qmodel/saved_network")

        if step_counter.get() % 300 == 0:
            print action_values

        if done:
            avg_max_q = episode_max_q_value / episode_t
            
            stats = [episode_score, avg_max_q, epsilon]
            for i in range(len(stats)):
                sess.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})

            summary_str = sess.run(summary_op)
            writer.add_summary(summary_str, step_counter.get())

            print "thread #%d\nepisode #%d\nepisode steps: #%d\nepisode score: %d\naverage max q:%s\ntimestep: %d\nthread timestep: %d\nepsilon: %s\n" % (index, episode_index, episode_t, episode_score, str(avg_max_q), step_counter.get(), t_thread_counter, str(epsilon))
            episode_t = 0.0
            episode_score = 0.0
            episode_max_q_value = 0.0
            episode_index += 1
            observation = env.reset()
    print "thread #%d has quit" % index

def build_network(num_actions, agent_history_length, resized_width, resized_height):
  with tf.device("/cpu:0"):
    state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
    inputs = Input(shape=(agent_history_length, resized_width, resized_height))
    model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
    model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(model)
    model = Flatten()(model)
    model = Dense(output_dim=256, activation='relu')(model)
    q_values = Dense(output_dim=num_actions, activation='linear')(model)
    m = Model(input=inputs, output=q_values)
  return state, m

def build_graph(num_actions):
    # Create shared deep q network
    s, q_network = build_network(num_actions=num_actions, agent_history_length=4, resized_width=84, resized_height=84)
    network_params = q_network.trainable_weights
    q_values = q_network(s)

    # Create shared target network
    st, target_q_network = build_network(num_actions=num_actions, agent_history_length=4, resized_width=84, resized_height=84)
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(st)

    # Op for periodically updating target network with online network weights
    reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
    
    # Define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.mul(q_values, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - action_q_values))
    optimizer = tf.train.AdamOptimizer(initial_learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s" : s, 
                 "q_values" : q_values,
                 "st" : st, 
                 "target_q_values" : target_q_values,
                 "reset_target_network_params" : reset_target_network_params,
                 "a" : a,
                 "y" : y,
                 "grad_update" : grad_update}

    return graph_ops

# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Max Q Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    tf.scalar_summary("Epsilon", logged_epsilon)
    logged_T = tf.Variable(0.)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.merge_all_summaries()
    return summary_placeholders, update_ops, summary_op

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

    # intialize global step counter
    step_counter = StepCounter()

    # initialize tensor flow
    sess = tf.InteractiveSession()
    #sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

    # build computational graph
    graph_ops = build_graph(num_actions)
    reset_target = graph_ops["reset_target_network_params"]

    # setup summaries
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]
    writer = tf.train.SummaryWriter("/tmp/tensorboard_logs", sess.graph)

    # initalize tensorflow variables
    sess.run(tf.initialize_all_variables())

    # initialize a saver
    saver = tf.train.Saver()

    # load checkpoint if there is any
    checkpoint = tf.train.get_checkpoint_state("/tmp/qmodel")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "successfully loaded checkpoint"

    sess.run(reset_target)

    # spawn agent threads
    threads = list()
    for i in range(num_threads):
        t = threading.Thread(target=actorLearner, args=(i, sess, graph_ops, saver, writer, summary_ops, step_counter, num_actions, action_offset, envs[i]))
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
                sess.run(reset_target)


    # Wait for all of them to finish
    for t in threads:
        t.join()

    sess.close()

    print "Done!!!!"

if __name__ == "__main__":
    main()
