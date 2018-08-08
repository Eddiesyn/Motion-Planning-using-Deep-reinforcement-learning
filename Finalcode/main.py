"""
Environment is a Robot Arm. The arm tries to get to the blue point.
The environment will return a geographic (distance) information for the arm to learn.

The far away from blue point the less reward; touch blue r+=1; stop at blue for a while then get r=+10.
 
You can train this RL by using LOAD = False, after training, this model will be store in the a local folder.
Using LOAD = True to reload the trained model for playing.

You can customize this script in a way you want.

View more on [Python] : https://morvanzhou.github.io/tutorials/

Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
tensorflow >= 1.0.1
"""

import tensorflow as tf
import numpy as np
import os
import shutil
import argparse
import time
#from arm_env import ArmEnv
from ENV import Env
from Marker import marker
from Speek import speek
import signal

# np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 10000
MAX_EP_STEPS = 200
LR_A = 1e-3  # learning rate for actor
LR_C = 1e-5  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 7000
BATCH_SIZE = 200
VAR_MIN = 0.01
RENDER = True
LOAD = True
MODE = ['easy', 'hard']
n_model = 0

#env = ArmEnv(mode=MODE[n_model])
mar = marker()
env = Env()
speek = speek()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound


# Parameters  
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default= 'True',
                     help='Load checkpoints or not' )
parser.add_argument('--vrep', type=bool, default=True,
                     help='Use vrep or not' )
parser.add_argument('--target_x', type=float, default= 33,
                     help='target position_x' )
parser.add_argument('--target_y', type=float, default= 5.4,
                     help='target position_y' )
args = parser.parse_args(['--load','True'])

# reward file
# f_re = open("reward/reward.txt","w+")
if not LOAD:
    with open("reward/reward.txt","w+") as f_re:
        pass

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


# def sigint_handler(signum, frame):
#     speek.sayingBye()
#     print('system terminate')


# signal.signal(signal.SIGINT, sigint_handler)
class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
            tf.summary.scalar('Critic_loss', self.loss)
            self.merged = tf.summary.merge_all()

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        outputs = self.sess.run([self.merged, self.train_op], feed_dict={S: s, self.a: a, R: r, S_: s_})
        self.summary = outputs[0]
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement=True
# sess = tf.Session(config=config)
sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './'+MODE[n_model]

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    saver.restore(sess, tf.train.latest_checkpoint(path))
    # sess.run(tf.global_variables_initializer())


def train():
    var = 3.  # control exploration
    reward_list = []
    board_writer = tf.summary.FileWriter(
        'summary'+ "/" + str(int(time.time())), 
        sess.graph
        )


    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for t in range(MAX_EP_STEPS):
        # while True:
            #if RENDER:
            #    env.render()
            #print('initial state: ', s )

            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
            s_, r, done = env.step(a)
            # print('it: ', t, 'action: ', a, 'reward: %.2f', r)
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                print('it: ', t, '--learn--')
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)
                #merged = tf.summary.merge_all()
                summary = critic.summary
                board_writer.add_summary(summary, t + ep*MAX_EP_STEPS)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS-1 or done:
            # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )
                break

        tf.summary.scalar('reward', ep_reward)
        # merged = tf.summary.merge_all()
        # summary = sess.run(merged)
        # board_writer.add_summary(summary, t + ep*MAX_EP_STEPS)
        reward_list.append(ep_reward)
        # print('reward: %.2f'%(float(reward) for reward in reward_list))
        print('< reward_list >: ', reward_list)
        

        if ep%10 ==0 :
            if os.path.isdir(path): shutil.rmtree(path)
            os.mkdir(path)
            ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
            save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
            print("\nSave Model %s\n" % save_path)
        # f_re.write("%.6f\r\n" % (reward_list[-1]))

        with open("reward/reward.txt","a+") as f_re:
        	f_re.write("%.6f, " % (reward_list[-1]))

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)

    # f_re.close()


def eval():
   # env.set_fps(30)
    s = env.reset()
    while True:
        # if RENDER:
        #     env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)
        s = s_

def eval_count():
    test_num = 100
    succ_count = 0
    succ_point = []
    fail_point = []
    # mar.StartDetecting()
    for i in range(test_num):
        time.sleep(1)
        # speek.sayingReset()
        s = env.reset()

        mar.StartDetecting()
        point = env.point_info.copy()        # env.point_info = np.array([33, 10
        set_target = np.array([args.target_x, args.target_y, -25])
        # env.point_info = np.array([33, 10,-25])
        target = np.array([-mar.T[0]+ 20, mar.T[1], -25])
        env.point_info = target
        print('target_pos: ', env.point_info, 'set_target: ', set_target )
        speek.sayingSearch()
        for j in range(20):
            # if (30< env.point_info[0] < 40) & (20<env.point_info[1] < 30):
            #     env.point_info += np.array([2, -1, 0])
            # mar.StartDetecting()
            a = actor.choose_action(s)
            # mar.StartDetecting()
            s_, r, done = env.step(a)
            mar.StartDetecting()
            
            if done :#& (not collision):
                succ_count += 1
                succ_point.append(point)
                print('After %i step Poppy succeed' % (j+1))
                break
            if mar.Flag == True:
                print('After %i step Poppy succeed' % (j+1))
                speek.sayingFind()
                time.sleep(1.5)
                speek.sayingBye()
                break
            else:
                s = s_

        if j == 19:
            speek.sayingNo()
    
        if not done:
            fail_point.append(point) 

        print('succ_point: ', succ_point)
        print('fail_point: ', fail_point)
        print('success cases/total test : %i/%i '%( succ_count , i+1))

    print('success cases/total test : %i/%i '%( succ_count , test_num))

def eval_test():
    s = env.reset()
    point = env.point_info.copy()
    for j in range(70):
        # if RENDER:
            #     env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)

        if done:
            print('After %i step Poppy succeed' % (j+1))
            break
        s = s_

    s = env.reset()



if __name__ == '__main__':
    a = True if  args.load == 'True' else False
    print('LOAD: ', LOAD, 'args.load: ', args.load, 'a: ', a )
    if LOAD:
        print('### Begin evaluation ###')
        # eval()
        eval_count()
        # eval_test()
    else:
        print('### Begin train ###')
        train()
