""" Implementation of:
-- Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. (https://arxiv.org/abs/1801.01290)
-- Learning to Walk via Deep Reinforcement Learning (https://arxiv.org/abs/1812.11103) --- learnable alpha param
"""

from collections import deque
import time

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from algs.memory import Memory, SymmetricMemory
from algs.models import GaussianPolicy, Model
from algs.explore import DisagreementExploration
import logger


class SAC:
    def __init__(self, observation_shape, action_shape, *,
                    policy_hidden_sizes, q_hidden_sizes, value_hidden_sizes, alpha, discount, tau, lr, learn_alpha=True):
        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.rewards_ph = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.dones_ph = tf.placeholder(tf.float32, [None, 1], name='dones')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')

        # build graph
        # 1. networks
        value_function = Model('value_function', hidden_sizes=value_hidden_sizes, output_size=1)
        target_value_function = Model('target_value_function', hidden_sizes=value_hidden_sizes, output_size=1)
        q1 = Model('q1', hidden_sizes=q_hidden_sizes, output_size=1)
        q2 = Model('q2', hidden_sizes=q_hidden_sizes, output_size=1)
        policy = GaussianPolicy('policy', hidden_sizes=policy_hidden_sizes, output_size=2*action_shape[0])
        if learn_alpha:
            beta = tf.Variable(tf.zeros(1), trainable=True, dtype=tf.float32, name='beta')
            alpha = tf.exp(beta)

        # 2. loss
        #   q value target
        next_v = target_value_function(self.next_obs_ph)
        target_q = self.rewards_ph + discount * next_v * (1 - self.dones_ph)
        #   q values loss terms
        q1_at_memory_action = q1(tf.concat([self.obs_ph, self.actions_ph], 1))
        q2_at_memory_action = q2(tf.concat([self.obs_ph, self.actions_ph], 1))
        q1_loss = tf.losses.mean_squared_error(q1_at_memory_action, target_q)
        q2_loss = tf.losses.mean_squared_error(q2_at_memory_action, target_q)
        #   policy loss term
        self.actions, log_pis = policy(self.obs_ph)
        q_at_policy_action = tf.minimum(q1(tf.concat([self.obs_ph, self.actions], 1)), q2(tf.concat([self.obs_ph, self.actions], 1)))
        policy_loss = tf.reduce_mean(alpha * log_pis - q_at_policy_action)
        #   value function loss term
        v = value_function(self.obs_ph)
        target_v = q_at_policy_action - alpha * log_pis
        v_loss = tf.losses.mean_squared_error(v, target_v)
        #   alpha loss term
        if learn_alpha:
            target_entropy = - 1 * np.sum(action_shape)
            alpha_loss = tf.reduce_mean(- alpha * log_pis - alpha * target_entropy)

        # 3. update ops
        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
        policy_train_op = optimizer.minimize(policy_loss, var_list=policy.trainable_vars, global_step=tf.train.get_or_create_global_step())
        with tf.control_dependencies([policy_train_op]):
            v_train_op = optimizer.minimize(v_loss, var_list=value_function.trainable_vars)
            q1_train_op = optimizer.minimize(q1_loss, var_list=q1.trainable_vars)
            q2_train_op = optimizer.minimize(q2_loss, var_list=q2.trainable_vars)
        #   combined train ops
        self.train_ops = [v_train_op, q1_train_op, q2_train_op, policy_train_op]
        if learn_alpha:
            with tf.control_dependencies([q1_train_op, q2_train_op, v_train_op, policy_train_op]):
                alpha_train_op = optimizer.minimize(alpha_loss, var_list=[beta])
            self.train_ops += [alpha_train_op]

        # target value fn update
        self.target_update_ops = tf.group([tf.assign(target, (1 - tau) * target + tau * source) for target, source in \
                                            zip(target_value_function.trainable_vars, value_function.trainable_vars)])

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def get_actions(self, obs):
        actions = self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})
        return actions

    def get_action_value(self, obs, actions):
        return self.sess.run(self.q_value_at_memory_action, {self.obs_ph: obs, self.actions_ph: actions})

    def update_target_net(self):
        self.sess.run(self.target_update_ops)

    def train_step(self, batch):
        self.sess.run(self.train_ops, {self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards,
                                       self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})

def learn(env, seed, n_total_steps, max_episode_length, alg_args, args):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # unpack variables
    max_memory_size = alg_args.pop('max_memory_size', int(1e6))
    n_prefill_steps = alg_args.pop('n_prefill_steps', 1000)
    batch_size = alg_args.pop('batch_size', 256)
    global best_ep_length

    # setup tracking
    stats = {}
    episode_rewards = np.zeros((env.num_envs, 1), dtype=np.float32)
    episode_lengths = np.zeros((env.num_envs, 1), dtype=int)
    episode_rewards_history = deque(maxlen=100)
    episode_lengths_history = deque(maxlen=100)
    n_episodes = 0
    start_step = 1

    # set up agent
    memory = Memory(int(max_memory_size), env.observation_space.shape, env.action_space.shape)
    agent = SAC(env.observation_space.shape, env.action_space.shape, **alg_args)
    exploration = DisagreementExploration(env.observation_space.shape, env.action_space.shape)

    # initialize session, agent, saver
    sess = tf.get_default_session()
    agent.initialize(sess)
    exploration.initialize(sess)
    # backward compatible to models that used mpi_adam -- ie only save and restore non-optimizer vars
    if n_total_steps == 0:
        vars_to_restore = [i[0] for i in tf.train.list_variables(args.load_path)]
        restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in vars_to_restore}
        saver = tf.train.Saver(restore_dict)
    else:
        saver = tf.train.Saver()  # keep track of all vars for training; only use non-optim vars for eval
        best_saver = tf.train.Saver(max_to_keep=1)
    sess.graph.finalize()
    if args.load_path is not None:
        saver.restore(sess, args.load_path)
        start_step = sess.run(tf.train.get_global_step()) + 1
        print('Restoring parameters at step {} from: {}'.format(start_step - 1, args.load_path))

    # init memory and env for training
    memory.initialize(env, n_prefill_steps, training=(n_total_steps > 0), policy=agent if args.load_path else None)
    obs = env.reset()

    for t in range(start_step, n_total_steps + start_step):
        tic = time.time()

        # sample action -> step env -> store transition
        actions = agent.get_actions(obs)
        next_obs, r, done, _ = env.step(actions)
        r += exploration.get_exploration_bonus(obs, actions)
        memory.store_transition(obs, actions, r, done, next_obs)
        obs = next_obs

        # keep records
        episode_rewards += r
        episode_lengths += 1

        # end of episode -- when all envs are done or max_episode length is reached, reset
        if any(done):
            for d in np.nonzero(done)[0]:
                episode_rewards_history.append(float(episode_rewards[d]))
                episode_lengths_history.append(int(episode_lengths[d]))
                n_episodes += 1
                # reset counters
                episode_rewards[d] = 0
                episode_lengths[d] = 0

        # train
        batch = memory.sample(batch_size)
        agent.train_step(batch)
        agent.update_target_net()
        expl_loss = exploration.train(batch)

        # save
        if t % args.save_interval == 0:
            saver.save(sess, args.output_dir + '/agent.ckpt', global_step=tf.train.get_global_step())
            if best_ep_length < ep_lengths_mean:
                best_ep_length = ep_lengths_mean
                best_saver.save(sess, args.output_dir + '/best_agent.ckpt', global_step=tf.train.get_global_step())

        # log stats
        if t % args.log_interval == 0:
            toc = time.time()
            stats['timestep'] = t
            stats['episodes'] = n_episodes
            stats['steps_per_second'] = args.log_interval / (toc - tic)
            stats['expl_loss'] = expl_loss
            stats['avg_return'] = np.mean(episode_rewards_history)
            stats['std_return'] = np.std(episode_rewards_history)
            stats['avg_episode_length'] = np.mean(episode_lengths_history)
            stats['std_episode_length'] = np.std(episode_lengths_history)
            logger.save_csv(stats, args.output_dir + '/log.csv')
            print(tabulate(stats.items(), tablefmt='rst'))

    return agent


# --------------------
# defaults
# --------------------

def defaults(env_name=None):
    if env_name == 'L2M2019':
        return {'policy_hidden_sizes': (128, 128),
                'value_hidden_sizes': (128, 128),
                'q_hidden_sizes': (128, 128),
                'discount': 0.96,
                'tau': 0.01,
                'lr': 1e-3,
                'batch_size': 256,
                'max_memory_size': int(1e6),
                'n_prefill_steps': 1000,
                'alpha': 0.2,
                'learn_alpha': True}
    else:  # mujoco
        alpha = {
            'Ant-v2': 0.1,
            'HalfCheetah-v2': 0.2,
            'Hopper-v2': 0.2,
            'Humanoid-v2': 0.05,
            'Walker2d-v2': 0.2,
            }.get(env_name, 0.2)

        return {'policy_hidden_sizes': (128, 128),
                'value_hidden_sizes': (128, 128),
                'q_hidden_sizes': (128, 128),
                'discount': 0.99,
                'tau': 0.01,
                'lr': 1e-3,
                'batch_size': 256,
                'max_memory_size': int(1e6),
                'alpha': alpha,
                'learn_alpha': True}
