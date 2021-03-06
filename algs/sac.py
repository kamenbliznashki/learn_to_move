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
from algs.models import Model, GaussianPolicy, FlowPolicy
from algs.explore import DisagreementExploration
import logger

best_ep_length = float('-inf')

class SAC:
    def __init__(self, observation_shape, action_shape, v_tgt_field_size, *,
                    policy_hidden_sizes, q_hidden_sizes, value_hidden_sizes, alpha, discount, tau, lr, policy_lr,
                    n_sample_actions=1, learn_alpha=True, use_nf=True, loss_ord=1):
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
        policy_cls = FlowPolicy if use_nf else GaussianPolicy
        policy = policy_cls('policy', hidden_sizes=policy_hidden_sizes, output_size=action_shape[0],
                            output_kernel_initializer='zeros', output_bias_initializer='zeros')
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
        self.q_at_memory_action = tf.minimum(q1_at_memory_action, q2_at_memory_action)
        q1_loss = tf.reduce_mean(tf.norm(q1_at_memory_action - target_q, axis=1, ord=loss_ord), axis=0)
        q2_loss = tf.reduce_mean(tf.norm(q2_at_memory_action - target_q, axis=1, ord=loss_ord), axis=0)
        #   policy loss term
        actions, log_pis = policy(self.obs_ph)  # (n_samples, batch_size, action_dim) and (...,1)
        actions, log_pis = tf.squeeze(actions, 0), tf.squeeze(log_pis, 0)
        q_at_policy_action = tf.minimum(q1(tf.concat([self.obs_ph, actions], 1)), q2(tf.concat([self.obs_ph, actions], 1)))
        policy_loss = tf.reduce_mean(alpha * log_pis - q_at_policy_action)
        #   value function loss term
        v = value_function(self.obs_ph)
        target_v = q_at_policy_action - alpha * log_pis
        v_loss = tf.reduce_mean(tf.norm(v - target_v, axis=1, ord=loss_ord), axis=0)
        #   alpha loss term
        if learn_alpha:
            target_entropy = - 1 * np.sum(action_shape)
            alpha_loss = tf.reduce_mean(- alpha * log_pis - alpha * target_entropy)

        # 3. update ops
        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
        policy_optimizer = tf.train.AdamOptimizer(policy_lr, name='policy_optimizer')
        policy_train_op = policy_optimizer.minimize(policy_loss, var_list=policy.trainable_vars, global_step=tf.train.get_or_create_global_step())
        self.train_ops = [policy_train_op]
        if use_nf:
            with tf.control_dependencies([policy_train_op]):
                flow_train_op = policy_optimizer.minimize(policy_loss, var_list=policy.flow_trainable_vars)
                self.train_ops += [flow_train_op]
        with tf.control_dependencies([policy_train_op]):
            v_train_op = optimizer.minimize(v_loss, var_list=value_function.trainable_vars)
            q1_train_op = optimizer.minimize(q1_loss, var_list=q1.trainable_vars)
            q2_train_op = optimizer.minimize(q2_loss, var_list=q2.trainable_vars)
        #   combined train ops
        self.train_ops += [v_train_op, q1_train_op, q2_train_op]
        if learn_alpha:
            with tf.control_dependencies(self.train_ops):
                alpha_train_op = optimizer.minimize(alpha_loss, var_list=[beta])
            self.train_ops += [alpha_train_op]

        #   target value fn update
        self.target_update_ops = tf.group([tf.assign(target, (1 - tau) * target + tau * source) for target, source in \
                                            zip(target_value_function.trainable_vars, value_function.trainable_vars)])

        # 4. get action op
        self.actions, _ = policy(self.obs_ph, n_sample_actions)

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def get_actions(self, obs):
        return self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})

    def get_action_value(self, obs, actions):
        return self.sess.run(self.q_at_memory_action, {self.obs_ph: obs, self.actions_ph: actions})

    def update_target_net(self):
        self.sess.run(self.target_update_ops)

    def train_step(self, batch):
        self.sess.run(self.train_ops, {self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards,
                                       self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})

def learn(env, exploration, seed, n_total_steps, max_episode_length, alg_args, args):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # unpack variables
    sym_memory = alg_args.pop('sym_memory', False)
    max_memory_size = alg_args.pop('max_memory_size', int(1e6))
    n_prefill_steps = alg_args.pop('n_prefill_steps', 1000)
    batch_size = alg_args.pop('batch_size', 256)
    n_train_steps_per_env_step = alg_args.pop('n_train_steps_per_env_step', 1)
    global best_ep_length

    # setup tracking
    stats = {}
    episode_rewards = np.zeros((env.num_envs, 1), dtype=np.float32)
    episode_aug_rew = np.zeros((env.num_envs, 1), dtype=np.float32)
    episode_bonus   = np.zeros((env.num_envs, 1), dtype=np.float32)
    episode_lengths = np.zeros((env.num_envs, 1), dtype=int)
    episode_rewards_history = deque(maxlen=100)
    episode_aug_rew_history = deque(maxlen=100)
    episode_bonus_history   = deque(maxlen=100)
    episode_lengths_history = deque(maxlen=100)
    n_episodes = 0
    start_step = 1

    # set up agent
    memory_cls = SymmetricMemory if sym_memory else Memory
    memory = memory_cls(int(max_memory_size), env.observation_space.shape, env.action_space.shape)
    agent = SAC(env.observation_space.shape, env.action_space.shape, env.v_tgt_field_size, **alg_args)

    # initialize session, agent, saver
    sess = tf.get_default_session()
    agent.initialize(sess)
    if exploration is not None: exploration.initialize(sess, env=env)
    # backward compatible to models that used mpi_adam -- ie only save and restore non-optimizer vars
    if n_total_steps == 0 and args.load_path is not None:
        vars_to_restore = [i[0] for i in tf.train.list_variables(args.load_path)]
        restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in vars_to_restore}
        saver = tf.train.Saver(restore_dict)
    else:
        saver = tf.train.Saver()  # keep track of all vars for training; only use non-optim vars for eval
        best_saver = tf.train.Saver(max_to_keep=2)
    sess.graph.finalize()
    if args.load_path is not None:
        saver.restore(sess, args.load_path)
        start_step = sess.run(tf.train.get_global_step()) + 1
        env.anneal_step = start_step  # annealing of init pose env
        print('Restoring parameters at step {} from: {}'.format(start_step - 1, args.load_path))

    # init memory and env for training
    memory.initialize(env, n_prefill_steps // env.num_envs, training=(n_total_steps > 0), policy=agent if args.load_path else None)
    obs = env.reset()

    for t in range(start_step, n_total_steps + start_step):
        tic = time.time()

        # sample action -> step env -> store transition
        actions = agent.get_actions(obs)  # (n_samples, batch_size, action_dim)
        actions = exploration.select_best_action(obs, actions) if exploration is not None else actions
        next_obs, r, done, info = env.step(actions)
        r_aug = np.vstack([i.get('rewards', 0) for i in info])
        r_bonus = exploration.get_exploration_bonus(obs, actions, next_obs) if exploration is not None else 0
        done_bool = np.where(episode_lengths + 1 == max_episode_length, np.zeros_like(done), done)  # only store true `done` in buffer not episode ends
        memory.store_transition(obs, actions, r + r_bonus + r_aug, done_bool, next_obs)
        obs = next_obs

        # keep records
        episode_rewards += r
        episode_aug_rew += r_aug
        episode_bonus   += r_bonus
        episode_lengths += 1

        # end of episode -- when all envs are done or max_episode length is reached, reset
        if any(done):
            for d in np.nonzero(done)[0]:
                episode_rewards_history.append(float(episode_rewards[d]))
                episode_aug_rew_history.append(float(episode_aug_rew[d]))
                episode_bonus_history.append(float(episode_bonus[d]))
                episode_lengths_history.append(int(episode_lengths[d]))
                n_episodes += 1
                # reset counters
                episode_rewards[d] = 0
                episode_aug_rew[d] = 0
                episode_bonus[d] = 0
                episode_lengths[d] = 0

        # train
        for _ in range(n_train_steps_per_env_step):
            batch = memory.sample(batch_size)
            agent.train_step(batch)
            agent.update_target_net()
        expl_loss = exploration.train(memory, batch_size)

        # save
        if t % args.save_interval == 0:
            saver.save(sess, args.output_dir + '/agent.ckpt', global_step=tf.train.get_global_step())
            if best_ep_length <= np.mean(episode_lengths_history):
                best_ep_length = np.mean(episode_lengths_history)
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
            stats['avg_aug_return'] = np.mean(episode_aug_rew_history)
            stats['std_aug_return'] = np.std(episode_aug_rew_history)
            stats['avg_bonus'] = np.mean(episode_bonus_history)
            stats['std_bonus'] = np.std(episode_bonus_history)
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
        return {'policy_hidden_sizes': (256, 256),
                'value_hidden_sizes': (256, 256),
                'q_hidden_sizes': (256, 256),
                'discount': 0.96,
                'tau': 0.01,
                'lr': 1e-3,
                'policy_lr': 1e-2,
                'batch_size': 256,
                'n_train_steps_per_env_step': 1,
                'sym_memory': False,
                'max_memory_size': int(1e6),
                'n_prefill_steps': 1000,
                'alpha': 0.2,
                'learn_alpha': True,
                'use_nf': True,
                'loss_ord': 1,
                'n_sample_actions': 10}
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
                'n_train_steps_per_env_step': 1,
                'max_memory_size': int(1e6),
                'alpha': alpha,
                'learn_alpha': True,
                'use_nf': True,
                'loss_ord': 1}
