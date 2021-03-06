from collections import deque
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tabulate import tabulate

from algs.memory import Memory
from algs.models import GaussianPolicy, Model
from algs.explore import DisagreementExploration
import logger

try:
    from mpi4py import MPI
    from mpi_adam import MpiAdam, flatgrad
    from mpi_utils import mpi_moments
except ImportError:
    MPI = None

best_ep_length = float('-inf')

class DDPG:
    def __init__(self, observation_shape, action_shape, min_action, max_action, *,
                    policy_hidden_sizes, q_hidden_sizes, discount, tau, q_lr, policy_lr, expl_noise, n_sample_actions):
        self.min_action = min_action
        self.max_action = max_action
        self.tau = tau
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.expl_noise = expl_noise
        self.n_sample_actions = n_sample_actions

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.rewards_ph = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.dones_ph = tf.placeholder(tf.float32, [None, 1], name='dones')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')

        # build graph
        # 1. networks
        self.q = Model('q', hidden_sizes=q_hidden_sizes, activation=tf.nn.selu, output_size=1)
        q_target = Model('q_target', hidden_sizes=q_hidden_sizes, activation=tf.nn.selu, output_size=1)
        self.policy = Model('policy', hidden_sizes=policy_hidden_sizes, activation=tf.tanh, output_size=action_shape[0],
                            output_activation=tf.tanh)
        policy_target = Model('policy_target', hidden_sizes=policy_hidden_sizes, activation=tf.tanh,
                                output_size=action_shape[0], output_activation=tf.tanh)

        # current q values
        self.q_value = self.q(tf.concat([self.obs_ph, self.actions_ph], 1))
        # q values at policy action
        self.actions = max_action * self.policy(self.obs_ph)
        q_value_at_policy_action = self.q(tf.concat([self.obs_ph, self.actions], 1))

        # select next action according to the policy_target
        next_actions = policy_target(self.next_obs_ph)
        # compute q targets
        q_target_value = q_target(tf.concat([self.next_obs_ph, next_actions], 1))
        q_target_value = self.rewards_ph + tf.stop_gradient(discount * q_target_value * (1 - self.dones_ph))

        # 2. loss on critics and actor
        self.q_loss = tf.losses.mean_squared_error(self.q_value, q_target_value)
        self.policy_loss = - tf.reduce_mean(q_value_at_policy_action)

        # 3. training
        self.q_optimizer = tf.train.AdamOptimizer(q_lr, name='q_optimizer')
        self.policy_optimizer = tf.train.AdamOptimizer(policy_lr, name='policy_optimizer')
        self.update_global_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
        self.train_ops = [self.q_optimizer.minimize(self.q_loss, var_list=self.q.trainable_vars),
                          self.policy_optimizer.minimize(self.policy_loss, var_list=self.policy.trainable_vars),
                          self.update_global_step]
        #   target update ops
        self.target_update_ops = tf.group(self.create_target_update_op(self.q, q_target) +
                                          self.create_target_update_op(self.policy, policy_target))

        # init target networks
        self.target_init_ops = tf.group(self.create_target_init_op(self.q, q_target) +
                                        self.create_target_init_op(self.policy, policy_target))

    def create_target_init_op(self, source, target):
        return [tf.assign(t, s) for t, s in zip(target.vars, source.vars)]

    def create_target_update_op(self, source, target):
        return [tf.assign(t, (1 - self.tau) * t + self.tau * s) for t, s in zip(target.vars, source.vars)]

    def initialize(self, sess):
        tf.train.get_or_create_global_step()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_ops)

    def get_actions(self, obs):
        actions = self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})
        if self.expl_noise != 0:
            actions = actions + np.random.normal(0, self.expl_noise, (self.n_sample_actions, *actions.shape))
            actions = np.clip(actions, self.min_action, self.max_action)
        return actions

    def get_action_value(self, obs, actions):
        return self.sess.run(self.q_value, {self.obs_ph: obs, self.actions_ph: actions})

    def update_target_net(self):
        self.sess.run(self.target_update_ops)

    def train(self, batch):
        self.sess.run(self.train_ops,
                feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards,
                           self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})

class DDPGMPI(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # overwrite train ops
        self.q_grads = flatgrad(self.q_loss, self.q.trainable_vars)
        self.policy_grads = flatgrad(self.policy_loss, self.policy.trainable_vars)
        world_size = MPI.COMM_WORLD.Get_size()
        self.q_optimizer = MpiAdam(var_list=self.q.trainable_vars, scale_grad_by_procs=False)
        self.policy_optimizer = MpiAdam(var_list=self.policy.trainable_vars, scale_grad_by_procs=False)
        self.update_global_step = tf.assign_add(tf.train.get_or_create_global_step(), world_size)
        self.train_ops = [self.policy_grads, self.policy_loss, self.q_grads, self.q_loss, self.update_global_step]

    def initialize(self, sess):
        super().initialize(sess)
        self.policy_optimizer.sync()
        self.q_optimizer.sync()
        self.sess.run(self.target_init_ops)

    def get_actions(self, obs):
        actions = self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})
        if self.expl_noise != 0:
            actions = actions + np.random.normal(0, self.expl_noise, (self.n_sample_actions, *actions.shape))
            actions = np.clip(actions, self.min_action, self.max_action)
        return actions

    def get_action_value(self, obs, actions):
        return self.sess.run(self.q_value, {self.obs_ph: obs, self.actions_ph: actions})

    def train(self, batch):
        policy_grads, _, q_grads, _, _ = self.sess.run(self.train_ops,
                feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards,
                           self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})
        self.policy_optimizer.update(policy_grads, stepsize=self.policy_lr)
        self.q_optimizer.update(q_grads, stepsize=self.q_lr)

def learn(env, exploration, seed, n_total_steps, max_episode_length, alg_args, args):

    np.random.seed(int(seed + 1e6*args.rank))
    tf.set_random_seed(int(seed + 1e6*args.rank))

    # unpack variables
    batch_size = alg_args.pop('batch_size')
    sym_memory = alg_args.pop('sym_memory', False)
    max_memory_size = alg_args.pop('max_memory_size')
    n_prefill_steps = alg_args.pop('n_prefill_steps')
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
    ep_lengths_mean = 0
    start_step = 1

    # set up agent
    memory_cls = SymmetricMemory if sym_memory else Memory
    memory = memory_cls(int(max_memory_size), env.observation_space.shape, env.action_space.shape)
    agent = DDPG if MPI is None else DDPGMPI
    agent = agent(env.observation_space.shape, env.action_space.shape, env.action_space.low[0], env.action_space.high[0], **alg_args)

    # initialize session, agent, saver
    sess = tf.get_default_session()
    agent.initialize(sess)
    if exploration is not None: exploration.initialize(sess, env=env)
    # compatibility with models trained with mpi_adam - keep track of all vars for training; only use non-optim vars for eval
    if n_total_steps == 0 and args.load_path is not None:
        vars_to_restore = [i[0] for i in tf.train.list_variables(args.load_path)]
        restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in vars_to_restore}
        saver = tf.train.Saver(restore_dict)
    else:
        saver = tf.train.Saver()
        best_saver = tf.train.Saver(max_to_keep=2)
    sess.graph.finalize()
    if args.load_path is not None:
        saver.restore(sess, args.load_path)
        start_step = sess.run(tf.train.get_global_step()) + 1
        env.anneal_step = start_step  # annealing of init pose env
        print('Restoring parameters at step {} from: {}'.format(start_step - 1, args.load_path))

    # init memory and env
    memory.initialize(env, n_prefill_steps, training=(n_total_steps > 0), policy=agent if args.load_path else None)
    obs = env.reset()

    for t in range(start_step, n_total_steps + start_step):
        tic = time.time()

        # sample action -> step env -> store transition
        actions = agent.get_actions(obs)
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
            agent.train(batch)
            agent.update_target_net()
        expl_loss = exploration.train(memory, batch_size)

        # save
        if t % args.save_interval == 0 and args.rank == 0:
            saver.save(sess, args.output_dir + '/agent.ckpt', global_step=tf.train.get_global_step())
            if best_ep_length <= ep_lengths_mean:
                best_ep_length = ep_lengths_mean
                best_saver.save(sess, args.output_dir + '/best_agent.ckpt', global_step=tf.train.get_global_step())

        # log stats
        if t % args.log_interval == 0:
            if MPI is not None:
                ep_rewards_mean, ep_rewards_std, _ = mpi_moments(episode_rewards_history)
                ep_aug_rew_mean, ep_aug_rew_std, _ = mpi_moments(episode_aug_rew_history)
                ep_bonus_mean, ep_bonus_std, _  = mpi_moments(episode_bonus_history)
                ep_lengths_mean, ep_lengths_std, _ = mpi_moments(episode_lengths_history)
                n_episodes_mean, _, n_episodes_count = mpi_moments([n_episodes])
                episodes_count = n_episodes_mean * n_episodes_count
            else:
                ep_rewards_mean = np.mean(episode_rewards_history)
                ep_rewards_std  = np.std(episode_rewards_history)
                ep_aug_rew_mean = np.mean(episode_aug_rew_history)
                ep_aug_rew_std  = np.std(episode_aug_rew_history)
                ep_bonus_mean   = np.mean(episode_bonus_history)
                ep_bonus_std    = np.std(episode_bonus_history)
                ep_lengths_mean = np.mean(episode_lengths_history)
                ep_lengths_std  = np.std(episode_lengths_history)
                episodes_count  = n_episodes

            toc = time.time()
            stats['timestep'] = args.world_size * t
            stats['episodes'] = episodes_count
            stats['steps_per_second'] = args.world_size * args.log_interval / (toc - tic)
            stats['avg_return'] = ep_rewards_mean
            stats['std_return'] = ep_rewards_std
            stats['avg_aug_return'] = ep_aug_rew_mean
            stats['std_aug_return'] = ep_aug_rew_std
            stats['avg_bonus'] = ep_bonus_mean
            stats['std_bonus'] = ep_bonus_std
            stats['avg_episode_length'] = ep_lengths_mean
            stats['std_episode_length'] = ep_lengths_std
            if args.rank == 0:
                logger.save_csv(stats, args.output_dir + '/log.csv')
                print(tabulate(stats.items(), tablefmt='rst'))

        # different threads have different seeds
        if MPI is not None:
            local_uniform = np.random.uniform(size=(1,))
            root_uniform = local_uniform.copy()
            MPI.COMM_WORLD.Bcast(root_uniform, root=0)
            if args.rank != 0:
                assert local_uniform[0] != root_uniform[0], '{} vs {}'.format(local_uniform[0], root_uniform[0])

    return agent


# --------------------
# defaults
# --------------------

def defaults(env_name=None):
    if env_name == 'L2M2019':
        return {'policy_hidden_sizes': (256, 256),
                'q_hidden_sizes': (256, 256),
                'discount': 0.96,
                'tau': 0.005,
                'q_lr': 1e-3,
                'policy_lr': 1e-3,
                'batch_size': 128,
                'n_train_steps_per_env_step': 1,
                'sym_memory': False,
                'max_memory_size': int(1e6),
                'n_prefill_steps': 1000,
                'expl_noise': 0.2,
                'n_sample_actions': 10}
    else:  # mujoco
        n_prefill_steps = {
            'Ant-v2': 10000,
            'HalfCheetah-v2': 10000,
            'Hopper-v2': 1000,
            'Humanoid-v2': 1000,
            'Walker2d-v2': 1000,
            }.get(env_name, 1000)

        return {'policy_hidden_sizes': (400, 300),
                'q_hidden_sizes': (400, 300),
                'expl_noise': 0.1,
                'discount': 0.99,
                'tau': 0.005,
                'q_lr': 1e-3,
                'policy_lr': 1e-3,
                'batch_size': 64,
                'n_train_steps_per_env_step': 1,
                'sym_memory': False,
                'max_memory_size': int(1e6),
                'n_prefill_steps': n_prefill_steps}

