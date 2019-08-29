from collections import deque
import time

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from mpi4py import MPI

from algs.memory import Memory, SymmetricMemory
from algs.models import GaussianPolicy, Model
from mpi_adam import MpiAdam, flatgrad
from mpi_utils import mpi_moments
import logger


class BootstrappedDDPG:
    def __init__(self, observation_shape, action_shape, min_action, max_action, n_heads, alg_args):
        # NOTE TODO -- can adjust alg_args across heads here e.g.: expl noise, discount, learning rate -- e.g. uniform sample
        del alg_args['discount'], alg_args['expl_noise']

        self.heads = [DDPG(i, observation_shape, action_shape, min_action, max_action, 
                           discount=np.random.uniform(0.95, 0.999),
                           expl_noise=np.random.uniform(0, 0.2),
                           **alg_args) for i in range(n_heads)]

    def get_actions(self, obs, head_idx=None):
        if head_idx is not None:
            # specific head acts
            actions = self.heads[head_idx].get_actions(obs)
        else:
            # ensemble voting -- get actions and q_values for each head, select actions with max q value
            actions = np.stack([head.get_actions(obs) for head in self.heads], 0)                          # (n_heads, n_env, action_dim)
            qs = np.stack([self.heads[i].get_q_values(obs, actions[i]) for i in range(len(self.heads))], 0) # (n_heads, n_env, 1)
            # normalize q values across the heads
            qs /= qs.sum(0, keepdims=True)
            # select action at highest q value
            actions = np.take_along_axis(actions, qs.argmax(0)[None,...], axis=0)
        return actions

    def initialize(self, sess):
        # setup global step
        world_size = MPI.COMM_WORLD.Get_size()
        self.global_step_update_op = tf.assign_add(tf.train.get_or_create_global_step(), world_size)
        # initialize all heads
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        for head in self.heads:
            head.initialize(sess)

    def train(self, memory, batch_size):
        for head in self.heads:
            batch = memory.sample(batch_size)
            head.train(batch)
        # update global step
        self.sess.run(self.global_step_update_op)

    def update_target_net(self):
        for head in self.heads:
            head.update_target_net()


class DDPG:
    def __init__(self, head_idx, observation_shape, action_shape, min_action, max_action, *,
                    policy_hidden_sizes, q_hidden_sizes, discount, tau, q_lr, policy_lr, expl_noise):
        self.min_action = min_action
        self.max_action = max_action
        self.tau = tau
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.expl_noise = expl_noise

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.rewards_ph = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.dones_ph = tf.placeholder(tf.float32, [None, 1], name='dones')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')

        # build graph
        # 1. networks
        q = Model('q{}'.format(head_idx), hidden_sizes=q_hidden_sizes, activation=tf.nn.selu, output_size=1)
        q_target = Model('q{}_target'.format(head_idx), hidden_sizes=q_hidden_sizes, activation=tf.nn.selu, output_size=1)
        policy = Model('policy{}'.format(head_idx), hidden_sizes=policy_hidden_sizes, activation=tf.tanh, output_size=action_shape[0],
                            output_activation=tf.tanh)
        policy_target = Model('policy{}_target'.format(head_idx), hidden_sizes=policy_hidden_sizes, activation=tf.tanh,
                                output_size=action_shape[0], output_activation=tf.tanh)

        # current q values
        self.q_value = q(tf.concat([self.obs_ph, self.actions_ph], 1))
        # q values at policy action
        self.actions = max_action * policy(self.obs_ph)
        q_value_at_policy_action = q(tf.concat([self.obs_ph, self.actions], 1))

        # select next action according to the policy_target
        next_actions = policy_target(self.next_obs_ph)
        # compute q targets
        q_target_value = q_target(tf.concat([self.next_obs_ph, next_actions], 1))
        q_target_value = self.rewards_ph + tf.stop_gradient(discount * q_target_value * (1 - self.dones_ph))

        # 2. loss on critics and actor
        self.q_loss = tf.losses.mean_squared_error(self.q_value, q_target_value)
        self.policy_loss = - tf.reduce_mean(q_value_at_policy_action)

        # 3. training
        self.q_grads = flatgrad(self.q_loss, q.trainable_vars)
        self.q_optimizer = MpiAdam(var_list=q.vars, scale_grad_by_procs=False)
        self.policy_grads = flatgrad(self.policy_loss, policy.trainable_vars)
        self.policy_optimizer = MpiAdam(var_list=policy.vars, scale_grad_by_procs=False)
        #   target update ops
        self.target_update_ops = tf.group(self.create_target_update_op(q, q_target) +
                                          self.create_target_update_op(policy, policy_target))

        # init target networks
        self.target_init_ops = tf.group(self.create_target_init_op(q, q_target) +
                                        self.create_target_init_op(policy, policy_target))

    def create_target_init_op(self, source, target):
        return [tf.assign(t, s) for t, s in zip(target.vars, source.vars)]

    def create_target_update_op(self, source, target):
        return [tf.assign(t, (1 - self.tau) * t + self.tau * s) for t, s in zip(target.vars, source.vars)]

    def initialize(self, sess):
        tf.train.get_or_create_global_step()
        self.sess = sess
        self.policy_optimizer.sync()
        self.q_optimizer.sync()
        self.sess.run(self.target_init_ops)

    def get_actions(self, obs):
        actions = self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})
        if self.expl_noise != 0:
            actions += np.random.normal(0, self.expl_noise, actions.shape)
            actions = np.clip(actions, self.min_action, self.max_action)
        return actions

    def get_q_values(self, obs, actions):
        return self.sess.run(self.q_value, {self.obs_ph: np.atleast_2d(obs), self.actions_ph: np.atleast_2d(actions)})

    def update_target_net(self):
        self.sess.run(self.target_update_ops)

    def train(self, batch):
        policy_grads, _, q_grads, _ = self.sess.run(
                [self.policy_grads, self.policy_loss, self.q_grads, self.q_loss],
                feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards,
                           self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})
        self.policy_optimizer.update(policy_grads, stepsize=self.policy_lr)
        self.q_optimizer.update(q_grads, stepsize=self.q_lr)


def learn(env, seed, n_total_steps, max_episode_length, alg_args, args):
    # extract training and memory buffer args
    batch_size = alg_args.pop('batch_size')
    max_memory_size = alg_args.pop('max_memory_size')
    n_prefill_steps = alg_args.pop('n_prefill_steps')
    n_heads = alg_args.pop('n_heads')
    episode_length = alg_args.pop('episode_length', 1000)

    np.random.seed(int(seed + 1e6*args.rank))
    tf.set_random_seed(int(seed + 1e6*args.rank))

    memory = Memory(int(max_memory_size), env.observation_space.shape, env.action_space.shape)
    agent = BootstrappedDDPG(env.observation_space.shape, env.action_space.shape, env.action_space.low[0],
                             env.action_space.high[0], n_heads, alg_args)

    # initialize session, agent and memory
    sess = tf.get_default_session()
    agent.initialize(sess)
    saver = tf.train.Saver()
    sess.graph.finalize()
    memory.initialize(env, n_prefill_steps, training=not args.play)

    # setup tracking
    stats = {}
    episode_rewards = np.zeros((env.num_envs, 1), dtype=np.float32)
    episode_lengths = np.zeros((env.num_envs, 1), dtype=int)
    episode_rewards_history = deque(maxlen=100)
    episode_lengths_history = deque(maxlen=100)
    episodes_completed = 0
    t = 0

    if args.load_path is not None:
        saver.restore(sess, args.load_path)
        t = sess.run(tf.train.get_global_step())
        print('Restoring parameters at step {} from: {}'.format(start_step - 1, args.load_path))

    # divide n_total_steps of training into episodes of fixed length during which a single head acts but all heads train;
    # select acting head uniformly at random
    n_episodes = n_total_steps // episode_length
    for e in range(n_episodes):
        # select head
        actor_head_idx = np.random.randint(0, n_heads)
        # reset env and trackers
        obs = env.reset()
        episode_lengths *= 0
        episode_rewards *= 0

        for i in range(episode_length):
            tic = time.time()

            # sample action -> step env -> store transition
            actions = agent.get_actions(obs, actor_head_idx)
            next_obs, r, done, _ = env.step(actions)
            done_bool = np.where(episode_lengths + 1 == max_episode_length, np.zeros_like(done), done)  # only store true `done` in buffer not episode ends
            memory.store_transition(obs, actions, r, done_bool, next_obs)
            obs = next_obs

            # keep records
            t += 1
            episode_rewards += r
            episode_lengths += 1

            # end of episode -- when all envs are done or max_episode length is reached, reset
            if any(done):
                for d in np.nonzero(done)[0]:
                    episode_rewards_history.append(float(episode_rewards[d]))
                    episode_lengths_history.append(int(episode_lengths[d]))
                    episodes_completed += 1
                    # reset counters
                    episode_rewards[d] = 0
                    episode_lengths[d] = 0

            # train
            agent.train(memory, batch_size)
            agent.update_target_net()

            # save
            if t % args.save_interval == 0 and args.rank == 0:
                saver.save(sess, args.output_dir + '/agent.ckpt', global_step=tf.train.get_global_step())

            # log stats
            if t % args.log_interval == 0:
                ep_rewards_mean, ep_rewards_std, _ = mpi_moments(episode_rewards_history)
                ep_lengths_mean, ep_lengths_std, _ = mpi_moments(episode_lengths_history)
                episodes_completed_mean, _, episodes_completed_count = mpi_moments([episodes_completed])
                toc = time.time()
                stats['timestep'] = args.world_size * t
                stats['episodes'] = episodes_completed_mean * episodes_completed_count
                stats['steps_per_second'] = args.world_size * args.log_interval / (toc - tic)
                stats['avg_return'] = ep_rewards_mean
                stats['std_return'] = ep_rewards_std
                stats['avg_episode_length'] = ep_lengths_mean
                stats['std_episode_length'] = ep_lengths_std
                if args.rank == 0:
                    logger.save_csv(stats, args.output_dir + '/log.csv')
                    print(tabulate(stats.items(), tablefmt='rst'))

            # different threads have different seeds
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
                'expl_noise': 0.,
                'discount': 0.96,
                'tau': 0.005,
                'q_lr': 1e-3,
                'policy_lr': 1e-3,
                'batch_size': 128,
                'max_memory_size': int(1e6),
                'n_prefill_steps': 1000,
                'n_heads': 8}
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
                'max_memory_size': int(1e6),
                'n_prefill_steps': n_prefill_steps,
                'n_heads': 1}

