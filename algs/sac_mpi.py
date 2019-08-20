from collections import deque
import time

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from mpi4py import MPI

from algs.memory import Memory
from algs.models import GaussianPolicy, Model
from mpi_adam import MpiAdam, flatgrad
from mpi_utils import mpi_moments
import logger


class SAC:
    def __init__(self, observation_shape, action_shape, *,
                    policy_hidden_sizes, q_hidden_sizes, value_hidden_sizes, alpha, discount, tau, lr):
        self.lr = lr

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.rewards_ph = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.dones_ph = tf.placeholder(tf.float32, [None, 1], name='dones')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')

        # build graph
        # 1. networks
#        embed_v_tgt = Model('vtgt_embed', hidden_sizes=[128, 128], output_size=8, output_activation=tf.tanh)
        value_function = Model('value_function', hidden_sizes=value_hidden_sizes, output_size=1)
        target_value_function = Model('target_value_function', hidden_sizes=value_hidden_sizes, output_size=1)
        q_function  = Model('q_function', hidden_sizes=q_hidden_sizes, output_size=1)
        q_function2 = Model('q_function2', hidden_sizes=q_hidden_sizes, output_size=1)
        policy = GaussianPolicy('policy', hidden_sizes=policy_hidden_sizes, output_size=2*action_shape[0])

#        # split obs from v_tgt_field and embed vtgt
#        vtgt_size = 2*11*11
#        vtgt = self.obs_ph[:,:vtgt_size]
#        next_vtgt = self.next_obs_ph[:,:vtgt_size]
#        embedded_vtgt = embed_v_tgt(vtgt)
#        embedded_next_vtgt = embed_v_tgt(next_vtgt)
#        obs = tf.concat([embedded_vtgt, self.obs_ph[:,vtgt_size:]], axis=1)
#        next_obs = tf.concat([embedded_next_vtgt, self.next_obs_ph[:,vtgt_size:]], axis=1)

        # 2. loss
        #   value function loss term
        v = value_function(self.obs_ph)
        self.actions, log_pis = policy(self.obs_ph)
        q_value_at_policy_action = tf.minimum(q_function(tf.concat([self.obs_ph, self.actions], 1)),
                                              q_function2(tf.concat([self.obs_ph, self.actions], 1)))
        target_v = q_value_at_policy_action - alpha * log_pis
        value_function_loss = tf.losses.mean_squared_error(v, target_v)
        #   q value loss term
        q_value_at_memory_action = q_function(tf.concat([self.obs_ph, self.actions_ph], 1))
        target_next_v = target_value_function(self.next_obs_ph) * (1 - self.dones_ph)
        target_q = self.rewards_ph + discount * target_next_v
        q_value_loss = tf.losses.mean_squared_error(q_value_at_memory_action, target_q)
        #   q_function2 loss
        q_value2_at_memory_action = q_function2(tf.concat([self.obs_ph, self.actions_ph], 1))
        q_value2_loss = tf.losses.mean_squared_error(q_value2_at_memory_action, target_q)
        #   policy loss term
        policy_loss = tf.reduce_mean(alpha * log_pis - q_value_at_policy_action)

        # 3. update ops
        v_grads = flatgrad(value_function_loss, value_function.trainable_vars)
        self.v_optimizer = MpiAdam(var_list=value_function.vars, scale_grad_by_procs=False)
        q_grads = flatgrad(q_value_loss, q_function.trainable_vars)
        self.q_optimizer = MpiAdam(var_list=q_function.vars, scale_grad_by_procs=False)
        q2_grads = flatgrad(q_value2_loss, q_function2.trainable_vars)
        self.q2_optimizer = MpiAdam(var_list=q_function2.vars, scale_grad_by_procs=False)
        policy_grads = flatgrad(policy_loss, policy.trainable_vars)
        self.policy_optimizer = MpiAdam(var_list=policy.vars, scale_grad_by_procs=False)
        update_global_step = tf.assign_add(tf.train.get_or_create_global_step(), MPI.COMM_WORLD.Get_size())
        #   combined train ops
        self.train_ops = [v_grads, value_function_loss, q_grads, q_value_loss, q2_grads, q_value2_loss, policy_grads,
                            policy_loss, update_global_step]
        self.target_update_ops = [tf.assign(target, (1 - tau) * target + tau * source) for target, source in \
                                    zip(target_value_function.trainable_vars, value_function.trainable_vars)]

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.v_optimizer.sync()
        self.q_optimizer.sync()
        self.q2_optimizer.sync()
        self.policy_optimizer.sync()

    def get_actions(self, obs):
        actions = self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})
        return actions

    def update_target_net(self):
        self.sess.run(self.target_update_ops)

    def train_step(self, batch):
        v_grads, value_function_loss, q_grads, q_value_loss, q2_grads, q_value2_loss, policy_grads, policy_loss, _ = self.sess.run(
                self.train_ops, {self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards,
                                       self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})
        self.v_optimizer.update(v_grads, stepsize=self.lr)
        self.q_optimizer.update(q_grads, stepsize=self.lr)
        self.q2_optimizer.update(q2_grads, stepsize=self.lr)
        self.policy_optimizer.update(policy_grads, stepsize=self.lr)
        return value_function_loss, q_value_loss, q_value2_loss, policy_loss

def learn(env, seed, n_total_steps, max_episode_length, alg_args, args):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    max_memory_size = alg_args.pop('max_memory_size', int(1e6))
    reward_scale = alg_args.pop('reward_scale', 1.)
    batch_size = alg_args.pop('batch_size', 256)
    memory = Memory(int(max_memory_size), env.observation_space.shape, env.action_space.shape, reward_scale)
    agent = SAC(env.observation_space.shape, env.action_space.shape, **alg_args)

    # initialize session, agent, memory, environment
    sess = tf.get_default_session()
    agent.initialize(sess)
    saver = tf.train.Saver()
    sess.graph.finalize()
    memory.initialize(env, training=not args.load_path)
    obs = env.reset()

    # setup tracking
    stats = {}
    episode_rewards = np.zeros((env.num_envs, 1), dtype=np.float32)
    episode_lengths = np.zeros((env.num_envs, 1), dtype=int)
    episode_rewards_history = deque(maxlen=100)
    episode_lengths_history = deque(maxlen=100)
    n_episodes = 0
    start_step = 1

    if args.load_path is not None:
        saver.restore(sess, args.load_path)
        start_step = sess.run(tf.train.get_global_step()) + 1
        print('Restoring parameters at step {} from: {}'.format(start_step - 1, args.load_path))

    for t in range(start_step, n_total_steps + start_step):
        tic = time.time()

        # sample action -> step env -> store transition
        actions = agent.get_actions(obs)
        next_obs, r, done, _ = env.step(actions)
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
        value_function_loss, q_value_loss, q_value2_loss, policy_loss = agent.train_step(batch)
        agent.update_target_net()

        # save
        if t % args.save_interval == 0 and args.rank == 0:
            saver.save(sess, args.output_dir + '/agent.ckpt', global_step=tf.train.get_global_step())

        # log stats
        if t % args.log_interval == 0:
            toc = time.time()
            ep_rewards_mean, ep_rewards_std, _ = mpi_moments(episode_rewards_history)
            ep_lengths_mean, ep_lengths_std, _ = mpi_moments(episode_lengths_history)
            n_episodes_mean, _, n_episodes_count = mpi_moments([n_episodes])
            stats['timestep'] = args.world_size * t
            stats['episodes'] = n_episodes_mean * n_episodes_count
            stats['steps_per_second'] = args.world_size * args.log_interval / (toc - tic)
            stats['fps'] = args.world_size * env.num_envs * batch_size / (toc - tic)
            stats['avg_return'] = ep_rewards_mean
            stats['std_return'] = ep_rewards_std
            stats['avg_episode_length'] = ep_lengths_mean
            stats['std_episode_length'] = ep_lengths_std
            stats['value_function_loss'] = value_function_loss
            stats['q_value_loss'] = q_value_loss
            stats['q_value2_loss'] = q_value2_loss
            stats['policy_loss'] = policy_loss
            if args.rank == 0:
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
                'alpha': 0.2,
                'discount': 0.99,
                'tau': 0.01,
                'lr': 1e-3,
                'batch_size': 256,
                'max_memory_size': int(1e6),
                'reward_scale': 1}
    else:  # mujoco
        alpha = {
            'Ant-v2': 0.1,
            'HalfCheetah-v2': 0.2,
            'Hopper-v2': 0.2,
            'Humanoid-v2': 0.05,
            'Walker2d-v2': 0.2,
            }.get(env_name, 0.2)

        reward_scale = {
            'Ant-v2': 5,
            'HalfCheetah-v2': 5,
            'Hopper-v2': 5,
            'Humanoid-v2': 20,
            'Walker2d-v2': 5,
            }.get(env_name, 5)

        return {'policy_hidden_sizes': (256, 256),
                'value_hidden_sizes': (256, 256),
                'q_hidden_sizes': (256, 256),
                'alpha': alpha,
                'discount': 0.99,
                'tau': 0.005,
                'lr': 3e-4,
                'batch_size': 256,
                'max_memory_size': int(1e6),
                'reward_scale': reward_scale}

