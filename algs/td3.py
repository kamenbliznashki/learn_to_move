from collections import deque
import time

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from algs.memory import Memory
from algs.models import GaussianPolicy, Model
import logger


class TD3:
    def __init__(self, observation_shape, action_shape, min_action, max_action, *,
                    policy_noise, noise_clip, policy_hidden_sizes, q_hidden_sizes, discount, tau, q_lr, policy_lr):
        self.min_action = min_action
        self.max_action = max_action
        self.tau = tau

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.rewards_ph = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.dones_ph = tf.placeholder(tf.float32, [None, 1], name='dones')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')

        # build graph
        # 1. networks
        q1 = Model('q1', hidden_sizes=q_hidden_sizes, output_size=1)
        q2 = Model('q2', hidden_sizes=q_hidden_sizes, output_size=1)
        policy = Model('policy', hidden_sizes=policy_hidden_sizes, output_size=action_shape[0], output_activation=tf.tanh)
        q1_target = Model('q1_target', hidden_sizes=q_hidden_sizes, output_size=1)
        q2_target = Model('q2_target', hidden_sizes=q_hidden_sizes, output_size=1)
        policy_target = Model('policy', hidden_sizes=policy_hidden_sizes, output_size=action_shape[0], output_activation=tf.tanh)

        # current q values
        q1_value = q1(tf.concat([self.obs_ph, self.actions_ph], 1))
        q2_value = q2(tf.concat([self.obs_ph, self.actions_ph], 1))
        # q values at policy action
        self.actions = max_action * policy(self.obs_ph)
        q1_value_at_policy_action = q1(tf.concat([self.obs_ph, self.actions], 1))

        # select next action according to the policy_target and add noise
        next_actions = max_action * policy_target(self.next_obs_ph)
        eps = tf.clip_by_value(tf.random_normal(tf.shape(next_actions), stddev=policy_noise), -noise_clip, noise_clip)
        next_actions = tf.clip_by_value(next_actions + eps, -max_action, max_action)
        # compute q targets
        q1_target_value = q1_target(tf.concat([self.next_obs_ph, next_actions], 1))
        q2_target_value = q2_target(tf.concat([self.next_obs_ph, next_actions], 1))
        q_target_value = self.rewards_ph + tf.stop_gradient(discount * tf.minimum(q1_target_value, q2_target_value) * (1 - self.dones_ph))

        # 2. loss on critics and actor
        q_loss = tf.losses.mean_squared_error(q1_value, q_target_value) + tf.losses.mean_squared_error(q2_value, q_target_value)
        policy_loss = - tf.reduce_mean(q1_value_at_policy_action)

        # 3. update ops
        q_optimizer = tf.train.AdamOptimizer(q_lr, name='q_optimizer')
        policy_optimizer = tf.train.AdamOptimizer(policy_lr, name='policy_optimizer')
        #   main train ops
        self.q_train_op = q_optimizer.minimize(q_loss, var_list=[q1.trainable_vars, q2.trainable_vars])
        self.policy_train_op = policy_optimizer.minimize(policy_loss, var_list=policy.trainable_vars)
        #   target update ops
        self.target_update_ops = tf.group(self.create_target_update_op(q1, q1_target) +
                                          self.create_target_update_op(q2, q2_target) +
                                          self.create_target_update_op(policy, policy_target))

        # init target networks
        self.target_init_ops = tf.group(self.create_target_init_op(q1, q1_target) +
                                        self.create_target_init_op(q2, q2_target) +
                                        self.create_target_init_op(policy, policy_target))

    def create_target_init_op(self, source, target):
        return [tf.assign(t, s) for t, s in zip(target.vars, source.vars)]

    def create_target_update_op(self, source, target):
        return [tf.assign(t, (1 - self.tau) * t + self.tau * s) for t, s in zip(target.vars, source.vars)]

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_ops)

    def get_actions(self, obs, expl_noise=0):
        actions = self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})
        if expl_noise != 0:
            actions += np.random.normal(0, expl_noise, actions.shape).clip(self.min_action, self.max_action)
        return actions

    def update_target_net(self):
        self.sess.run(self.target_update_ops)

    def train_critic(self, batch):
        self.sess.run(self.q_train_op, {self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards, 
                                         self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})

    def train_actor(self, batch):
        self.sess.run(self.policy_train_op, {self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards,
                                              self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})


def learn(env, seed, n_total_steps, max_episode_length, alg_args, args):
    # extract training and memory buffer args
    expl_noise = alg_args.pop('expl_noise', 0)
    batch_size = alg_args.pop('batch_size', 256)
    policy_update_freq = alg_args.pop('policy_update_freq', 2)
    max_memory_size = alg_args.pop('max_memory_size', int(1e6))
    n_prefill_steps = alg_args.pop('n_prefill_steps', 1000)
    reward_scale = alg_args.pop('reward_scale', 1.)

    np.random.seed(seed)
    tf.set_random_seed(seed)

    memory = Memory(int(max_memory_size), env.observation_space.shape, env.action_space.shape, reward_scale)
    agent = TD3(env.observation_space.shape, env.action_space.shape, env.action_space.low[0], env.action_space.high[0], **alg_args)

    # initialize session, agent and memory
    sess = tf.get_default_session()
    agent.initialize(sess)
    saver = tf.train.Saver()
    sess.graph.finalize()
    memory.initialize(env, n_prefill_steps)

    if args.load_path is not None:
        print('Restoring parameters from: ', args.load_path)
        saver.restore(sess, args.load_path)

    # setup tracking
    stats = {}
    episode_rewards = np.zeros((env.num_envs, 1), dtype=np.float32)
    episode_lengths = np.zeros((env.num_envs, 1), dtype=int)
    episode_rewards_history = deque(maxlen=100)
    episode_lengths_history = deque(maxlen=100)
    last_episode_rewards = 0
    n_episodes = 0

    obs = env.reset()

    for t in range(1, n_total_steps + 1):
        tic = time.time()

        # sample action -> step env -> store transition
        actions = agent.get_actions(obs, expl_noise)
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
        agent.train_critic(batch)
        if t % policy_update_freq == 0:
            agent.train_actor(batch)
            agent.update_target_net()

        # save
        if t % args.save_interval == 0:
            saver.save(sess, args.output_dir + '/agent.ckpt', global_step=tf.train.get_global_step())

        # log stats
        if t % args.log_interval == 0:
            toc = time.time()
            stats['timestep'] = t
            stats['episodes'] = n_episodes
            stats['steps_per_second'] = args.log_interval / (toc - tic)
            stats['fps'] = env.num_envs * batch_size / (toc - tic)
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
        return {'policy_hidden_sizes': (400, 300),
                'q_hidden_sizes': (400, 300),
                'expl_noise': 0.1,
                'policy_noise': 0.2,
                'noise_clip': 0.5,
                'discount': 0.99,
                'tau': 0.005,
                'q_lr': 1e-3,
                'policy_lr': 1e-3,
                'policy_update_freq': 2,
                'batch_size': 100,
                'max_memory_size': int(1e6),
                'n_prefill_steps': 1000,
                'reward_scale': 1}
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
                'policy_noise': 0.2,
                'noise_clip': 0.5,
                'discount': 0.99,
                'tau': 0.005,
                'q_lr': 1e-3,
                'policy_lr': 1e-3,
                'policy_update_freq': 2,
                'batch_size': 100,
                'max_memory_size': int(1e6),
                'n_prefill_steps': n_prefill_steps,
                'reward_scale': 1}
