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


class DDPG:
    def __init__(self, observation_shape, action_shape, min_action, max_action, *,
                    policy_hidden_sizes, q_hidden_sizes, discount, tau, q_lr, policy_lr, n_heads):
        self.min_action = min_action
        self.max_action = max_action
        self.tau = tau
        self.policy_lr = policy_lr
        self.q_lr = q_lr

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.rewards_ph = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.dones_ph = tf.placeholder(tf.float32, [None, 1], name='dones')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')

        # obs indices to pass to policy and q function
        # policy gets pelvis hight pitch roll, joints, fiber lengths -- no velocity
#        idxs = [*list(range(3)), *list(range(12,16)), *list(range(20,20+3*11))[1::3], *list(range(20+3*11+3,20+3*11+3+4)), \
#                        *list(range(20+3*11+3+4+4, 20+2*3*11+3+4+4))[1::3]]
#        policy_idxs = np.asarray(policy_idxs) + 2*3*3
#        policy_idxs = policy_idxs.tolist()

        # build graph
        # 1. networks
        qs, q_targets = [], []
        policies, policy_targets = [], []
        for i in range(n_heads):
            qs.append(Model('q_{}'.format(i), hidden_sizes=q_hidden_sizes, output_size=1))
            q_targets.append(Model('q_target_{}'.format(i), hidden_sizes=q_hidden_sizes, output_size=1))
            policies.append(Model('policy_{}'.format(i), hidden_sizes=policy_hidden_sizes, output_size=action_shape[0], output_activation=tf.tanh))
            policy_targets.append(Model('policy_target_{}'.format(i), hidden_sizes=policy_hidden_sizes, output_size=action_shape[0], output_activation=tf.tanh))

        # current q values (choose 1 q head to act; train on all)
        q_value = [q(tf.concat([self.obs_ph, self.actions_ph], 1)) for q in qs]
        # q values at policy action
        self.actions = [max_action * policy(self.obs_ph) for policy in policies]
#        self.actions = max_action * policy(tf.gather(self.obs_ph, policy_idxs, axis=1))
        self.q_value_at_policy_action = [q(tf.concat([self.obs_ph, action], 1)) for q, action in zip(qs, self.actions)]

        # select next action according to the policy_target
        next_actions = [policy_target(self.next_obs_ph) for policy_target in policy_targets]
#        next_actions = policy_target(tf.gather(self.next_obs_ph, policy_idxs, axis=1))
        # compute q targets
        q_target_value = [q_target(tf.concat([self.next_obs_ph, next_action], 1)) for q_target, next_action in zip(q_targets, next_actions)]
        q_target_value = [self.rewards_ph + tf.stop_gradient(discount * q_target_value[i] * (1 - self.dones_ph)) for i in range(n_heads)]

        # 2. loss on critics and actor
        self.q_loss = tf.reduce_mean([tf.losses.mean_squared_error(q_v, q_target_v) for q_v, q_target_v in zip(q_value, q_target_value)])
        self.policy_loss = - tf.reduce_mean([tf.reduce_mean(self.q_value_at_policy_action[i]) for i in range(n_heads)])

        # 3. training
        self.q_grads = flatgrad(self.q_loss, [i for q in qs for i in q.trainable_vars])
        self.q_optimizer = MpiAdam(var_list=[i for q in qs for i in q.vars], scale_grad_by_procs=False)
        self.policy_grads = flatgrad(self.policy_loss, [i for policy in policies for i in policy.trainable_vars])
        self.policy_optimizer = MpiAdam(var_list=[i for policy in policies for i in policy.vars], scale_grad_by_procs=False)
        world_size = MPI.COMM_WORLD.Get_size()
        self.update_global_step = tf.assign_add(tf.train.get_or_create_global_step(), world_size)
        #   target update ops
        self.target_update_ops = [tf.group(self.create_target_update_op(q, q_target) +
                                           self.create_target_update_op(policy, policy_target))
                                    for q, q_target, policy, policy_target in zip(qs, q_targets, policies, policy_targets)]

        # init target networks
        self.target_init_ops = [tf.group(self.create_target_init_op(q, q_target) +
                                        self.create_target_init_op(policy, policy_target))
                                    for q, q_target, policy, policy_target in zip(qs, q_targets, policies, policy_targets)]

    def create_target_init_op(self, source, target):
        return [tf.assign(t, s) for t, s in zip(target.vars, source.vars)]

    def create_target_update_op(self, source, target):
        return [tf.assign(t, (1 - self.tau) * t + self.tau * s) for t, s in zip(target.vars, source.vars)]

    def initialize(self, sess):
        tf.train.get_or_create_global_step()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.policy_optimizer.sync()
        self.q_optimizer.sync()
        self.sess.run(self.target_init_ops)

    def get_actions(self, obs, expl_noise=0, head_idxs=None):
        actions = self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})
        actions = np.stack(actions, axis=1)  # (n_env, n_heads, action_dim)

        if head_idxs is not None:
            actions = actions[np.arange(actions.shape[0]), head_idxs]  # for each env select the head_idx
        else:
            actions = np.squeeze(actions, 0)            # (n_heads, action dim)
            obs = np.tile(obs, (actions.shape[0], 1))   # (n_heads, obs dim)
            # batch actions from each policy and run action-obs through each q function; out (n_heads, 1)
            qs = self.sess.run(self.q_value_at_policy_action, {self.obs_ph: obs, self.actions_ph: actions})
            qs = np.vstack(qs)
            # normalize q values
            qs /= qs.sum(1, keepdims=True)
            # select action at highest q value
            actions = actions[qs.argmax()]

        if expl_noise != 0:
            actions += np.random.normal(0, expl_noise, actions.shape)
            actions = np.clip(actions, self.min_action, self.max_action)
        return actions

    def update_target_net(self):
        self.sess.run(self.target_update_ops)

    def train(self, batch):
        policy_grads, _, q_grads, _, _ = self.sess.run(
                [self.policy_grads, self.policy_loss, self.q_grads, self.q_loss, self.update_global_step],
                feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.rewards_ph: batch.rewards,
                           self.dones_ph: batch.dones, self.next_obs_ph: batch.next_obs})
        self.policy_optimizer.update(policy_grads, stepsize=self.policy_lr)
        self.q_optimizer.update(q_grads, stepsize=self.q_lr)


def learn(env, seed, n_total_steps, max_episode_length, alg_args, args):
    # extract training and memory buffer args
    expl_noise = alg_args.pop('expl_noise', 0)
    batch_size = alg_args.pop('batch_size', 256)
    max_memory_size = alg_args.pop('max_memory_size', int(1e6))
    n_prefill_steps = alg_args.pop('n_prefill_steps', 1000)
    reward_scale = alg_args.pop('reward_scale', 1.)
    n_heads = alg_args.get('n_heads', 1)

    np.random.seed(int(seed + 1e6*args.rank))
    tf.set_random_seed(int(seed + 1e6*args.rank))

    memory = Memory(int(max_memory_size), env.observation_space.shape, env.action_space.shape, reward_scale)
    agent = DDPG(env.observation_space.shape, env.action_space.shape, env.action_space.low[0], env.action_space.high[0], **alg_args)

    # initialize session, agent and memory
    sess = tf.get_default_session()
    agent.initialize(sess)
    saver = tf.train.Saver()
    sess.graph.finalize()
    memory.initialize(env, n_prefill_steps, training=not args.play)
    obs = env.reset()
    head_idxs = np.random.randint(0, n_heads, (env.num_envs,))

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
        actions = agent.get_actions(obs, expl_noise, head_idxs)
        next_obs, r, done, _ = env.step(actions)
        done_bool = np.where(episode_lengths + 1 == max_episode_length, np.zeros_like(done), done)  # only store true `done` in buffer not episode ends
        memory.store_transition(obs, actions, r, done_bool, next_obs)
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
                # select a new head
                head_idxs[d] = np.random.randint(0, n_heads)

        # train
        batch = memory.sample(batch_size)
        agent.train(batch)
        agent.update_target_net()

        # save
        if t % args.save_interval == 0 and args.rank == 0:
            saver.save(sess, args.output_dir + '/agent.ckpt', global_step=tf.train.get_global_step())

        # log stats
        if t % args.log_interval == 0:
            ep_rewards_mean, ep_rewards_std, _ = mpi_moments(episode_rewards_history)
            ep_lengths_mean, ep_lengths_std, _ = mpi_moments(episode_lengths_history)
            n_episodes_mean, _, n_episodes_count = mpi_moments([n_episodes])
            toc = time.time()
            stats['timestep'] = args.world_size * t
            stats['episodes'] = n_episodes_mean * n_episodes_count
            stats['steps_per_second'] = args.world_size * args.log_interval / (toc - tic)
            stats['fps'] = args.world_size * env.num_envs * batch_size / (toc - tic)
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
                'reward_scale': 1,
                'n_heads': 2}
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
                'reward_scale': 1}

