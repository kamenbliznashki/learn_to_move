import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from algs.models import Model
from env_wrappers import RewardAugEnv


class SPEnsemble:
    def __init__(self, observation_shape, action_shape, action_space_low, action_space_high, v_tgt_field_size, *,
                    n_models, model_hidden_sizes,
                    n_step_lookahead, lr, batch_size, n_train_steps, bonus_scale):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.actions_space_low = action_space_low
        self.actions_space_high = action_space_high
        self.v_tgt_field_size = v_tgt_field_size
        self.n_models = n_models
        self.model_hidden_sizes = model_hidden_sizes
        self.n_step_lookahead = n_step_lookahead
        self.lr = lr
        self.batch_size = batch_size
        self.n_train_steps = n_train_steps
        self.bonus_scale = bonus_scale

    def initialize(self, sess, memory, policy):
        self.sess = sess
        self.policy = policy
        self.memory = memory
        self.build()

    def build(self):
        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *self.observation_shape], name='sp_obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *self.action_shape], name='sp_actions')  # (n_sample_actions, n_env, action_dim)
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *self.observation_shape], name='sp_next_obs')

        # setup state predictor models
        self.models = [Model('state_predictor_{}'.format(i), hidden_sizes=self.model_hidden_sizes, output_size=self.observation_shape[0])
                                    for i in range(self.n_models)]

        # setup dynamics function
        self.delta_pred_normed, self.pred_next_obs = self.dynamics_fn(self.obs_ph, self.actions_ph)

        # setup training
        delta_actual = self.next_obs_ph - self.obs_ph
        delta_actual_normed = (delta_actual - self.memory.delta_obs_mean) / (self.memory.delta_obs_std + 1e-8)
        self.losses = [tf.losses.mean_squared_error(pred, delta_actual_normed) for pred in self.delta_pred_normed]

        # setup training
        optimizer = tf.train.AdamOptimizer(self.lr, name='sp_optimizer')
        self.train_ops = [optimizer.minimize(loss, var_list=model.trainable_vars) for loss, model in zip(self.losses, self.models)]

    def dynamics_fn(self, obs, actions):
        # normalize states and actions
        normed_obs = (obs - self.memory.obs_mean) / (self.memory.obs_std + 1e-8)
        normed_actions = (actions - self.memory.action_mean) / (self.memory.action_std + 1e-8)

        # predict delta using models
        delta_pred_normed_mean = [model(tf.concat([normed_obs, normed_actions], 1)) for model in self.models]
        # sample transition from each model N(delta_pred, 0.5)
        delta_pred_normed = [pred + 0.25 * tf.random_normal(tf.shape(pred)) for pred in delta_pred_normed_mean]

        # unnormalize mean prediction and predict next state
        delta_pred = tf.reduce_mean(delta_pred_normed, axis=0) * self.memory.delta_obs_std + self.memory.delta_obs_mean
        pred_next_obs = obs + delta_pred
        pred_next_obs = tf.concat([tf.one_hot(tf.argmin(pred_next_obs[:, :3], axis=1), depth=3), # x_vtgt_onehot
                                   tf.one_hot(tf.argmin(pred_next_obs[:,3:6], axis=1), depth=3), # y_vtgt_onehot
                                   pred_next_obs[:, 6:]], axis=1)

        return delta_pred_normed, pred_next_obs

    def reward_fn(self, obs, actions, pred_next_obs):
        vtgt_field = pred_next_obs[...,:self.v_tgt_field_size]
        x_vtgt_onehot, y_vtgt_onehot, goal_dist = np.split(
                vtgt_field, [(self.v_tgt_field_size - 1)//2, self.v_tgt_field_size - 1], axis=-1)  # split to produce 3 arrays of shape (n,), (n,) and (1,)  where n is half the pooled v_tgt_field
        height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw = np.split(pred_next_obs[...,self.v_tgt_field_size: self.v_tgt_field_size+9], 9, -1)
        rewards = RewardAugEnv.compute_rewards(x_vtgt_onehot, goal_dist, height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw, None, None, None, None, None, None)
        return np.sum([v for v in rewards.values()], axis=0)  # (B, 1)

    def train(self):
        # train bootstrapped ensemble
        losses = []
        for loss_op, train_op in zip(self.losses, self.train_ops):
            for _ in range(self.n_train_steps):
                # sample memory
                batch = self.memory.sample(self.batch_size)
                # train
                loss, _ = self.sess.run([loss_op, train_op], feed_dict=
                        {self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.next_obs_ph: batch.next_obs})
                losses.append(loss)
        return np.mean(losses)

    def get_exploration_bonus(self, obs, actions):
        """
        Args
            obs     -- np array; (n_env, obs_dim)
            actions -- np array; (n_env, actions_dim)
        """
        delta_pred_normed = self.sess.run(self.delta_pred_normed, {self.obs_ph: obs, self.actions_ph: actions})
        # bonus is variance among the state predictors and along the predicted state vector
        #   ie a/ incent disagreement among the state predictors (explore state they can't model well);
        #      b/ incent exploring diverse state vectors; eg left-right leg mid-stride having opposite signs is higher var than standing / legs in same position
        return self.bonus_scale * np.var(delta_pred_normed, axis=(0,2))[:,None]

    def get_best_action(self, obs, init_actions):
        """
        Args
            obs     -- np array; (n_env, obs_dim) during training; (1, obs_dim) during eval
            actions -- np array; (n_samples, n_env, actions_dim) during training; (n_samples, 1, actions_dim) during eval
        """
        obs = np.atleast_2d(obs)
        obs = np.tile(obs, (init_actions.shape[0], 1))

        rewards = 0
        for i in range(self.n_step_lookahead):
            # use the actions arg for the first step otherwise obtain samples from current sampling dist
            actions = init_actions.reshape(-1, *self.action_shape) if i == 0 else self.policy.get_actions(obs)  # (B, action_dim)
            # predict next state
            pred_next_obs = self.get_pred_next_obs(obs, actions)
            # compute rewards
            rewards += self.reward_fn(obs, actions, pred_next_obs)

            obs = pred_next_obs

        # select the best first action given the total rewards
        rewards = rewards.reshape(init_actions.shape[0], -1)                            # (n_sample_actions, n_env)
        best_actions = init_actions[rewards.argmax(0), np.arange(rewards.shape[1]), :]  # (n_env,) -- [take argmax of rewards, for each n_env, full action dim]
        return np.atleast_2d(best_actions)

    def get_pred_next_obs(self, obs, actions):
        return self.sess.run(self.pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})

# --------------------
# defaults
# --------------------

def defaults(class_name=None):
    if class_name == 'SPEnsemble':
        return {'n_models': 3,
                'model_hidden_sizes': (512, 512),
                'n_step_lookahead': 5,
                'lr': 1e-3,
                'batch_size': 256,
                'n_train_steps': 50,
                'bonus_scale': 2}

