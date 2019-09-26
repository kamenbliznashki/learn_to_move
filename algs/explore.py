import numpy as np
import tensorflow as tf

from algs.models import Model
from env_wrappers import RewardAugEnv

# selct obs indices to model using the state predictors;
# 1 .select vtgt field idxs; 0 index is start of v_tgt_field (after PoolVtgt and Obs2Vec env transforms)
vtgt_idxs = np.array(list(range(5)))  # NOTE -- this should match the vtgt field size from obs2vec and poolvtgt
VTGT_OFFSET = len(vtgt_idxs)
# 2.1 select pose idxs; 0 index is start of pose data (ie v_tgt_field excluded)
pose_idxs = np.array([*list(range(9)),                        # pelvis       (9 obs)
                      *list(range(12,16)),                    # joints r leg (4 obs)
                      *list(range(20+3*11+3,20+3*11+3+4))])   # joints l leg (4 obs)
# 2.2 offset the pose info for the v_tgt_field size -- NOTE NOTE NOTE -- this should match vtgt field size from Obs2Vec and PoolVtgt
pose_idxs += VTGT_OFFSET
# 3. stack all selected idxs
IDXS = np.hstack([vtgt_idxs, pose_idxs])


class SPEnsemble:
    def __init__(self, action_shape, action_space_low, action_space_high, *,
                    n_state_predictors, state_predictor_hidden_sizes,
                    n_step_lookahead, lr, batch_size, n_train_steps, bonus_scale):
        self.action_shape = action_shape
        self.actions_space_low = action_space_low
        self.actions_space_high = action_space_high
        self.n_state_predictors = n_state_predictors
        self.state_predictor_hidden_sizes = state_predictor_hidden_sizes
        self.n_step_lookahead = n_step_lookahead
        self.lr = lr
        self.batch_size = batch_size
        self.n_train_steps = n_train_steps
        self.bonus_scale = bonus_scale

    def initialize(self, sess, policy):
        self.sess = sess
        self.policy = policy

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, len(IDXS)], name='sp_obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, None, *self.action_shape], name='actions')  # (n_sample_actions, n_env, action_dim)
        self.next_obs_ph = tf.placeholder(tf.float32, [None, len(IDXS)], name='sp_next_obs')

        # setup state predictor models
        self.state_predictors = [Model('state_predictor_{}'.format(i), hidden_sizes=self.state_predictor_hidden_sizes, output_size=len(IDXS))
                                    for i in range(self.n_state_predictors)]

        # setup dynamics function
        self.pred_normed_next_obs, self.pred_next_obs = self.dynamics_fn(self.obs_ph, self.actions_ph)

        # setup loss
        self.losses = [tf.losses.mean_squared_error(pred, self.next_obs_ph) for pred in self.pred_next_obs]

        # setup training
        optimizer = tf.train.AdamOptimizer(self.lr, name='sp_optimizer')
        self.train_ops = [optimizer.minimize(loss, var_list=model.trainable_vars) for loss, model in zip(self.losses, self.state_predictors)]

        # setup action selection
        self.actions = self.policy(self.obs_ph)  # include only to visualize model based rollouts
        self.best_actions = self.setup_action_selection(self.obs_ph, self.actions_ph)

    def dynamics_fn(self, obs, actions):
        actions = tf.reshape(actions, [-1, *self.action_shape])

        # normalize state
        obs_mean, obs_var = tf.nn.moments(obs, axes=0)
        normed_obs = (self.obs_ph - obs_mean) / (obs_var**0.5 + 1e-8)

        pred_normed_next_obs = [model(tf.concat([normed_obs, actions], 1)) for model in self.state_predictors]
        pred_next_obs = [pred * (obs_var[None,:]**0.5 + 1e-8) + obs_mean[None,:] for pred in pred_normed_next_obs] # each element is (B, obs_dim)

        return pred_normed_next_obs, pred_next_obs

    def reward_fn(self, obs, actions, pred_next_obs):
        y_vtgt = pred_next_obs[...,:VTGT_OFFSET]
        height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw = tf.split(pred_next_obs[...,VTGT_OFFSET: VTGT_OFFSET+9], 9, -1)
        rewards = RewardAugEnv.compute_rewards(y_vtgt, height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw, None, None, None, None, None, None)
        rewards = tf.reduce_sum([v for v in rewards.values()], 0)  # (B, 1)
        return rewards

    def setup_action_selection(self, obs, actions_ph):
        rewards = 0
        for i in range(self.n_step_lookahead):
            # use the actions arg for the first step otherwise obtain samples from current sampling dist
            if i == 0:
                actions = actions_ph
            else:
                actions, _ = self.policy(obs)  # (1, B, action_dim)
            # predict next state
            _, pred_next_obs = self.dynamics_fn(obs, actions)  # TODO -- need selection among the ensemble eg particles; mean over state predictions is not a real state
            pred_next_obs = tf.reduce_mean(pred_next_obs, 0)
            # compute rewards
            rewards += self.reward_fn(obs, actions, pred_next_obs)

            obs = pred_next_obs

        # select the best first action given the total rewards
        #   reshape all to (n_sampled_actions, batch_size, *_dim) where batch_size is n_env and *_dim is rewards_dim or actions_dim
        rewards = tf.reshape(rewards, [tf.shape(actions_ph)[0], -1])  # (n_sample_actions, n_env)
        best_action_idx = tf.argmax(rewards, axis=0)[0]
        return actions_ph[best_action_idx]

    def train(self, memory):
        # train bootstrapped ensemble
        losses = []
        for loss_op, train_op in zip(self.losses, self.train_ops):
            for _ in range(self.n_train_steps):
                # sample memory
                batch = memory.sample(self.batch_size)
                obs = np.take(batch.obs, IDXS, 1)
                next_obs = np.take(batch.next_obs, IDXS, 1)
                actions = batch.actions[None,...]
                # train
                loss, _ = self.sess.run([loss_op, train_op],
                                        feed_dict={self.obs_ph: obs, self.actions_ph: actions, self.next_obs_ph: next_obs})
                losses.append(loss)
        return np.mean(losses)

    def get_exploration_bonus(self, obs, actions):
        """
        Args
            obs     -- np array; (n_env, obs_dim)
            actions -- np array; (1, n_env, actions_dim)
        """
        obs = np.take(obs, IDXS, 1)
        actions = actions[None,...]
        pred_normed_next_obs = self.sess.run(self.pred_normed_next_obs, {self.obs_ph: obs, self.actions_ph: actions})
        # bonus is variance among the state predictors and along the predicted state vector
        #   ie a/ incent disagreement among the state predictors (explore state they can't model well);
        #      b/ incent exploring diverse state vectors; eg left-right leg mid-stride having opposite signs is higher var than standing / legs in same position
        return self.bonus_scale * np.var(pred_normed_next_obs, axis=(0,2))[:,None]

    def get_best_action(self, obs, actions):
        """
        Args
            obs     -- np array; (n_env, obs_dim) during training; (1, obs_dim) during eval
            actions -- np array; (n_samples, n_env, actions_dim) during training; (n_samples, 1, actions_dim) during eval
        """
        obs = np.atleast_2d(obs)
        obs = np.take(obs, IDXS, 1)
        obs = np.tile(obs, (actions.shape[0], 1))
        best_actions = self.sess.run(self.best_actions, {self.obs_ph: obs, self.actions_ph: actions})
        return np.atleast_2d(best_actions)

    def get_pred_next_obs(self, obs, actions):
        obs = np.take(obs, IDXS, 1)
        actions = actions[None,...]
        _, pred_next_obs = self.sess.run(self.pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})
        return pred_next_obs

    def get_student_policy_action(self, obs):
        obs = np.atleast_2d(obs)
        obs = np.take(obs, IDXS, 1)
        return self.sess.run(self.actions, {self.obs_ph, obs})

# --------------------
# defaults
# --------------------

def defaults(class_name=None):
    if class_name == 'SPEnsemble':
        return {'n_state_predictors': 5,
                'state_predictor_hidden_sizes': (128, 128),
                'n_step_lookahead': 5,
                'lr': 1e-3,
                'batch_size': 256,
                'n_train_steps': 2,
                'bonus_scale': 2}

