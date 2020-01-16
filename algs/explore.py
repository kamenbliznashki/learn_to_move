import numpy as np
import tensorflow as tf

from algs.models import Model
from env_wrappers import RewardAugEnv


class BaseExploration:
    def __init__(self, observation_shape, action_shape):
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        # subset of obs indices to model / use in exploration modules
        self.v_tgt_field_size = observation_shape[0] - (339 - 2*11*11)  # L2M original obs dims are 339 where 2*11*11 is the orig vtgt field size
        # pose_idxs start with pelvis at 0 index (obs vector after the v_tgt_field)
        self.pose_idxs = np.array([*list(range(9)),                        # pelvis       (9 obs)
                                   *list(range(12,16)),                    # joints r leg (4 obs)
                                   *list(range(20+3*11+3,20+3*11+3+4))])   # joints l leg (4 obs)
        self.pose_idxs += self.v_tgt_field_size  # offset for v_tgt_field
        self.idxs = np.hstack([list(range(self.v_tgt_field_size)), self.pose_idxs])
        # NOTE --- HalfCheetah uses full state
#        self.idxs = list(range(observation_shape[0]))  # state predictors model the full state space vec
#        self.pose_idxs = self.idxs

    def initialize(self, sess, **kwargs):
        self.sess = sess

    def get_exploration_bonus(self, obs, actions, next_obs):
        raise NotImplementedError

    def train(self, memory, batch_size):
        raise NotImplementedError

class DisagreementExploration(BaseExploration):
    def __init__(self, *args, lr, state_predictor_hidden_sizes, n_state_predictors, bonus_scale, **kwargs):
        super().__init__(*args)
        self.lr = lr
        self.bonus_scale = bonus_scale

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *self.observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *self.action_shape], name='actions')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *self.observation_shape], name='next_obs')

        # build graph
        # 1. networks
        state_predictors = [Model('state_predictor_{}'.format(i), hidden_sizes=state_predictor_hidden_sizes, output_size=len(self.idxs)) for i in range(n_state_predictors)]

        # 2. state predictor outputs
        #   select obs indices to model using the state predictors
        obs = tf.gather(self.obs_ph, self.idxs, axis=1)
        next_obs = tf.gather(self.next_obs_ph, self.idxs, axis=1)

        #   whiten obs and next obs
        obs_mean, obs_var = tf.nn.moments(obs, axes=0)
        next_obs_mean, next_obs_var = tf.nn.moments(next_obs, axes=0)
        normed_obs = (obs - obs_mean) / (obs_var**0.5 + 1e-8)
        normed_next_obs = (next_obs - next_obs_mean) / (next_obs_var**0.5 + 1e-8)

        #   predict from zero-mean unit var obs; shift and scale result by obs mean and var
        normed_pred_next_obs = [model(tf.concat([normed_obs, self.actions_ph], 1)) for model in state_predictors]
        self.normed_pred_next_obs = tf.stack(normed_pred_next_obs, axis=1)  # (B, n_state_predictors, obs_dim)
        self.pred_next_obs = self.normed_pred_next_obs * (obs_var[None,None,:]**0.5 + 1e-8) + obs_mean[None,None,:]  # (B, n_state_predictors, obs_dim)

        # 2. loss
        self.loss_ops = [tf.losses.mean_squared_error(pred, normed_next_obs) for pred in normed_pred_next_obs]

        # 3. training
        optimizer = tf.train.AdamOptimizer(lr, name='sp_optimizer')
        self.train_ops = [optimizer.minimize(loss, var_list=model.trainable_vars) for loss, model in zip(self.loss_ops, state_predictors)]

    def get_exploration_bonus(self, obs, actions, next_obs):
        normed_pred_next_obs = self.sess.run(self.normed_pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})
        # bonus is variance among the state predictors and along the predicted state vector
        #   ie a/ incent disagreement among the state predictors (explore state they can't model well);
        #      b/ incent exploring diverse state vectors; eg left-right leg mid-stride having opposite signs is higher var than standing / legs in same position
        return self.bonus_scale * np.var(normed_pred_next_obs, axis=(1,2))[:,None]

    def select_best_action(self, obs, actions):
        # input is obs = (n_env, obs_dim); actions = (n_samples, n_env, action_dim)
        n_samples, n_env, action_dim = actions.shape

        # reshape inputs to (n_samples*n_env, *_dim)
        obs = np.tile(obs, (n_samples, 1, 1)).reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1, action_dim)
        pred_next_obs = self.sess.run(self.pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})  # (n_samples*n_env, n_state_predictors, obs_dim)

        # compute rewards
        height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw = np.split(pred_next_obs[:,:,self.v_tgt_field_size: self.v_tgt_field_size+9], 9, -1)
        x_vtgt_onehot, _, goal_dist = np.split(pred_next_obs[:,:,:self.v_tgt_field_size], [1, self.v_tgt_field_size - 1], axis=-1)  # split to produce 3 arrays of shape (n,), (n,) and (1,)  where n is half the pooled v_tgt_field
        rewards = RewardAugEnv.compute_rewards(x_vtgt_onehot, goal_dist, height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw, None, None, None, None, None, None)
        rewards = np.sum([v for v in rewards.values()], 0)  # (n_samples*n_env, n_state_predictors, 1)
        rewards = np.reshape(rewards, [n_samples, n_env, -1, 1])  # (n_samples, n_env, n_state_predictors, 1)
        rewards = np.sum(rewards, 2)  # sum over state predictors; out (n_samples, n_env, 1)

        actions = actions.reshape([n_samples, n_env, action_dim])
        best_actions = np.take_along_axis(actions, rewards.argmax(0)[None,...], 0)  # out (1, n_env, action_dim)
        return np.squeeze(best_actions, 0)

    def train(self, memory, batch_size):
        losses = []
        for loss_op, train_op in zip(self.loss_ops, self.train_ops):
            batch = memory.sample(batch_size)
            loss, _ = self.sess.run([loss_op, train_op],
                        feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.next_obs_ph: batch.next_obs})
            losses.append(loss)
        return np.mean(loss)


class ExemplarExploration(BaseExploration):
    def __init__(self, *args, n_state_samples, bonus_scale, **kwargs):
        super().__init__(*args)
        self.n_state_samples = n_state_samples
        self.bonus_scale = bonus_scale

    def compute_distance(self, obs):
        return np.mean(np.sum((self.dataset[None,:,:] - obs[:,None,:])**2, -1), 1, keepdims=True)

    def initialize(self, *args, env=None):
        super().initialize(*args)
        # compile pose dataset given the random init poses given by the env
        dataset = []
        for i in range(self.n_state_samples // env.num_envs):
            dataset.append(env.reset())
        dataset = np.concatenate(dataset, 0)
        dataset = np.take(dataset, self.pose_idxs, -1)
        self.dataset_mean = np.mean(dataset, 0)
        self.dataset = dataset - self.dataset_mean
        self.dataset_mean_distance = self.compute_distance(np.atleast_2d(self.dataset_mean))
        print('Exemplar dataset initialized.')

    def get_exploration_bonus(self, obs, actions, next_obs):
        pose_next_obs = np.take(next_obs, self.pose_idxs, -1) - self.dataset_mean
        # compute l2 distance from exemplars
        distance = self.compute_distance(pose_next_obs)
        distance = np.clip(distance - self.dataset_mean_distance, 0, None)
        distance = np.tanh(0.1 * distance)
        return - self.bonus_scale * distance

    def select_best_action(self, obs, actions):
        return actions[0]

    def train(self, memory, batch_size):
        return 0


# --------------------
# defaults
# --------------------

def defaults(class_name=None):
    if class_name == 'DisagreementExploration':
        return {'n_state_predictors': 5,
                'state_predictor_hidden_sizes': (64, 64),
                'lr': 1e-3,
                'bonus_scale': 1}
    elif class_name == 'ExemplarExploration':
        return {'n_state_samples': 100,
                'bonus_scale': 1}

