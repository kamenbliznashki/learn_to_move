import numpy as np
import tensorflow as tf

from algs.models import Model


class DisagreementExploration:
    def __init__(self, observation_shape, action_shape, *,
                    lr=1e-3, state_predictor_hidden_sizes=[64, 64], n_state_predictors=5, bonus_scale=100):
        self.lr = lr
        self.bonus_scale = bonus_scale

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')

        # obs indices to model using the state predictors; NOTE idx 0 here is in the obs vector after the v_tgt_field, ie starting with pelvis
        idxs = np.array([*list(range(9)),                        # pelvis       (9 obs)
                         *list(range(12,16)),                    # joints r leg (4 obs)
                         *list(range(20+3*11+3,20+3*11+3+4))])   # joints l leg (4 obs)
        idxs += 1*3  # offset for v_tgt_field -- NOTE NOTE NOTE -- this should match the obs2vec offset (if excluding vtgt) and the poolvtgtenv given the pooling size
        idxs = np.hstack([list(range(3)), idxs])
#        idxs = list(range(observation_shape[0]))  # state predictors model the full state space vec

        # build graph
        # 1. networks
        state_predictors = [Model('state_predictor_{}'.format(i), hidden_sizes=state_predictor_hidden_sizes, output_size=len(idxs)) for i in range(n_state_predictors)]

        # 2. state predictor outputs
        #   select obs indices to model using the state predictors
        obs = tf.gather(self.obs_ph, idxs, axis=1)
        next_obs = tf.gather(self.next_obs_ph, idxs, axis=1)

        #   whiten obs and next obs
        obs_mean, obs_var = tf.nn.moments(obs, axes=0)
        next_obs_mean, next_obs_var = tf.nn.moments(next_obs, axes=0)
        normed_obs = (obs - obs_mean) / (obs_var**0.5 + 1e-8)
        normed_next_obs = (next_obs - next_obs_mean) / (next_obs_var**0.5 + 1e-8)

        #   predict from zero-mean unit var obs; shift and scale result by obs mean and var
        normed_pred_next_obs = [model(tf.concat([normed_obs, self.actions_ph], 1)) for model in state_predictors]
        self.normed_pred_next_obs = tf.stack(normed_pred_next_obs, axis=1)  # (B, n_state_predictors, obs_dim)
        self.pred_next_obs = tf.stack(normed_pred_next_obs, axis=1) * (obs_var[None,None,:] + 1e-8) + obs_mean[None,None,:]  # (B, n_state_predictors, obs_dim)

        # 2. loss
        self.loss = tf.reduce_sum([tf.losses.mean_squared_error(pred, normed_next_obs) for pred in normed_pred_next_obs])

        # 3. training
        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
        self.train_op = optimizer.minimize(self.loss, var_list=[var for model in state_predictors for var in model.vars])

    def initialize(self, sess):
        self.sess = sess

    def get_exploration_bonus(self, obs, actions):
        normed_pred_next_obs = self.sess.run(self.normed_pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})
        # bonus is variance among the state predictors and along the predicted state vector
        #   ie a/ incent disagreement among the state predictors (explore state they can't model well);
        #      b/ incent exploring diverse state vectors; eg left-right leg mid-stride having opposite signs is higher var than standing / legs in same position
        return self.bonus_scale * np.var(normed_pred_next_obs, axis=(1,2))[:,None]

    def select_best_action(self, obs, actions):
        # input is obs = (batch_size, obs_dim); actions = (n_samples, batch_size, action_dim)
        n_samples, B, action_dim = actions.shape

        # reshape inputs to (n_samples*batch_size, *_dim)
        obs = np.tile(obs, (n_samples, 1, 1)).reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1, action_dim)
        pred_next_obs = self.sess.run(self.pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})  # (n_samples*batch_size, obs_dim)

        # compute rewards -- NOTE -- match this to the idx selection that is input to the state predictors
        height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw = np.split(pred_next_obs[:,:,3:3+9], 9, -1)
        rewards = {}
        rewards['pitch'] = - 1 * np.clip(pitch * dpitch, a_min=0, a_max=None) # if in different direction ie counteracting ie diff signs, then clamped to 0, otherwise positive penalty
        rewards['roll']  = - 1 * np.clip(roll * droll, a_min=0, a_max=None)
        rewards['dx'] = 3 * np.clip(abs(dy)/abs(dx), 1, None) * np.tanh(dx)
        rewards['dy'] = - 2 * np.tanh(2*dy)**2
        rewards['dz'] = - np.tanh(dz)**2
        rewards['height'] = np.where(height > 0.7, np.zeros_like(height), -5 * np.ones_like(height))
        rewards = np.sum([v for v in rewards.values()], 0)  # (n_samples*batch_size, n_state_predictors, 1)
        rewards = np.reshape(rewards, [n_samples, B, pred_next_obs.shape[1], 1])
        rewards = np.sum(rewards, 2)  # sum over state predictors; out (n_samples, B, 1)

        actions = actions.reshape([n_samples, B, action_dim])
        best_actions = np.take_along_axis(actions, rewards.argmax(0)[None,...], 0)  # out (1, B, action_dim)
        return np.squeeze(best_actions, 0)

    def train(self, batch):
        loss, _ = self.sess.run([self.loss, self.train_op],
                feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.next_obs_ph: batch.next_obs})
        return loss


# --------------------
# defaults
# --------------------

def defaults(class_name=None):
    if class_name == 'DisagreementExploration':
        return {'n_state_predictors': 5,
                'state_predictor_hidden_sizes': (64, 64),
                'lr': 1e-3,
                'bonus_scale': 10}

