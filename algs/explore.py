import numpy as np
import tensorflow as tf

from algs.models import Model


class DisagreementExploration:
    def __init__(self, observation_shape, action_shape, *,
                    lr=1e-3, state_predictor_hidden_sizes=[64, 64], n_state_predictors=5):
        self.lr = lr

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')

        # obs indices to model using the state predictors; NOTE idx 0 here is in the obs vector after the v_tgt_field, ie starting with pelvis
        idxs = np.array([*list(range(9)),                        # pelvis
                         *list(range(12,16)),                    # joints r leg
                         *list(range(20+3*11+3,20+3*11+3+4))])   # joints l leg
        idxs += 2*3  # offset for v_tgt_field -- NOTE NOTE NOTE -- this should match the obs2vec offset (if excluding vtgt) and the poolvtgtenv given the pooling size

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

        # 2. loss
        self.loss = tf.reduce_sum([tf.losses.mean_squared_error(pred, normed_next_obs) for pred in normed_pred_next_obs])

        # 3. training
        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
        self.train_op = optimizer.minimize(self.loss, var_list=[var for model in state_predictors for var in model.vars])

    def initialize(self, sess):
        self.sess = sess

    def get_exploration_bonus(self, obs, actions, scale=100):
        normed_pred_next_obs = self.sess.run(self.normed_pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})
        # bonus is variance among the state predictors and along the predicted state vector
        #   ie a/ incent disagreement among the state predictors (explore state they can't model well);
        #      b/ incent exploring diverse state vectors; eg left-right leg mid-stride having opposite signs is higher var than standing / legs in same position
        return scale * np.var(normed_pred_next_obs, axis=(1,2))[:,None]

    def train(self, batch):
        loss, _ = self.sess.run([self.loss, self.train_op],
                feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.next_obs_ph: batch.next_obs})
        return loss


