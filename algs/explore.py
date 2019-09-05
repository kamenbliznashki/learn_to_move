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

        # obs indices to pass to policy and q function
        # policy gets pelvis hight pitch roll, joints, fiber lengths -- no velocity
        idxs = np.array([*list(range(9)),                        # pelvis
                         *list(range(12,16)),                    # joints r leg
                         *list(range(20+3*11+3,20+3*11+3+4))])   # joints l leg
        idxs += 2*3  # offset for v_tgt_field

        # build graph
        # 1. networks
        state_predictors = [Model('state_predictor_{}'.format(i), hidden_sizes=state_predictor_hidden_sizes, output_size=len(idxs)) for i in range(n_state_predictors)]
        self.pred_next_obs = [model(tf.concat([tf.gather(self.obs_ph, idxs, axis=1), self.actions_ph], 1)) for model in state_predictors]

        # 2. loss on critics and actor
        self.loss = tf.reduce_mean([tf.losses.mean_squared_error(pred, tf.gather(self.next_obs_ph, idxs, axis=1))
                                    for pred in self.pred_next_obs])

        # 3. training
        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
        self.train_op = optimizer.minimize(self.loss, var_list=[var for model in state_predictors for var in model.vars])

    def initialize(self, sess):
        self.sess = sess

    def get_exploration_bonus(self, obs, actions):
        pred_next_obs = self.sess.run(self.pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})
        return np.var(pred_next_obs)

    def train(self, batch):
        loss, _ = self.sess.run([self.loss, self.train_op],
                feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.next_obs_ph: batch.next_obs})
        return loss


