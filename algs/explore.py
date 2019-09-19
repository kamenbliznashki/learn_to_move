import numpy as np
import tensorflow as tf

from algs.models import Model

try:
    from mpi4py import MPI
    from mpi_adam import MpiAdam, flatgrad
    from mpi_utils import mpi_moments
except ImportError:
    MPI = None

class DisagreementExploration:
    """ Implementation of Pathak et al. 'Self Supervised Exploration via Disagreement' """
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

    def train(self, batch):
        loss, _ = self.sess.run([self.loss, self.train_op],
                feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.next_obs_ph: batch.next_obs})
        return loss



class BootstrappedAgent:
    """ Implementation of Osband et al. 'Randomized Prior Functions for Deep RL' """
    def __init__(self, agent_cls, n_heads, args, kwargs):
        # NOTE TODO -- can adjust alg_args across heads here e.g.: expl noise, discount, learning rate -- e.g. uniform sample
        kwargs.pop('discount', None)
        kwargs.pop('lr', None)

        discount = np.random.uniform(0.95, 0.98, (n_heads,))
        lr = np.random.uniform(1e-5, 5e-3, (n_heads,))

        # broadcast to nodes so local copies of each head use same hyperparams
        if MPI is not None:
            MPI.COMM_WORLD.Bcast(discount, root=0)
            MPI.COMM_WORLD.Bcast(lr, root=0)

        self.heads = [agent_cls(i, *args, discount=discount[i], lr=lr[i], **kwargs) for i in range(n_heads)]

    def __getitem__(self, k):
        return self.heads[k]

    def get_actions(self, obs, head_idx=None):
        if head_idx is not None:
            # specific head acts
            actions = self.heads[head_idx].get_actions(obs)
        else:
            # ensemble voting -- get actions and q_values for each head, select actions with max q value
            actions = np.stack([head.get_actions(obs) for head in self.heads], 0)                               # (n_heads, n_env, action_dim)
            qs = np.stack([head.get_action_value(obs, action) for head, action in zip(self.heads, actions)], 0) # (n_heads, n_env, 1)
            # normalize q values across the heads
            qs /= qs.sum(0, keepdims=True)
            # select action at highest q value
            actions = np.take_along_axis(actions, qs.argmax(0)[None,...], axis=0).squeeze(0)  # (n_env, action_dim)
        return actions

    def initialize(self, sess):
        # setup global step
        self.global_step_update_op = tf.assign_add(tf.train.get_or_create_global_step(), 1)
        # initialize all heads
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        for head in self.heads:
            head.initialize(sess)

    def train(self, ensemble_memory, batch_size):
        for head, memory in zip(self.heads, ensemble_memory):
            batch = memory.sample(batch_size)
            head.train(batch)
        # update global step
        self.sess.run(self.global_step_update_op)

    def update_target_net(self):
        for head in self.heads:
            head.update_target_net()


# --------------------
# defaults
# --------------------

def defaults(class_name=None):
    if class_name == 'DisagreementExploration':
        return {'n_state_predictors': 5,
                'state_predictor_hidden_sizes': (64, 64),
                'lr': 1e-3,
                'bonus_scale': 10}

