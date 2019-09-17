from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def mlp(x, hidden_sizes, output_size, activation, output_activation):
    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=output_size, activation=output_activation)


class Model:
    def __init__(self, name, hidden_sizes, output_size, activation=tf.nn.relu, output_activation=None):
        self.name = name
        self.network = partial(mlp, hidden_sizes=hidden_sizes, output_size=output_size,
                                activation=activation, output_activation=output_activation)

    def __call__(self, obs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.network(obs)
        return x

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class GaussianPolicy(Model):
    def __call__(self, obs, n_samples=1):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.network(obs)
            mu, logstd = tf.split(x, num_or_size_splits=2, axis=1)
            logstd = tf.clip_by_value(logstd, -20, 2)
            dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logstd))

            # sample action and evaluate log prob (note: change of variabes formula logdet is added to the log probs)
            raw_actions = dist.sample(n_samples)  # out (n_samples, B, action_dim)
            actions = tf.tanh(raw_actions)
            log_probs = tf.expand_dims(dist.log_prob(raw_actions), -1)  # (n_samples, B, 1)
            log_probs -= tf.reduce_sum(-2 * (raw_actions - np.log(2) + tf.nn.softplus(-2*raw_actions)), -1, keepdims=True)

        return actions, log_probs  # (n_samples, B, action_dim) and (n_samples, B, 1)

    def get_action(self, obs):
        assert obs.shape[0] == 1
        return self.__call__(obs)[0]
