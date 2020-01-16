from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def mlp(x, hidden_sizes, output_size, activation, output_activation, output_kernel_initializer, output_bias_initializer):
    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=output_size, activation=output_activation,
            kernel_initializer=output_kernel_initializer, bias_initializer=output_bias_initializer)


class Model:
    def __init__(self, name, hidden_sizes, output_size, activation=tf.nn.relu, output_activation=None,
                    output_kernel_initializer=None, output_bias_initializer=None):
        self.name = name
        self.network = partial(mlp, hidden_sizes=hidden_sizes, output_size=output_size,
                                activation=activation, output_activation=output_activation,
                                output_kernel_initializer=output_kernel_initializer,
                                output_bias_initializer=output_bias_initializer)

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
    def __init__(self, *args, **kwargs):
        output_size = kwargs.pop('output_size')
        super().__init__(*args, output_size=2*output_size, **kwargs)

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


# --------------------
# Normalizing flow based policy
# --------------------

class FlowPolicy(tf.keras.Model):
    def __init__(self, name, hidden_sizes, output_size, activation=None, **kwargs):
        super().__init__()
        self.output_size = output_size

        # base distribution of the flow
        self.base_dist = tfp.distributions.Normal(loc=tf.zeros([output_size], tf.float32),
                                                  scale=tf.ones([output_size], tf.float32))

        # affine transform of state to condition the flow
#        self.affine = tf.keras.layers.Dense(2*output_size, kernel_initializer='zeros', bias_initializer='zeros')
        self.affine = Model('base', hidden_sizes=hidden_sizes, output_size=2*output_size, **kwargs)

        # normalizing flow on top of the base distribution
#        self.flow = BNAF(hidden_sizes, output_size)
        self.flow = BNAF([4*output_size for _ in range(2)], output_size)

    def call(self, obs, n_samples=1):
        # sample actions from base distribution
        raw_actions = self.base_dist.sample([n_samples, tf.shape(obs)[0]])  # (n_samples, n_obs, action_dim)

        # affine transform conditions on state
        mu, logstd = tf.split(self.affine(obs), num_or_size_splits=2, axis=1)
        logstd = tf.clip_by_value(logstd, -20, 2)
        actions = mu[None,...] + tf.exp(logstd)[None,...] * raw_actions
        actions = tf.reshape(actions, [-1, self.output_size])
#        actions, sum_logdet = self.tanh((actions, logstd))

        # apply flow
        actions, sum_logdet = self.flow(actions)
        sum_logdet += logstd
        logprob = tf.reduce_sum(self.base_dist.log_prob(actions) - sum_logdet, 1)

        # reshape to (n_samples, B, action_dim)
        actions = tf.reshape(actions, [n_samples, -1, self.output_size])
        logprob = tf.reshape(logprob, [n_samples, -1, 1])
        return actions, logprob # (n_samples, B, action_dim) and (n_samples, B, 1)

    @property
    def flow_trainable_vars(self):
        return self.flow.trainable_variables

    @property
    def trainable_vars(self):
        return self.affine.trainable_vars

# --------------------
# BNAF implementation
# --------------------

class MaskedLinear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, data_dim):
        super().__init__()
        self.data_dim = data_dim

        weight = np.zeros((out_features, in_features))
        mask_d = np.zeros_like(weight)
        mask_o = np.zeros_like(weight)
        for i in range(data_dim):
            # select block slices
            h     = slice(i * out_features // data_dim, (i+1) * out_features // data_dim)
            w     = slice(i * in_features // data_dim,  (i+1) * in_features // data_dim)
            w_row = slice(0,                            (i+1) * in_features // data_dim)
            # initialize block-lower-triangular weight and construct block diagonal mask_d and lower triangular mask_o
            fan_in = in_features // data_dim
            weight[h, w_row] = np.random.uniform(-np.sqrt(1/fan_in), np.sqrt(1/fan_in), weight[h, w_row].shape)
            mask_d[h, w] = 1
            mask_o[h, w_row] = 1

        mask_o = mask_o - mask_d

        self.weight = self.add_variable('weight', weight.shape, tf.float32, initializer=
                        tf.initializers.constant(weight))
        self.logg = self.add_variable('logg', [out_features, 1], tf.float32, initializer=
                        tf.initializers.constant(np.log(np.random.rand(out_features, 1))))
        self.bias = self.add_variable('bias', [out_features], tf.float32, initializer=tf.initializers.constant(
                        np.random.uniform(-1/np.sqrt(in_features), 1/np.sqrt(in_features))))
        self.mask_d = self.add_variable('mask_d', mask_d.shape, tf.float32, initializer=
                        tf.initializers.constant(mask_d), trainable=False)
        self.mask_o = self.add_variable('mask_o', mask_o.shape, tf.float32, initializer=
                        tf.initializers.constant(mask_o), trainable=False)

    def call(self, inputs):
        x, sum_logdets = inputs
        # 1. compute BNAF masked weight eq 8
        v = tf.exp(self.weight) * self.mask_d + self.weight * self.mask_o
        # 2. weight normalization
        v_norm = tf.norm(v, ord=2, axis=1, keepdims=True)
        w = tf.exp(self.logg) * v / v_norm
        # 3. compute output and logdet of the layer
        out = tf.matmul(x, w, transpose_b=True) + self.bias
        logdet = self.logg + self.weight - 0.5 * tf.log(v_norm**2)
        logdet = tf.boolean_mask(logdet, tf.cast(self.mask_d, tf.uint8))
        logdet = tf.reshape(logdet, [1, self.data_dim, out.shape[1]//self.data_dim, x.shape[1]//self.data_dim])
        logdet = tf.tile(logdet, [tf.shape(x)[0], 1, 1, 1])  # output (B, data_dim, out_dim // data_dim, in_dim // data_dim)

        # 4. sum with sum_logdets from layers before (BNAF section 3.3)
        # Compute log det jacobian of the flow (eq 9, 10, 11) using log-matrix multiplication of the different layers.
        # Specifically for two successive MaskedLinear layers A -> B with logdets A and B of shapes
        #  logdet A is (B, data_dim, outA_dim, inA_dim)
        #  logdet B is (B, data_dim, outB_dim, inB_dim) where outA_dim = inB_dim
        #
        #  Note -- in the first layer, inA_dim = in_features//data_dim = 1 since in_features == data_dim.
        #            thus logdet A is (B, data_dim, outA_dim, 1)
        #
        #  Then:
        #  logsumexp(A.transpose(2,3) + B) = logsumexp( (B, data_dim, 1, outA_dim) + (B, data_dim, outB_dim, inB_dim) , dim=-1)
        #                                  = logsumexp( (B, data_dim, 1, outA_dim) + (B, data_dim, outB_dim, outA_dim), dim=-1)
        #                                  = logsumexp( (B, data_dim, outB_dim, outA_dim), dim=-1) where dim2 of tensor1 is broadcasted
        #                                  = (B, data_dim, outB_dim, 1)

        sum_logdets = tf.math.reduce_logsumexp(tf.transpose(sum_logdets, [0,1,3,2]) + logdet, axis=-1, keepdims=True)

        return out, sum_logdets

class Tanh(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x, sum_logdets = inputs
        # derivation of logdet:
        # d/dx tanh = 1 / cosh^2; cosh = (1 + exp(-2x)) / (2*exp(-x))
        # log d/dx tanh = - 2 * log cosh = -2 * (x - log 2 + log(1 + exp(-2x)))
        logdet = -2 * (x - np.log(2) + tf.nn.softplus(-2.*x))
        sum_logdets = sum_logdets + tf.reshape(logdet, tf.shape(sum_logdets))
        return tf.tanh(x), sum_logdets

class FlowSequential(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        gated = kwargs.pop('gated')
        if gated:
            self.gate = tf.Variable(tf.random_normal([1]))
        super().__init__(*args, **kwargs)

    def call(self, x):
        out = x
        sum_logdets = tf.zeros([1, tf.shape(x)[-1], 1, 1], tf.float32)
        for l in self.layers:
            out, sum_logdets = l((out, sum_logdets))
        if hasattr(self, 'gate'):
            gate = tf.sigmoid(self.gate)
            out = gate * out + (1 - gate) * x
            sum_logdets = tf.nn.softplus(sum_logdets + self.gate) - tf.nn.softplus(self.gate)
        return out, tf.squeeze(sum_logdets, [2, 3])

class BNAF(tf.keras.Model):
    def __init__(self, hidden_sizes, output_size, n_flows=1):
        assert all(h % output_size == 0 for h in hidden_sizes), 'Size of hidden layer must divide output (actions) dim.'
        super().__init__()

        # construct model
        self.flow = []
        for i in range(n_flows):
            modules = []
            modules += [MaskedLinear(output_size, hidden_sizes[0], output_size), Tanh()]
            for h in hidden_sizes:
                modules += [MaskedLinear(h, h, output_size), Tanh()]
            modules += [MaskedLinear(h, output_size, output_size)]
            modules += (i + 1 == n_flows)*[Tanh()]  # final output only -- policy outputs actions in [-1,1]
            self.flow += [FlowSequential(modules, gated=True if i + 1 != n_flows else False)]

    def call(self, x):
        sum_logdets = 0
        for i, f in enumerate(self.flow):
            x, logdet = f(x)
            x = x[:, ::-1] if i + 1 < len(self.flow) else x   # reverse ordering between intermediate flow steps
            sum_logdets += logdet
        return x, sum_logdets
