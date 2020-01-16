""" Source openai baselines """

import tensorflow.compat.v1 as tf
import numpy as np
from mpi4py import MPI


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat([tf.reshape(g if g is not None else tf.zeros_like(v), [-1]) for v, g in zip(var_list, grads)], axis=0)

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = [v.get_shape().as_list() for v in var_list]
        total_size = int(sum([np.prod(shape) for shape in shapes]))

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = int(np.prod(shape))
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        tf.get_default_session().run(self.op, feed_dict={self.theta: theta})

class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat([tf.reshape(v, [-1]) for v in var_list], axis=0)

    def __call__(self):
        return tf.get_default_session().run(self.op)

class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = int(sum(np.prod(v.get_shape().as_list()) for v in var_list))
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = SetFromFlat(var_list)
        self.getflat = GetFlat(var_list)
        self.comm = MPI.COMM_WORLD

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        if self.comm is not None:
            globalg = np.zeros_like(localg)
            self.comm.Allreduce(localg, globalg, op=MPI.SUM)
            if self.scale_grad_by_procs:
                globalg /= self.comm.Get_size()
        else:
            globalg = np.copy(localg)

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        if self.comm is None:
            return
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm is None:
            return
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)


def test_MpiAdam():
    sess = tf.get_default_session()

    np.random.seed(0)
    tf.set_random_seed(0)

    a = tf.Variable(np.random.randn(3).astype('float32'))
    b = tf.Variable(np.random.randn(2,5).astype('float32'))
    loss = tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    stepsize = 1e-2
    update_op = tf.train.AdamOptimizer(stepsize).minimize(loss)

    sess.run(tf.global_variables_initializer())
    losslist_ref = []
    for i in range(10):
        l, _ = sess.run([loss, tf.group(update_op)])
        print(i, l)
        losslist_ref.append(l)

    tf.set_random_seed(0)
    sess.run(tf.global_variables_initializer())

    var_list = [a,b]
    adam = MpiAdam(var_list)

    losslist_test = []
    for i in range(10):
        l, g = sess.run([loss, flatgrad(loss, var_list)])
        adam.update(g, stepsize)
        print(i,l)
        losslist_test.append(l)

    np.testing.assert_allclose(np.array(losslist_ref), np.array(losslist_test), atol=1e-4)


if __name__ == '__main__':
    with tf.Session():
        test_MpiAdam()
