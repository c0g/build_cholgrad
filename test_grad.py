import tensorflow as tf
import numpy as np
import chol_diff
import time

shape = (600,600)
arr = np.random.randn(*shape).astype(np.float64)
arrPD = arr.T.dot(arr)
npL = np.linalg.cholesky(arrPD)
np.set_printoptions(precision=4, suppress=True)
mod = tf.load_op_library('/home/tron/nicksontf_tests/gpu_cholgrad.so')
npAbar = np.tril(np.random.randn(*shape))
gpu_maybe = None
cpu_maybe = None
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.device('/gpu:0'):
        tfL = tf.placeholder(tf.float64, arr.shape)
        tfAbar = tf.placeholder(tf.float64, arr.shape)
        d = mod.gpu_chol_grad(tfL, tfAbar)
        t0 = time.time()
        gpu_maybe = sess.run(d, {tfL:npL, tfAbar:npAbar} )
        print "gpu {}".format(time.time() - t0)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.device('/cpu:0'):
        tfL = tf.placeholder(tf.float64, arr.shape)
        tfAbar = tf.placeholder(tf.float64, arr.shape)
        d = mod.gpu_chol_grad(tfL, tfAbar)
        t0 = time.time()
        cpu_maybe = sess.run(d, {tfL:npL, tfAbar:npAbar} )
        print "cpu {}".format(time.time() - t0)
correct = chol_diff.chol_rev(npL, npAbar)
correct = 0.5 *correct.T + 0.5*correct;
print gpu_maybe
print cpu_maybe
print correct
print np.sum(np.isclose(correct, gpu_maybe)) == np.prod(npAbar.shape)
print np.sum(np.isclose(correct, cpu_maybe)) == np.prod(npAbar.shape)
