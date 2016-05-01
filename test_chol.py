import tensorflow as tf
import numpy as np

arr = np.random.randn(5,5).astype(np.float32)
arr = arr.T.dot(arr)
print np.linalg.cholesky(arr)
mod = tf.load_op_library('/home/tron/nicksontf_tests/gpu_cholesky_op.so')
a = tf.placeholder(tf.float32, arr.shape)
d = mod.gpu_cholesky(a)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
with tf.device("/cpu:0"):
    print sess.run(d, {a:arr})
with tf.device("/gpu:0"):
    print sess.run(d, {a:arr})
