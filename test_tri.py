import tensorflow as tf
import numpy as np

arr = np.random.randn(5,5).astype(np.float32)
print arr
mod = tf.load_op_library('/home/tron/nicksontf_tests/triangle.so')
a = tf.placeholder(tf.float32, arr.shape)
d = mod.triangle(a, 'upper')
print arr
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print sess.run(d, {a:arr})
