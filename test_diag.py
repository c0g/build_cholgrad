import tensorflow as tf
import numpy as np

arr = np.random.randn(20,20).astype(np.float32)
mod = tf.load_op_library('/home/tron/nicksontf_tests/get_diag.so')
a = tf.placeholder(tf.float32, [20,20])
d = mod.get_diag(a)
print arr
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print sess.run(d, {a:arr})
