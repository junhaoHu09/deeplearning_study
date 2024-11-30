import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x = tf.Variable(x)
# 把所有计算记录在磁带上（不太懂tensorflow）
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
print(y)

x_grad = t.gradient(y, x)
print(x_grad)

with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
