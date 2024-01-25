import tensorflow as tf

@tf.function
def my_func(x):
  print('Tracing.\n')
  return tf.reduce_sum(x)

x = tf.constant([1, 2, 3])
my_func(x)

x = tf.constant([10, 9, 8])
my_func(x)

x = tf.constant([10.0, 9.1, 8.2], dtype=tf.float32)
my_func(x)