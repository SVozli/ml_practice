import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

#  Tensors
print("Tensors \n")
print(x)
print(x.shape)
print(x.dtype)

print(x + x)

print(5 * x)

print(x @ tf.transpose(x))

print(tf.concat([x, x, x], axis=0))
print(tf.concat([x, x, x], axis=1))

print(tf.nn.softmax(x, axis=-1))

print(tf.reduce_sum(x))

print(tf.convert_to_tensor([1,2,3]))

print(tf.reduce_sum([1,2,3]))

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

print("\n")

#  Variables
print("Variables \n")

var = tf.Variable([0.0, 0.0, 0.0])
print(var)

print(var.assign([1, 2, 3]))

print(var.assign_add([1, 1, 1]))
print("\n")

#  Automatic differentiation
print("Automatic differentiation \n")

x = tf.Variable(1.0)

def f(x):
  y = x**2 + 2*x - 5
  return y

print(f(x))

with tf.GradientTape() as tape:
  y = f(x)

g_x = tape.gradient(y, x)  # g(x) = dy/dx

print(g_x)