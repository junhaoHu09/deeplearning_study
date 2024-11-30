import tensorflow as tf

# ################################
#           1.数据创建
# ################################
a = tf.range(12)
print(a)
print(tf.size(a))
print(tf.reshape(a,(3,4)))

b = tf.ones((2,3,4))
print(b)
b = tf.zeros((2,3,4))

c = tf.random.normal(shape=[3,4])
print(c)

d = tf.constant([[1,2,3,4], [2,4,3,1], [4,3,2,1]])
print(d)
print("-------------1-----------------")

# ################################
#           2.运算符
# ################################
e = tf.constant([1.0,2,3,4])
print(e+e)
print(e-e)
print(e*e)
print(e/e)
print(e**e) #求幂运算
print(tf.exp(e)) #求e的幂

f = tf.reshape(tf.range(12, dtype=tf.float32),(3,4))
g = tf.constant([[2.0, 1,4,3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 沿行（轴-0，形状的第一个元素）连结两个矩阵
print(tf.concat([f,g], axis=0))
# 按列（轴-1，形状的第二个元素）连结两个矩阵
print(tf.concat([f,g], axis=1))

print(f == g)

print(tf.reduce_sum(f))

print("--------------2----------------")

# ################################
#           3.广播机制
# ################################
# 在某些情况下，即使形状不同，我们仍然可以通过调用 广播机制（broadcasting mechanism）来执行按元素操作。 这种机制的工作方式如下：
#     1.通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
#     2.对生成的数组执行按元素操作。
h = tf.reshape(tf.range(3), (3,1))
i = tf.reshape(tf.range(2), (1,2))
print(h+i)
print("-------------3-----------------")

# ################################
#           4.索引和切片
# ################################
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
print(X[-1])
print(X[1:3])
# TensorFlow中的Tensors是不可变的，也不能被赋值。
# TensorFlow中的Variables是支持赋值的可变容器。 请记住，TensorFlow中的梯度不会通过Variable反向传播。
# 给元素赋值
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
print(X_var)

X_var2 = tf.Variable(X)
X_var2[0:2,:].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
print(X_var2)
print("-----------4-------------------")

# ################################
#           5.节约内存
# ################################

# ################################
#           6.转化为其他python对象
# ################################