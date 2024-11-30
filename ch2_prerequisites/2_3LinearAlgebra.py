import tensorflow as tf

# ################################
#           1.标量
# ################################
x = tf.constant(3.0)
y = tf.constant(2.0)
print(x + y)
print(x * y)
print(x / y)
print(x**y)
print("-------------1-----------------")

# ################################
#           2.向量
# ################################
x = tf.range(4)
print(x)
print(x[3])
print("-------------2-----------------")

# ################################
#           3.矩阵
# ################################
A = tf.reshape(tf.range(20), (5, 4))
print(A)
print(tf.transpose(A))
print(A[3, :])
print("-------------3-----------------")

# ################################
#           4.张量
# ################################
X = tf.reshape(tf.range(24), (2, 3, 4))
print(X)
print("-------------4-----------------")

# ################################
#           5.张量算法的基本性质
# ################################
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # 不能通过分配新内存将A克隆到B
print(A)
print(A + B)

# 两个矩阵的按元素乘法称为Hadamard积（Hadamard product）
print(A*B)

# 将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
print(a + X)
print((a*X))
print("-------------5-----------------")

# ################################
#           6.降维
# ################################
x = tf.range(4, dtype=tf.float32)
print(tf.reduce_sum(x))

# 默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。 我们还可以指定张量沿哪一个轴来通过求和降低维度。
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
A_sum_axis0 = tf.reduce_sum(A, axis=0) #按列求和降维
print(A_sum_axis0)
A_sum_axis1 = tf.reduce_sum(A, axis=1) #按行求和降维
print(A_sum_axis1)

print(tf.reduce_sum(A, axis=[0, 1]))  # 结果和tf.reduce_sum(A)相同

A_mean_axis0 = tf.reduce_mean(A, axis=0) #按列求平均值降维
print(A_mean_axis0)

# 非降维求和
print(A_sum_axis1)
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
print(sum_A)

# 如果我们想沿某个轴计算A元素的累积总和，比如axis=0（按行计算），可以调用cumsum函数。此函数不会沿任何轴降低输入张量的维度。
print(tf.cumsum(A, axis=0))
print("-------------6-----------------")

# ################################
#           7.点积
# ################################
x = tf.range(4, dtype=tf.float32)
y = tf.ones(4, dtype=tf.float32)
print(tf.tensordot(x, y, axes=1))
print(tf.reduce_sum(x * y))
print("-------------7-----------------")

# ################################
#           8.矩阵-向量积
# ################################
x = tf.range(4, dtype=tf.float32)
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
print(tf.linalg.matvec(A, x))
print("-------------8-----------------")

# ################################
#           9.矩阵-矩阵乘法
# ################################
B = tf.ones((4, 3), tf.float32)
print(tf.matmul(A, B))
print("-------------8-----------------")

# ################################
#           10.范数
# ################################
u = tf.constant([3.0, -4.0])
tf.norm(u)
tf.reduce_sum(tf.abs(u))
tf.norm(tf.ones((4, 9)))
