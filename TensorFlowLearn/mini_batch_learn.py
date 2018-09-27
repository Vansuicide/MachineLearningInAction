# -*-coding:utf-8-*-
# fetch_california_housing 数据集包含9个变量的20640个观测值，
# 目标变量为平均房屋价，
# 特征包括：平均收入、房屋平均年龄、平均房间、平均卧室、人口、平均占用、维度和经度。

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()  # 获取放假数据
m, n = housing.data.shape  # 获取数据维度，矩阵的行列长度

scalar = StandardScaler()  # 将特征进行标准归一化
scaled_housing_data = scalar.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]        # np.c_是连接的含义，加了一个全为1的列

learning_rate = 0.01

# X 和 y 为placeholder, 后面将要传进来的数据占位
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")  # None 就是没有限制，可以任意长
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

# 随机生成 theta，形状为(n+1, n)，元素在[-1.0, 1.0) 之间
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")

# 线性回归模型
y_pred = tf.matmul(X, theta, name="predictions")

# 损失用MSE
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# 初始化所有向量
init = tf.global_variables_initializer()

n_epochs = 10

# 每一批样本数设为100
batch_size = 100
n_batches = int(np.ceil(m/batch_size))  # 总样本数除以每一批的样本数，得到批的个数，要得到比它大的最近的整数


# 从整批中获取数据
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # 用于 random，每次可以得到不同的整数
    indices = np.random.randint(m, size=batch_size)  # 设置随机索引，最大值为m
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})

    best_theta = theta.eval()

print("Best theta:\n", best_theta)
