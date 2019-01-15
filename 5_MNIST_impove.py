#!/usr/bin/python
# coding:utf-8

import tensorflow as tf
import input_data
# 加载数据
mnist = input_data.read_data_sets('Mnist_data', one_hot=True)

# 用占位符定义输入图片x与输出类别y_
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
# 将权重W和偏置b定义为变量,并初始化为0向量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 类别预测与损失函数
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 交叉熵损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 训练模型
# 用最速下降法让交叉熵下降,步长为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 运行交互计算图
sess = tf.InteractiveSession()
# 变量初始化
sess.run(tf.initialize_all_variables())
# 每次加载50个训练样本,然后执行一次train_step,通过feed_dict将x和y_用训练训练数据替代
for i in range(1000):
    # 每次加载50个样本,返回一个tuple,元素1为样本,元素2为标签
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 评估模型
# 用tf.equal来检测预测是与否真实标签匹配
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 将布尔值转换为浮点数来代表对错然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
