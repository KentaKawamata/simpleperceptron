import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


fig = plt.figure()
X_0 = np.arange(-4.0, 14.0, 0.1)

#入力データの次元
M = 2
#クラス数
K = 3
#クラスごとのデータ数
n = 100
#全データ数
N = n * K

X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

for i in range(n):
    plt.plot(X1[i][0], X1[i][1], 'bo', marker='o')
    plt.plot(X2[i][0], X2[i][1], 'ro', marker='^')
    plt.plot(X3[i][0], X3[i][1], 'go', marker='s')

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#交差エントロピー誤差関数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
#確率的勾配降下法による最小化
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

batch_size = 50
n_batches = N


for epoch in range(20):
    X_, Y_ = shuffle(X, Y)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        #[start:end]は、startオフセットからend-1オフセットまでのシーケンスを抽出する
        sess.run(train_step, feed_dict={x: X_[start:end], t: Y_[start:end]})

X_, Y_ = shuffle(X, Y)

classified = correct_prediction.eval(session=sess, feed_dict={x: X_[0:10], t: Y_[0:10]})
prob = y.eval(session=sess, feed_dict={x: X_[0:10]})

print('classified')
print(classified)
print()
print('output probability:')
print(prob)
print()
print('W:')
print(sess.run(W))
print('b:')
print(sess.run(b))
#print(sess.run(W[1][2]))

##########DANGER############
#XX X_1_1 = ((sess.run(W[1][0])-sess.run(W[0][0]))*X_0 - sess.run(b[0]) + sess.run(b[1])) / (sess.run(W[0][1])-sess.run(W[1][1]))
##########DANGER############

X_1_1 = ((sess.run(W[0][1])-sess.run(W[0][0]))*X_0 - sess.run(b[0]) + sess.run(b[1])) / (sess.run(W[1][0])-sess.run(W[1][1]))
plt.plot(X_0, X_1_1)

X_1_2 = ((sess.run(W[0][2])-sess.run(W[0][1]))*X_0 - sess.run(b[1]) + sess.run(b[2])) / (sess.run(W[1][1])-sess.run(W[1][2]))
plt.plot(X_0, X_1_2)

plt.xlim([-4.0, 14.0])
plt.ylim([-4.0, 14.0])
plt.show()
