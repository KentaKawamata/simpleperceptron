import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

fig = plt.figure()
X0 = np.arange(-8.0, 8.0, 0.1)

tf.set_random_seed(0)

w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w)+b)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

for i in range(4):
    
    if Y[i] == 0:
        plt.plot(X[i][0], X[i][1], 'ro')
    
    elif Y[i] == 1:
        plt.plot(X[i][0], X[i][1], 'bo')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(200):
    sess.run(train_step, feed_dict={x: X, t: Y })

    
classified = correct_prediction.eval(session=sess, feed_dict={x: X, t: Y })
print(classified)

prob = y.eval(session=sess, feed_dict={x: X, t: Y})
print(prob)

print('w:', sess.run(w))
print('b:', sess.run(b))


X1 = -X0 * (sess.run(w[0]) / sess.run(w[1])) - sess.run(b)
plt.plot(X0, X1)

plt.xlim([-8.0, 8.0])
plt.ylim([-10.0, 10.0])

plt.show()
