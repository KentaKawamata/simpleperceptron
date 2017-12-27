import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

with open("dot.txt", 'r') as fp:
    data = fp.readlines()

x0 = np.array([])
x1 = np.array([])
label = np.array([])


for line in data:
    
    line = line.split(" ")
    
    x_0 = float(line[0])
    x_1 = float(line[1])
    label_ = float(line[2])

    x0 = np.append(x0, x_0)
    x1 = np.append(x1, x_1)
    label = np.append(label, label_)

    if label_==1:
        plt.plot(x_0, x_1, 'ro')
    elif label_==0:
        plt.plot(x_0, x_1, 'bo')

w = [0.3, 0.1]
#学習率
n = 0.001
b = 0.6
ims = []
x = np.array([-6.0, 8.0])

for j in range(10):
    for i in range(len(label)):

        a = w[0]*x0[i] + w[1]*x1[i] + b
        #シグモイド関数
        h = 1 / (1 + np.exp(-a))
        #二乗和誤差
        L_0 = label[i] - h
        L_1= (L_0**2)/2

        #シグモイド関数の微分 y' = y*(1-y)
        # w = w - 学習率n * 誤差関数の微分
        w[0] = w[0] - n*(-L_0 * h*(1-h) *x0[i])
        w[1] = w[1] - n*(-L_0 * h*(1-h) *x1[i])
        b = b - n*(-L_0 *h*(1-h))

        y = -(w[1]*x) / w[0] - b

        im = plt.plot(x, y)
        ims.append(im)

print(w[0], w[1], b)
ani = animation.ArtistAnimation(fig, ims, interval=1, repeat=False)
plt.xlim([-6.0, 8.0])
plt.ylim([-5.0, 10.0])

plt.show()
ani.save('simple.gif', writer='imagemagick')
