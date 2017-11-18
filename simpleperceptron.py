import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

x1 = np.arange(-6.0, 4.0, 0.1)

my_file = open("dot.txt", 'r')
data = my_file.readlines()

Types = [line.split(" ") for line in data]

x_0 = [Type[0] for Type in Types]
x_1 = [Type[1] for Type in Types]
label_ = [Type[2] for Type in Types]

x0 = np.array([])
x1 = np.array([])
label = np.array([])

for x_0, x_1, label_ in zip(x_0, x_1, label_):
    x_0 = float(x_0)
    x_1 = float(x_1)
    label_ = float(label_)

    if label_==1:
        plt.plot(x_0, x_1, 'ro')
    elif label_==0:
        plt.plot(x_0, x_1, 'bo')

    x0 = np.append(x0, x_0)
    x1 = np.append(x1, x_1)
    label =np.append(label, label_)


w = [0.5, 0.2]
#学習率
n = 0.5
L=0
ims = []

for i in range(len(label)):

        a = w[0]*x0[i] + w[1]*x1[i]
        #シグモイド関数
        h = 1 / (1 + np.exp(-a))
        #二乗和誤差
        L_0 = label[i] - h
        L_1= 0.5*(L_0**2)

        #シグモイド関数の微分 y' = y*(1-y)
        # w = w - 学習率n * 誤差関数の微分
        w[0] = w[0] - n * (L_0 * h * (1-h)**2 )
        w[1] = w[1] - n * (L_0 * h * (1-h)**2 )
        #L = L + L_1

        x1 = (w[1]*x0) / w[0]

        im = plt.plot(x0, x1)
        ims.append(im)

ani = animation.ArtistAnimation(fig, ims, interval=1, repeat=False)
plt.xlim([-6.0, 8.0])
plt.ylim([-5.0,10])
plt.show()

ani.save("simple.gif")
