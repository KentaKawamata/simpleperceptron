import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

X0 = np.arange(-6.0, 8.0, 0.1)

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
    label = np.append(label, label_)

X = np.empty((0, 2))

for i in range(len(label)):

    X = np.append(X, np.array([[x0[i], x1[i]]]), axis=0)

W = np.array([[0.5], [0.2]])
#学習率
n = 0.5
L=0
ims = []

for i in range(len(label)):

        a = np.dot(X[i], W)
        #シグモイド関数
        h = 1 / (1 + np.exp(-a))
        #二乗和誤差
        L_0 = label[i] - h
        L_1= 0.5*(L_0**2)

        #シグモイド関数の微分 y' = y*(1-y)
        # w = w - 学習率n * 誤差関数の微分
        W[0] = W[0] - n * (L_0 * h * (1-h)**2 )
        W[1] = W[1] - n * (L_0 * h * (1-h)**2 )
        #L = L + L_1

        X1 = (W[1]*X0) / W[0]

        im = plt.plot(X0, X1)
        ims.append(im)

ani = animation.ArtistAnimation(fig, ims, interval=1, repeat=False)

plt.xlim([-6.0, 8.0])
plt.ylim([-5.0,10])

plt.show()
