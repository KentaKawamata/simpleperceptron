import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
# 0から6まで0.1刻みで生成
x1 = np.arange(-6.0, 4.0, 0.1)

#ファイルを開いて一行ずつ読み込み
my_file = open("input.txt", 'r')
data = my_file.readlines()




#line.split("")にて値と値の間にスペースを設ける
#x座標点,y座標点,信号yの3つを一組として,1行ずつに分割
#上記をfor文で全ての値に対して行う
Types = [line.split(" ") for line in data]




#データ全体...Types
#データを1行単位にしたもの...Type
#x_0...Typesの各行から最初の値のみ抽出
#x_1...Typesの各行から2番目の値のみ抽出
#y_0...Typesの各行から3番目の値のみ抽出
x_0 = [Type[0] for Type in Types]
x_1 = [Type[1] for Type in Types]
y_0 = [Type[2] for Type in Types]



#NumPy配列x0, x1, y を定義
x0 = np.array([])
x1 = np.array([])
y = np.array([])


for x_0, x_1, y_0 in zip(x_0, x_1, y_0):
    x_0 = float(x_0)
    x_1 = float(x_1)
    y_0 = float(y_0)

    if y_0==1:
        plt.plot(x_0, x_1, 'ro')
    elif y_0==0:
        plt.plot(x_0, x_1, 'bo')

    x0 = np.append(x0, x_0)
    x1 = np.append(x1, x_1)
    y =np.append(y, y_0)

#print(x0)
#print(x1)



#各種重み
w = [0.9, 0.1]
n = 1.6
L=0

ims = []

for i in range(len(y)):

        a = w[0]*x0[i] + w[1]*x1[i]
        #シグモイド関数
        y_0 = 1 / (1 + np.exp(-a))
        #二乗和誤差
        L_0 = (y[i]-y_0)**2
        L_1= 0.5*L_0

        w[0] = w[0] - n*L_0 * L_1*(1 - L_1)
        w[1] = w[1] - n*L_0 * L_1*(1 - L_1)
        L = L + L_1

        x2 = (w[0]*x1) / w[1]

        im = plt.plot(x1, x2)
        ims.append(im)

#アニメーション作成
ani = animation.ArtistAnimation(fig, ims, interval=1)

plt.xlim([-6.0, 8.0])
plt.ylim([-5.0,10])

plt.show()
