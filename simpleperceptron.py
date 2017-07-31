import numpy as np
import matplotlib.pyplot as plt

my_file = open("input.txt", 'r')
data = my_file.readlines()   #一行ずつ読み込み

Types = [line.split(" ") for line in data]
x_1 = [Type[0] for Type in Types]
x_2 = [Type[1] for Type in Types]
y_0 = [Type[2] for Type in Types]
#x1[len(data)] =()
#x2[len(data)] =()
#y[len(data)] =()
#x1x2y = []
x1 = np.array([])
x2 = np.array([])
y = np.array([])
for x_1, x_2, y_0 in zip(x_1, x_2, y_0):
    x_1 = float(x_1)
    x_2 = float(x_2)
    y_0 = float(y_0)

    if y_0==1:
        plt.plot(x_1, x_2, 'ro')
    elif y_0==0:
        plt.plot(x_1, x_2, 'bo')

    x1 = np.append(x1, x_1)
    x2 = np.append(x2, x_2)
    y =np.append(y, y_0)

print(x1)
print(x2)

#for x1, x2, y in zip(x1, x2, y):



#x1 = np.arange(-8.0, 8.0, 0.1) # 0から6まで0.1刻みで生成
w = [0.9, 0.1]

# w[0]*x1 + w[1]*x2 + y
#x2 = (w[0]*x1 + y) / w[1]
#y001 = 1 / (1 + exp(-a))
#L = 0.5*((y-y001)**2)
n = 1.6
"""f = ax + by + c
wm = np.matrix([a, b])
xm = np.matrix([x, y])
g = np.dot(wm, xm) + c"""
L=0
#while abs(y-y001) > 1:
#x2 = -1*(w[0]*x1 + y) / w[1]
#plt.plot(x1, x2)
#plt.show()
ims = []

for i in range(len(y)):

        a = w[0]*x1[i] + w[1]*x2[i]
        y_0 = 1 / (1 + np.exp(-a))
        L_0 = (y[i]-y_0)**2
        L_1= 0.5*L_0
        #L = 0.5*((y - 1/(1+np.exp(- w[0]*x1 + w[1]*x2 + y)))**2)
        w[0] = w[0] - n*L_0 * L_1*(1 - L_1)
        w[1] = w[1] - n*L_0 * L_1*(1 - L_1)
        L = L + L_1

        #x1 = np.arange(-6.0, 6.0, 0.1) # 0から6まで0.1刻みで生成
        #x2 = (w[0]*x1) / w[1]
        #plt.plot(x1, x2)
        #ims.append(im)

    #"""
    #    t = -1
    #E0 = max(0, -(t*w[0]*1))
    #if E0!=0:
    #    w[0] = w[0] - R*t*1
    #E1 = max(0, -(t*w[0]*xf))
    #if E1!=0:
    #    w[0] = w[0] - R*t*x
    #E2 = max(0, -(t*w[0]*yf))
    #if E2!=0:
    #    w[0] = w[0] - R*t*y"""

    #wm = np.matrix([a, b])
    #xm = np.matrix([xf, yf])
    #if yf*(np.dot(wm, xm)+c) < 0:
print(L)

x1 = np.arange(-6.0, 6.0, 0.1) # 0から6まで0.1刻みで生成
x2 = (w[0]*x1) / w[1]
plt.plot(x1, x2)
plt.show()

#for i in range(len(im)):
#    plt.show()
#    plt.clf()           # 画面初期化
























































    #x = int(data[j])          # 配列の1つ目は、整数に変換して、変数xに代入
    #y = int(data[j+1])          # 配列の2つ目は、整数に変換して、変数yに代入
    #w = int(data[j+2])
    #j+=3
    #print(x, y, z)
    #print(data[i])
    #print(data[i])

"""point = [[]for i1 in range(len(data))]

for i in range(len(data)):
    data1 = data.strip()         #\n削除
    data1 = data1.split(" ")      #スペース削除
    #print(data)
    x = float(data1[i])          # 配列の1つ目は、整数に変換して、変数xに代入
    y = float(data1[i])          # 配列の2つ目は、整数に変換して、変数yに代入
    w = float(data1[i])
    point[i].append(x)
    point[i].append(y)
    point[i].append(w)

for i in range(len(point)):
    print(point[i])
my_file.close()
#print(point[0][1])
#for i in range(len(point)):
    #plt.plot(point[i][0], point[i][1], 'ro')

x = []
y = []
for i in point:
    x.append(i[0])
    y.append(i[1])
    print(i[0])

plt.plot(x, y)
plt.show()"""
