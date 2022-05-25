import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "./unet-Result/"

x0 = list(range(500, 5001, 100))
x1 = list(range(200, 2001, 100))
X, Y = np.meshgrid(x0, x1)

sumList = [[[0 for k in range(18)] for j in range(200, 2001, 100)] for i in range(500, 5001, 100)]
Z = [[0 for j in range(len(x0))] for i in range(len(x1))]
for filename in os.listdir(path):
	# print(filename)
	name = os.path.splitext(filename)[0]
	x, y = name.split("-")
	x, y = int(x), int(y)
	x, y = (x-500)//100, (y-200)//100

	
	if(name != "sidewindow" and name != "ori"):
		sumValue = 0
		with open(os.path.join(path, filename), newline='') as csvfile:
			rows = csv.reader(csvfile)
			
			for i, row in enumerate(rows):
				sumValue += float(row[-1])
				sumList[x][y][i] +=  float(row[-1])
		Z[y][x] = sumValue/18
	if(sumValue/18<0.3):
		print(filename)
Z = np.array(Z)
a, b = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
print(a*100+500, b*100+200, np.max(Z))
print(np.array(sumList).shape)
sumList = np.array(sumList).reshape((-1, 18))

print(np.max(sumList, axis=0))
print(np.sum(np.max(sumList, axis=0))/18)

temp = np.argmax(sumList, axis=0)
print(temp//19*100+500, temp%19*100+200)

fig = plt.figure()
axis = fig.gca(projection='3d')

print(np.array(X).shape, np.array(Y).shape, np.array(Z).shape)
#畫圖
surface = axis.plot_surface(np.array(X), np.array(Y), np.array(Z), rstride=1, cstride=1, cmap='coolwarm_r')
fig.colorbar(surface, shrink=1.0, aspect=20)
 
# 設置圖表訊息
plt.title("F1-Score", fontsize=16)
plt.ylabel("n", fontsize=16)
plt.xlabel("m", fontsize=16)
# plt.zlabel("F1-Score", fontsize=16)
# plt.set_zlabel('F1-Score', fontsize=16)
 
plt.show()

# best 2300 1500
 