import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

path = "./yolo-Result/"
datas = [[] for i in range(18)]
yLabel = []
for filename in os.listdir(path):
	print(filename)
	size = os.path.splitext(filename)[0]
	if(size != "sidewindow" and size != "ori"):
		yLabel.append(int(size))
		with open(os.path.join(path, filename), newline='') as csvfile:
			rows = csv.reader(csvfile)
			for i, row in enumerate(rows):
				datas[i].append(float(row[-1]))

# sidewindow = []
# with open("./yolo-Result/sidewindow.csv", newline='') as csvfile:
# 	rows = csv.reader(csvfile)
# 	for i, row in enumerate(rows):
# 		sidewindow.append(float(row[-1]))
# print("np.mean(sidewindow)", np.mean(sidewindow))
# sidewindow = [np.mean(sidewindow)]*44

# ori = []
# with open("./yolo-Result/ori.csv", newline='') as csvfile:
# 	rows = csv.reader(csvfile)
# 	for i, row in enumerate(rows):
# 		ori.append(float(row[-1]))
# print("np.mean(ori)", np.mean(ori))
# ori = [np.mean(ori)]*44
# print(datas)
# plt.title('')
# for i in range(len(datas)):
# 	r, g, b = i/9/3, i%9/9, i%3/3
# 	# plt.plot(yLabel, datas[i], color=(r, g, b), label=str(i))
# print(np.max(datas[12]))
# print(np.max(datas, 1), "max")
# print(np.argmax(datas, 1)*100+608, "maxarg")

# print(np.max(np.mean(datas, 0)))
# plt.plot(yLabel, ori, color=(1, 0, 0), label="第一種模式",linewidth=5.0)
# plt.plot(yLabel, sidewindow, color=(0, 0, 1), label="第二種模式",linewidth=5.0)

# plt.plot(yLabel, np.mean(datas, 0), color=(0, 1, 0), label="第三種模式",linewidth=5.0)
# print(np.mean(datas, 0))
# print(yLabel)
# plt.legend() # 显示图例
# plt.xlabel('m')
# plt.ylabel('F1-Score')
# plt.show()

sidewindow = []
with open("./yolo-Result/sidewindow.csv", newline='') as csvfile:
	rows = csv.reader(csvfile)
	for i, row in enumerate(rows):
		sidewindow.append(float(row[-1]))
print("np.mean(sidewindow)", np.mean(sidewindow))
sidewindow = [sidewindow[12]]*44

ori = []
with open("./yolo-Result/ori.csv", newline='') as csvfile:
	rows = csv.reader(csvfile)
	for i, row in enumerate(rows):
		ori.append(float(row[-1]))
print("np.mean(ori)", np.mean(ori))
ori = [ori[12]]*44

plt.plot(yLabel, ori, color=(1, 0, 0), label="第一種模式",linewidth=5.0)
plt.plot(yLabel, sidewindow, color=(0, 0, 1), label="第二種模式",linewidth=5.0)

plt.plot(yLabel, datas[12], color=(0, 1, 0), label="第三種模式",linewidth=5.0)

plt.legend() # 显示图例
plt.xlabel('m')
plt.ylabel('F1-Score')
plt.show()