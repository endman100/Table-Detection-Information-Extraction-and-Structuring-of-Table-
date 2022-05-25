import os

fp = open("train.txt", "w")

for i in range(1, 100000):
	fp.write("../../data/yoloGtold1/train/"+str(i)+".png"+"\n")
fp.close()
fp = open("test.txt", "w")

for i in range(100000, 101000+1):
	fp.write("../../data/yoloGtold1/test/"+str(i)+".png"+"\n")
fp.close()


'''
darknet detector train ../my/obj.data ../my/yolov3.cfg ../my/weights/temp/yolov3_last.weights

darknet detect ../my/yolov3.cfg ../my/weights/yolov3_final.weights ../my/question/FPK_01.jpg
'''