import numpy as np 
import json
import cv2
import shapely
import os
import argparse
import csv

with open("CrossTest.csv", 'w', newline='') as csvfileOut:
    detects = ['yolo-Result-best-rulebase', 'unet-Result-best-rulebase',  'marge']
    regres = ['RARETYPE1', 'RARERGB','RAREBINARY',  'dataColor.RARERGB', 
              'CRNNTYPE1','CRNNRGB','CRNNBINARY',   'dataColor.CRNNRGB', 
              'RosettaTYPE1','RosettaRGB','RosettaBINARY',  'dataColor.RosettaRGB',
              'STARNetTYPE1', 'STARNetRGB','STARNetBINARY',  'dataColor.STARNetRGB',
              'dataColor.NoneResNetBiLSTMCTCRGB',
              "voteRGB", "vote2RGB"]
    table = np.zeros((18, len(detects), len(regres)))
    for filename in os.listdir("./CrossTest/"):
        name, ext = os.path.splitext(filename)
        
        detect, regre, color = name.split("_")
        regre = regre+color
        print(filename, detect, regre)
        with open(os.path.join('./CrossTest/', filename), newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=':')
            for i, row in enumerate(rows):
                row = row[0].split(",")
                if detect not in detects:
                    # print(detect, "not in ")
                    continue
                if regre not in regres:
                    # print(regre, "not in ")
                    continue
                table[i, detects.index(detect), regres.index(regre)] = float(row[-1])
    print("table", table.shape)
    tableMean = np.mean(table, axis=0)
    # print(detects)
    # print(regres)
    # print(table)

    writer = csv.writer(csvfileOut)
    writer.writerow([""]+detects)
    for i in range(len(tableMean[0])):
        print(row)
        row = tableMean[:,i]
        writer.writerow([regres[i]]+row.tolist())

    with open("CrossTestbest.csv", 'w', newline='') as csvfileOut2:
        writer2 = csv.writer(csvfileOut2)
        tableMaxIdx = np.argmax(table.reshape((18, len(detects)*len(regres))), axis=1)
        tableMaxIdx = np.unravel_index(tableMaxIdx, (len(detects), len(regres)))
        print(tableMaxIdx)
        for i in range(len(tableMaxIdx[0])):
            print(i, detects[tableMaxIdx[0][i]].split("-")[0], regres[tableMaxIdx[1][i]], table[i,tableMaxIdx[0][i],tableMaxIdx[1][i]])
            writer2.writerow(["FPK_"+str(i+1).zfill(2)+".jpg", detects[tableMaxIdx[0][i]].split("-")[0], regres[tableMaxIdx[1][i]], table[i,tableMaxIdx[0][i],tableMaxIdx[1][i]]])