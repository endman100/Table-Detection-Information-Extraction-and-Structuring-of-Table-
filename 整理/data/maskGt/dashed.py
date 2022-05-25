import cv2
import numpy as np
import math
def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)

def drawcircle(img, center, radius, color, thickness=1, style='line'):
    poly = []
    if style == 'line':
        for angle in range(0, 360, 15):
            for i in range(angle*1000, (angle+7)*1000):
                h = int(center[1] + math.sin(math.pi/180*i/1000)*radius)
                w = int(center[0] + math.cos(math.pi/180*i/1000)*radius)
                if(len(poly) == 0):
                    poly.append([[w, h]])
                    continue
                if(poly[-1][0][0] == w and poly[-1][0][1] == h):
                    continue
                poly.append([[w, h]])
        cv2.drawContours(img, np.array(poly), -1, color, thickness)
if __name__ == '__main__':
    img = np.zeros((1000, 1000, 3), np.uint8)
    img[:,:,:] = 255
    drawcircle(img, (500, 400), 100, None)

    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
