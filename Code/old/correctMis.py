import cv2
import os
import numpy as np
import copy

# 剪切出鼠标选定部分， 记为finalImg.jpg
points = []
def on_mouse(event,x,y,flags,param):
    global points, img,Cur_point,Start_point 
    copyImg = copy.deepcopy(img)
    h,w = img.shape[:2]
    mask_img = np.zeros([h+2,w+2],dtype=np.uint8)
    if  event == cv2.EVENT_LBUTTONDOWN:
        Start_point = [x,y]
        points.append(Start_point)
        cv2.circle(img,tuple(Start_point),1,(255,255,255),0)
        cv2.imshow("",img)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        Cur_point = [x,y]
        cv2.line(img,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.imshow("",img)
        points.append(Cur_point)
    elif event == cv2.EVENT_LBUTTONUP:
        Cur_point=Start_point
        cv2.line(img,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.circle(img,tuple(Cur_point),1,(255,255,255))
        cimg = np.zeros_like(img)
        cimg[:, :, :] = 255
        cv2.fillConvexPoly(cimg,np.array(points),(0,0,0))
        final = cv2.bitwise_or(copyImg, cimg)
        cv2.imshow('finalImg', final) 
        cv2.imwrite("finalImg.jpg", final)

# 读取原图并切割
path = 'Right4_810.jpg'
img = cv2.imread(path)
cv2.namedWindow("Test")
cv2.setMouseCallback("Test",on_mouse,0)
cv2.imshow("Test",img)
cv2.waitKey(0)

#--------------------


def nothing(x):
    pass
#通过OpenCV读取图片信息
img = cv2.imread('finalImg.jpg')
cv2.imshow("img", img)

lower_red = np.array([50,30,30])
upper_red = np.array([70,255,255]) #选取绿色的HSV（120degree/2,30-255,30-255)

cv2.namedWindow('img2')


rows,cols,channels = img.shape

'''
cv2.createTrackbar('H','img2',140,180,nothing)
cv2.createTrackbar('S','img2',100,180,nothing)
cv2.createTrackbar('V','img2',117,180,nothing)


for i in range(2):
    print(i)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red) # 低于、高于_red变0
    #将制定像素点的数据设置为0, 要注意的是这三个参数对应的值是Blue, Green, Red。
    h = cv2.getTrackbarPos('H', 'img2')
    s = cv2.getTrackbarPos('S', 'img2')
    v = cv2.getTrackbarPos('V', 'img2') # 绿色
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 255: # 在区间内
                hsv.itemset((r, c, 0), hsv.item(r, c, 0) -h)
                hsv.itemset((r, c, 1), hsv.item(r, c, 1) +90-s)
                hsv.itemset((r, c, 2), hsv.item(r, c, 2) +90-v)
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # 转回bgr
'''
# 绿色
h = 140
s = 100
v = 117
print("current h s v =", h, s, v)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_red, upper_red) # 低于、高于_red变0
#将制定像素点的数据设置为0, 要注意的是这三个参数对应的值是Blue, Green, Red。

for r in range(rows):
    for c in range(cols):
        if mask[r, c] == 255: # 在区间内
            hsv.itemset((r, c, 0), hsv.item(r, c, 0) -h)
            hsv.itemset((r, c, 1), hsv.item(r, c, 1) +90-s)
            hsv.itemset((r, c, 2), hsv.item(r, c, 2) +90-v)
img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # 转回bgr

cv2.imwrite('Replace.jpg',img2)
cv2.imshow("img2", img2)

#---------------------------

# fiximg=cv2.imread(path);
# rows, cols, channels = img2.shape;
# for i in range(rows):
#     for j in range(cols):
#         if not all(img2[i,j]>210): # all true
#             fiximg[i,j]=img2[i,j]; # 替换回原图

# cv2.imwrite('fixedPicture.jpg',fiximg);
# cv2.imshow('fixedPicture',fiximg);
# cv2.waitKey(0)

fiximg=cv2.imread(path);
rows, cols, channels = img2.shape;
for i in range(rows):
    for j in range(cols):
        if not all(img2[i,j]>210): # all true
            img[i,j]=img2[i,j]; # 替换回原图

cv2.imwrite('fixedPicture.jpg',fiximg);
cv2.imshow('fixedPicture',fiximg);
cv2.waitKey(0)
