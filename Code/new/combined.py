from cmath import pi
import numpy as np
import cv2
import copy

# —————— hsv color
# black
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,100])
#blue
# lower_black = np.array([100,43,46])
# upper_black = np.array([124,255,255])
# white
lower_other = np.array([0,0,221])
upper_other = np.array([255,30,255])
# green
lower_green = np.array([50,30,30])
upper_green = np.array([70,255,255])
# green
h = 140
s = 100
v = 117

picname = "1023_3"
# picname = str(input('picname = '))
picnameL=picname+"-1"
picnameR=picname+"-2"

# 剪切出鼠标选定部分，记为pathMouse/imgCutGreen
points = []
def on_mouse(event,x,y,flags,param):
    global points, imgOrig,Cur_point,Start_point 
    copyImg = copy.deepcopy(imgOrig)
    h,w = imgOrig.shape[:2]
    mask_img = np.zeros([h+2,w+2],dtype=np.uint8)
    if  event == cv2.EVENT_LBUTTONDOWN:
        Start_point = [x,y]
        points.append(Start_point)
        cv2.circle(imgOrig,tuple(Start_point),1,(255,255,255),0)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        Cur_point = [x,y]
        cv2.line(imgOrig,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        points.append(Cur_point)
    elif event == cv2.EVENT_LBUTTONUP:
        Cur_point=Start_point
        cv2.line(imgOrig,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.circle(imgOrig,tuple(Cur_point),1,(255,255,255))
        cimg = np.zeros_like(imgOrig)
        cimg[:, :, :] = 255
        # cv2.fillConvexPoly(cimg,np.array(points),(0,0,0))
        cv2.fillPoly(cimg,[np.array(points)],(0,0,0))
        imgCutGreen = cv2.bitwise_or(copyImg, cimg)
        cv2.imwrite(pathMouse, imgCutGreen)
    cv2.imshow(winowName,imgOrig)
        
# ——————
# LEFT

# 读取原图
pathIn="PicIn/"+picnameL+".jpg"
imgColChan = cv2.imread(pathIn)

# 颜色范围
hsv_img = cv2.cvtColor(imgColChan, cv2.COLOR_BGR2HSV) # 转到HSV
mask_green = cv2.inRange(hsv_img, lower_black, upper_black)
# 中值滤波
mask_green = cv2.medianBlur(mask_green, 7)
# 轮廓提取
contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

pathOut="PicIgnore/outpColChan-"+picnameL+".jpg"
for i in range(6):
    colorNum=40+10*i # 大致筛选色块数
    print("Now select ",colorNum," ranges with least size.")
    contourD = sorted(contours,key= cv2.contourArea, reverse= False) [:colorNum]
    imgbackup=imgColChan
    cv2.drawContours(imgbackup,contourD,-1,(0,255,0),-1) # 绿色
    cv2.drawContours(imgbackup,contourD,-1,(0,255,0),3)
    cv2.imwrite(pathOut, imgbackup)
    cv2.namedWindow("previewLeft", 0)
    cv2.imshow('previewLeft',imgbackup)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    feedback=input("Is the result OK? (Y/N) ")
    if feedback=='Y': break

# correctMisAdjust.py

# 读取原图并切割
pathMouse="PicIgnore/midCutGreen-"+picnameL+".jpg"
path = "PicIgnore/outpColChan-"+picnameL+".jpg"
imgOrig = cv2.imread(path)
winowName="DrawLeft"
cv2.namedWindow(winowName, 0)
cv2.setMouseCallback(winowName,on_mouse)
cv2.waitKey(0)

#--------------------

#通过OpenCV读取图片信息
imgOrig = cv2.imread("PicIgnore/midCutGreen-"+picnameL+".jpg")

rows,cols,channels = imgOrig.shape

hsv = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_green, upper_green) # 低于、高于_red变0
#将制定像素点的数据设置为0, 要注意的是这三个参数对应的值是Blue, Green, Red。

# hsv: green
for r in range(rows):
    for c in range(cols):
        if mask[r, c] == 255: # 在区间内
            hsv.itemset((r, c, 0), hsv.item(r, c, 0) -h) 
            hsv.itemset((r, c, 1), hsv.item(r, c, 1) +90-s)
            hsv.itemset((r, c, 2), hsv.item(r, c, 2) +90-v)
imgCutRed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # 转回bgr

#---------------------------

imgFix=cv2.imread(path)
rows, cols, channels = imgCutRed.shape
for i in range(rows):
    for j in range(cols):
        if not all(imgCutRed[i,j]>210): # all true
            imgFix[i,j]=imgCutRed[i,j]; # 替换回原图

cv2.imwrite("PicOut/outpFix-"+picnameL+".jpg",imgFix)
cv2.namedWindow("fixedPictureLeft", 0)
cv2.imshow('fixedPictureLeft',imgFix)
cv2.waitKey(0)


# ——————
# RIGHT

# 读取原图
pathIn="PicIn/"+picnameR+".jpg"

imgColChan = cv2.imread(pathIn)

# 颜色范围
hsv_img = cv2.cvtColor(imgColChan, cv2.COLOR_BGR2HSV) # 转到HSV
mask_green = cv2.inRange(hsv_img, lower_black, upper_black)
# 中值滤波
mask_green = cv2.medianBlur(mask_green, 7)
# 轮廓提取
contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

pathOut="PicIgnore/outpColChan-"+picnameR+".jpg"
for i in range(6):
    colorNum=40+10*i # 大致筛选色块数
    print("Now select ",colorNum," ranges with least size.")
    contourD = sorted(contours,key= cv2.contourArea, reverse= False) [:colorNum]
    imgbackup=imgColChan
    cv2.drawContours(imgbackup,contourD,-1,(0,255,0),-1) # 绿色
    cv2.drawContours(imgbackup,contourD,-1,(0,255,0),3)
    cv2.imwrite(pathOut, imgbackup)
    cv2.namedWindow("previewRight", 0)
    cv2.imshow('previewRight',imgbackup)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    feedback=input("Is the result OK? (Y/N) ")
    if feedback=='Y': break

# correctMisAdjust.py

# 读取原图并切割
pathMouse="PicIgnore/midCutGreen-"+picnameR+".jpg"
path = "PicIgnore/outpColChan-"+picnameR+".jpg"
imgOrig = cv2.imread(path)
winowName="DrawRight"
cv2.namedWindow(winowName, 0)
cv2.setMouseCallback(winowName,on_mouse)
cv2.waitKey(0)

#--------------------

#通过OpenCV读取图片信息
imgOrig = cv2.imread("PicIgnore/midCutGreen-"+picnameR+".jpg")

rows,cols,channels = imgOrig.shape

hsv = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_green, upper_green) # 低于、高于_red变0
#将制定像素点的数据设置为0, 要注意的是这三个参数对应的值是Blue, Green, Red。

# hsv: green
for r in range(rows):
    for c in range(cols):
        if mask[r, c] == 255: # 在区间内
            hsv.itemset((r, c, 0), hsv.item(r, c, 0) -h) 
            hsv.itemset((r, c, 1), hsv.item(r, c, 1) +90-s)
            hsv.itemset((r, c, 2), hsv.item(r, c, 2) +90-v)
imgCutRed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # 转回bgr

#---------------------------

imgFix=cv2.imread(path)
rows, cols, channels = imgCutRed.shape
for i in range(rows):
    for j in range(cols):
        if not all(imgCutRed[i,j]>210): # all true
            imgFix[i,j]=imgCutRed[i,j]; # 替换回原图

cv2.imwrite("PicOut/outpFix-"+picnameR+".jpg",imgFix)
cv2.namedWindow("fixedPictureRight", 0)
cv2.imshow('fixedPictureRight',imgFix)
cv2.waitKey(0)