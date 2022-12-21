from cmath import pi
import numpy as np
import cv2
import copy

MAX_WL_RATIO = 3.5
BLUR_SIZE = 3
EXTEND_SIZE = 5

# —————— hsv color
# black
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,46])
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

picname = "edit1027_2_1"
picnameL=picname+"-1"
picnameR=picname+"-2"

# ——————

# 读取原图
pathIn="PicIn/"+picnameL+".jpg"

imgColChan = cv2.imread(pathIn)

# 颜色范围
hsv_img = cv2.cvtColor(imgColChan, cv2.COLOR_BGR2HSV) # 转到HSV
# hsv_img = cv2.medianBlur(hsv_img, BLUR_SIZE)
# hsv_img = cv2.GaussianBlur(hsv_img, (BLUR_SIZE, BLUR_SIZE), 0)
mask_green = cv2.inRange(hsv_img, lower_black, upper_black)
mask_other = cv2.inRange(hsv_img, lower_other, upper_other)
# 高斯滤波
mask_green = cv2.GaussianBlur(mask_green, (EXTEND_SIZE, EXTEND_SIZE), 0)
mask_other = cv2.GaussianBlur(mask_other, (EXTEND_SIZE, EXTEND_SIZE), 0)
# 轮廓提取
contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contourAll, hierarchyAll = cv2.findContours(mask_other, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

colorNum=90 # 大致色块数
contourRed = sorted(contourAll, key = cv2.contourArea, reverse = False)# [:colorNum]
contours = sorted(contours, key = cv2.contourArea, reverse= False) #[:colorNum]

contours_ = [];
for contour in contours:
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    # print(area)
    if (area > 0 and area < 2000 and rect[1][0] / rect[1][1] < MAX_WL_RATIO and rect[1][1] / rect[1][0] < MAX_WL_RATIO):
        contours_.append(contour)

cv2.drawContours(imgColChan,contourRed,-1,(0,0,255),-1) # 红色
cv2.drawContours(imgColChan,contours_,-1,(0,255,0),-1) # 绿色
# cv2.drawContours(imgColChan,contours,-1,(0,0,0),-1) # black

pathOut="PicOut/outpColChan-"+picnameL+".jpg"

cv2.imwrite(pathOut, imgColChan)

# ——————
# correctMisAdjust.py

# 剪切出鼠标选定部分， 记为finalImg.jpg
points = []
imgCutGreen = copy.deepcopy(imgColChan)
cimg = np.zeros_like(imgColChan)
cimg[:, :, :] = 255
def on_mouse(event,x,y,flags,param):
    global points, imgColChan,Cur_point,Start_point, imgCutGreen, cimg
    # h,w = imgColChan.shape[:2]
    # mask_img = np.zeros([h+2,w+2],dtype=np.uint8)
    if  event == cv2.EVENT_LBUTTONDOWN:
        points = []
        Start_point = [x,y]
        points.append(Start_point)
        cv2.circle(imgColChan,tuple(Start_point),1,(255,255,255),0)
        cv2.imshow("Window", imgColChan)
        cv2.setWindowTitle("Window", "Drawing")
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        Cur_point = [x,y]
        cv2.line(imgColChan,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.imshow("Window", imgColChan)
        points.append(Cur_point)
    elif event == cv2.EVENT_LBUTTONUP:
        Cur_point=Start_point
        cv2.line(imgColChan,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.circle(imgColChan,tuple(Cur_point),1,(255,255,255))
        cv2.fillConvexPoly(cimg,np.array(points),(0,0,0))
        # cv2.imwrite("test.jpg", cimg)

# 读取原图并切割
pathMouse="PicOut/midCutGreen-"+picnameL+".jpg"
path = "PicOut/outpColChan-"+picnameL+".jpg"
cv2.namedWindow("Window", 0)
cv2.moveWindow("Window", 0, 0)
cv2.setWindowTitle("Window", "DrawLeft")
cv2.setMouseCallback("Window",on_mouse,0)
cv2.imshow("Window",imgColChan)
cv2.waitKey(0)
cv2.destroyWindow("Window")

imgCutGreen = cv2.bitwise_or(imgCutGreen, cimg)
if (len(points) > 0):
    cv2.imwrite(pathMouse, imgCutGreen)

#--------------------

#通过OpenCV读取图片信息
imgColChan = cv2.imread("PicOut/midCutGreen-"+picnameL+".jpg")

rows,cols,channels = imgColChan.shape

hsv = cv2.cvtColor(imgColChan, cv2.COLOR_BGR2HSV)
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
cv2.namedWindow('fixedPictureLeft', 0)
cv2.imshow('fixedPictureLeft',imgFix)
cv2.moveWindow("fixedPictureLeft", 100, 0)
cv2.waitKey(0)
cv2.destroyWindow("fixedPictureLeft")


# 读取原图
print()
pathIn="PicIn/"+picnameR+".jpg"

imgColChan = cv2.imread(pathIn)

# 颜色范围
hsv_img = cv2.cvtColor(imgColChan, cv2.COLOR_BGR2HSV) # 转到HSV
mask_green = cv2.inRange(hsv_img, lower_black, upper_black)
mask_other = cv2.inRange(hsv_img,lower_other,upper_other)
# 高斯滤波
mask_green = cv2.GaussianBlur(mask_green, (EXTEND_SIZE, EXTEND_SIZE), 0)
mask_other = cv2.GaussianBlur(mask_other, (EXTEND_SIZE, EXTEND_SIZE), 0)
# 轮廓提取
contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contourAll, hierarchyAll = cv2.findContours(mask_other, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

colorNum=90 # 大致色块数
contourRed = sorted(contourAll, key = cv2.contourArea, reverse = False)# [:colorNum]
contours = sorted(contours,key= cv2.contourArea, reverse= True) #[:colorNum]

contours_ = [];
for contour in contours:
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    # print(area)
    if (area > 0 and area < 2000 and rect[1][0] / rect[1][1] < MAX_WL_RATIO and rect[1][1] / rect[1][0] < MAX_WL_RATIO):
        contours_.append(contour)

cv2.drawContours(imgColChan,contourRed,-1,(0,0,255),-1) # 红色
cv2.drawContours(imgColChan,contours_,-1,(0,255,0),-1) # 绿色
# cv2.drawContours(imgColChan,contours,-1,(0,0,0),-1) # black

pathOut="PicOut/outpColChan-"+picnameR+".jpg"
cv2.imwrite(pathOut, imgColChan)

# ——————
# correctMisAdjust.py

# imgCutGreen = copy.deepcopy(imgColChan)
# cimg = np.zeros_like(imgColChan)
# cimg[:, :, :] = 255

# # 读取原图并切割
# pathMouse="PicOut/midCutGreen-"+picnameR+".jpg"
# path = "PicOut/outpColChan-"+picnameR+".jpg"
# cv2.namedWindow("Window")
# cv2.moveWindow("Window", 0, 0)
# cv2.setWindowTitle("Window", "DrawRight")
# cv2.setMouseCallback("Window",on_mouse,0)
# cv2.imshow("Window",imgColChan)
# cv2.waitKey(0)
# cv2.destroyWindow("Window")

# imgCutGreen = cv2.bitwise_or(imgCutGreen, cimg)
# cv2.imwrite(pathMouse, imgCutGreen)

# #--------------------

# #通过OpenCV读取图片信息
# imgColChan = cv2.imread("PicOut/midCutGreen-"+picnameR+".jpg")

# rows,cols,channels = imgColChan.shape

# hsv = cv2.cvtColor(imgColChan, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, lower_green, upper_green) # 低于、高于_red变0
# #将制定像素点的数据设置为0, 要注意的是这三个参数对应的值是Blue, Green, Red。

# # hsv: green
# for r in range(rows):
#     for c in range(cols):
#         if mask[r, c] == 255: # 在区间内
#             hsv.itemset((r, c, 0), hsv.item(r, c, 0) -h) 
#             hsv.itemset((r, c, 1), hsv.item(r, c, 1) +90-s)
#             hsv.itemset((r, c, 2), hsv.item(r, c, 2) +90-v)
# imgCutRed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # 转回bgr

# #---------------------------

# imgFix=cv2.imread(path)
# rows, cols, channels = imgCutRed.shape
# for i in range(rows):
#     for j in range(cols):
#         if not all(imgCutRed[i,j]>210): # all true
#             imgFix[i,j]=imgCutRed[i,j]; # 替换回原图

# cv2.imwrite("PicOut/outpFix-"+picnameR+".jpg",imgFix)
cv2.imwrite("PicOut/outpFix-"+picnameR+".jpg", imgColChan)
cv2.namedWindow('fixedPictureRight', 0)
cv2.imshow('fixedPictureRight',imgColChan)
cv2.moveWindow("fixedPictureRight", 0, 0)
cv2.waitKey(0)
cv2.destroyAllWindows()