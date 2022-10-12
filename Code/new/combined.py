import numpy as np
import cv2
import copy

# 红色色块未处理

#font = cv2.FONT_HERSHEY_SIMPLEX
lower_black = np.array([0,0,0])  # 黑色范围低阈值
upper_black = np.array([180,255,46])  # 黑色范围高阈值
#黑色 lower[0,0,0]   upper[180,255,46]
#黄色 lower[26,43,46]   upper[34,255,255]
lower_red = np.array([0, 127, 128])  # 红色范围低阈值
upper_red = np.array([10, 255, 255])  # 红色范围高阈值
# 需要更多颜色，可以去百度一下HSV阈值！

# 读取原图
pathIn="C:/0t_my/0t_study/2022_2023Fall/490/my/0918/inpTest.jpg"
imgColChan = cv2.imread(pathIn)
# 根据颜色范围删选
hsv_img = cv2.cvtColor(imgColChan, cv2.COLOR_BGR2HSV) # 转到HSV
mask_green = cv2.inRange(hsv_img, lower_black, upper_black)
mask_red = cv2.inRange(hsv_img,lower_red,upper_red)
# 中值滤波
mask_green = cv2.medianBlur(mask_green, 7)
mask_red = cv2.medianBlur(mask_red,7)
mask = cv2.bitwise_or(mask_green, mask_red)
# 轮廓提取
contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

colorNum=85 # 大致色块数
contours = sorted(contours, key = cv2.contourArea, reverse = False)     #  从大减少
contours = sorted(contours,key= cv2.contourArea, reverse= True)[:colorNum]
# print 面积大小
c = 0
for i in contours:
    area = cv2.contourArea(i)
    c = c + 1
    print(c)
    print(area)


cv2.drawContours(imgColChan,contours,-1,(0,255,0),-1) # 绿色


pathOut="C:/0t_my/0t_study/2022_2023Fall/490/my/0918/outpColChanTest.jpg" # 储存路径
# cv2.namedWindow("dection",cv2.WINDOW_NORMAL)
# cv2.imshow("dection",imgColChan)

cv2.imwrite(pathOut, imgColChan)


cv2.waitKey(0)
cv2.destroyAllWindows()




# correctMisAdjust.py




# 剪切出鼠标选定部分， 记为finalImg.jpg
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
        cv2.imshow("",imgOrig)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        Cur_point = [x,y]
        cv2.line(imgOrig,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.imshow("",imgOrig)
        points.append(Cur_point)
    elif event == cv2.EVENT_LBUTTONUP:
        Cur_point=Start_point
        cv2.line(imgOrig,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.circle(imgOrig,tuple(Cur_point),1,(255,255,255))
        cimg = np.zeros_like(imgOrig)
        cimg[:, :, :] = 255
        cv2.fillConvexPoly(cimg,np.array(points),(0,0,0))
        imgCutGreen = cv2.bitwise_or(copyImg, cimg)
        # cv2.imshow('finalImg', imgCutGreen) 
        cv2.imwrite("C:/0t_my/0t_study/2022_2023Fall/490/my/0918/midCutGreen.jpg", imgCutGreen)

# 读取原图并切割
path = 'C:/0t_my/0t_study/2022_2023Fall/490/my/0918/outpColChanTest.jpg'
imgOrig = cv2.imread(path)
cv2.namedWindow("Test")
cv2.setMouseCallback("Test",on_mouse,0)
cv2.imshow("Test",imgOrig)
cv2.waitKey(0)

#--------------------


def nothing(x):
    pass
#通过OpenCV读取图片信息
imgOrig = cv2.imread('C:/0t_my/0t_study/2022_2023Fall/490/my/0918/midCutGreen.jpg')
# cv2.imshow("imgOrig", imgOrig)

lower_red = np.array([50,30,30])
upper_red = np.array([70,255,255]) #选取绿色的HSV（120degree/2,30-255,30-255)

# cv2.namedWindow('imgCutRed')


rows,cols,channels = imgOrig.shape

# 绿色
h = 140
s = 100
v = 117
print("current h s v =", h, s, v)


hsv = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_red, upper_red) # 低于、高于_red变0
#将制定像素点的数据设置为0, 要注意的是这三个参数对应的值是Blue, Green, Red。

for r in range(rows):
    for c in range(cols):
        if mask[r, c] == 255: # 在区间内
            hsv.itemset((r, c, 0), hsv.item(r, c, 0) -h)
            hsv.itemset((r, c, 1), hsv.item(r, c, 1) +90-s)
            hsv.itemset((r, c, 2), hsv.item(r, c, 2) +90-v)
imgCutRed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # 转回bgr

# cv2.imwrite('C:/0t_my/0t_study/2022_2023Fall/490/my/0918/midCutRed.jpg',imgCutRed)
# cv2.imshow("imgCutRed", imgCutRed)

#---------------------------

imgFix=cv2.imread(path);
rows, cols, channels = imgCutRed.shape;
for i in range(rows):
    for j in range(cols):
        if not all(imgCutRed[i,j]>210): # all true
            imgFix[i,j]=imgCutRed[i,j]; # 替换回原图

cv2.imwrite('C:/0t_my/0t_study/2022_2023Fall/490/my/0918/outpFixTest.jpg',imgFix);
cv2.imshow('fixedPicture',imgFix);
cv2.waitKey(0)





