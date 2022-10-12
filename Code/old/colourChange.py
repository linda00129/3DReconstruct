import numpy as np
import cv2

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
pathIn="new6_1.jpg"
img = cv2.imread(pathIn)
# 根据颜色范围删选
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 转到HSV
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

'''
for i in contours:
    ((x,y),radius) = cv2.minEnclosingCircle(i)
    print("中心坐标",[x,y])
'''

cv2.drawContours(img,contours,-1,(0,255,0),-1) # 绿色


'''
for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    'cv2.putText(img, "Black", (x, y - 5), font, 0.7, (0, 255, 0), 2)'
'''

pathOut="Right4_810.jpg" # 储存路径
cv2.namedWindow("dection",cv2.WINDOW_NORMAL)
cv2.imshow("dection",img)

cv2.imwrite(pathOut, img)


cv2.waitKey(0)
cv2.destroyAllWindows()














