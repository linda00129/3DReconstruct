import cv2
import numpy as np
import matplotlib.pyplot as plt
import combined

#1.跨越架的具体参数
#----------------------------------------------------------------------------------
length = int(input("please input length: "))
width  = int(input("please input width: "))
height = int(input("please input height: "))
f_l = int(input("please input number of fastenings (length): "))
f_w = int(input("please input number of fastenings (width):  "))
f_h = int(input("please input number of fastenings (height): "))
'''def get_foot(start_point, end_point, point_a):
    start_x, start_y = start_point
    end_x, end_y = end_point
    pa_x, pa_y = point_a

    p_foot = np.array([0, 0], dtype=np.float)
    if start_point[0] == end_point[0]:
        p_foot[0] = start_point[0]
        p_foot[1] = point_a[1]
        return p_foot

    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot[0] = (b * b * pa_x - a * b * pa_y - a * c) / (a * a + b * b)
    p_foot[1] = (a * a * pa_y - a * b * pa_x - b * c) / (a * a + b * b)

    return p_foot


F = np.load("Fundamental.npy")'''
#2.二维像素点
#----------------------------------------------------------------------------------
# pts_right = np.array([[233,91],[682,145],[1004,192],[196,153],[777,205],[1150,251],
#  [292,478],[641,462],[911,464],[270,643],[698,603],[1017,577],
#  [326,773],[608,721],[846,684],[331,926],[653,845],[917,794]
# ], dtype=float)

# pts_left = np.array([[163,108],[620,159],[951,204],[108,171],[702,219],[1090,263],
# [234,493],[586,476],[862,476],[202,654],[638,614],[965,590],
# [281,785],[561,734],[803,698],[278,939],[601,856],[870,806]
# ], dtype=float)

pts_right = []
pts_left  = []

#2.1获取左相机的坐标点
# img1=cv2.imread('NewPicture/0807_1.jpg')
img1="PicIn/"+combined.picnameL+".jpg"
 
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        pts_right.append([x,y])
        cv2.circle(img1, (x, y), 2, (0, 0, 255))
        cv2.putText(img1, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255))
        cv2.imshow("image", img1)
        
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow("image", img1)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
pts_right = np.array(pts_right,dtype=float)
cv2.destroyAllWindows()


# 2.2获取右相机的坐标点
# img2=cv2.imread('NewPicture/0807_2.jpg')
img2="PicIn/"+combined.picnameL+".jpg"
 
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        pts_left.append([x,y])
        cv2.circle(img2, (x, y), 2, (0, 0, 255))
        cv2.putText(img2, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255))
        cv2.imshow("image", img2)
        
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow("image", img2)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
pts_left = np.array(pts_left,dtype=float)
cv2.destroyAllWindows()

print("pts_left")
print(pts_left)
print("pts_right")
print(pts_right)

# ————————————————
# 版权声明：本文为CSDN博主「安岳第二帅」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/People1007/article/details/122420735
#----------------------------------------------------------------------------------




#3.相机矩阵
# ----------------------------------------------------------------------------------
proj_mats_left = np.array([[853.600131816948,	-0.234413875545894,	711.824824003976,	0], 
                           [0,	853.600457775920,	521.205616198394,	0], 
                           [0,	0,	1,	0]], dtype=float)
# ([[842.3037, 1.1647, 720.7515, 0], 
#                            [0, 842.6949, 518.2422, 0], 
#                            [0, 0, 1, 0]], dtype=float)
proj_mats_right = np.array([[862.556681899433,	2.82011633929761,	700.939253794893,	-49512.1536983289],
                            [7.20622905959800,	856.304092118309,	516.701468893003,	-339.498256256133],
                            [0.0126794427383513,	0.00520997843976957,	0.999906039513965,	0.481779459104346]], dtype=float)
# ([[8.54836113e+02, -5.07523434e+00,  6.80629342e+02, -4.61477934e+04],
#                             [5.62137355e+00,  8.45859320e+02,  5.27496332e+02, 4.10971997e+02],
#                             [8.30000000e-03, -4.10000000e-03,  1.00000000e+00, 2.49430000e+00]], dtype=float)
world_points = cv2.triangulatePoints(proj_mats_left, proj_mats_right, pts_left.T, pts_right.T).T
#----------------------------------------------------------------------------------


#4.点和点之间的配对
#----------------------------------------------------------------------------------
# pairs = [[0, 1], [0, 3], [0, 6], [1, 2], [1, 4], [1, 7], [2, 5], [2, 8], [3, 4], [3, 9], [4, 5], [4, 10], [5, 11],
#          [6, 7], [6, 9], [6, 12], [7, 8], [7, 10], [7, 13], [8, 11], [8, 14], [9, 10], [9, 15], [10, 11], [10, 16],
#          [11, 17], [12, 13], [12, 15], [13, 14], [13, 16], [14, 17], [15, 16], [16, 17]]
pairs_1 =[]
pairs_2 =[]
pairs_3 =[]

for i in range(f_l*f_h*f_w):
    if (i+1)%f_l != 0:
        j = [i, i+1]
        pairs_1.append(j)
pairs_1 = np.array(pairs_1)

for i in range(f_l*f_h*f_w):
    if (i)%(f_l*f_w) < f_l:
        j = [i, i+f_l]
        pairs_2.append(j)
pairs_2 = np.array(pairs_2)

for i in range(f_l*f_h*f_w):
    if  i < (f_l*f_h*f_w-f_l*f_w):
        j = [i, i+f_l*f_w]
        pairs_3.append(j)
pairs_3 = np.array(pairs_3)
# print(pairs_1)
# print(pairs_2)
# print(pairs_3)
#----------------------------------------------------------------------------------

#5.跨越架扣件的世界坐标
#----------------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for pt in world_points:
    ax.scatter(pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3], s=5, c=None, depthshade=True)
    # print(pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3])
#----------------------------------------------------------------------------------


#6.用红线连接扣件+算出横向，宽，纵向的杆子的长度（单位:mm）
#----------------------------------------------------------------------------------
dist_1 = []
dist_2 = []
dist_3 = []

for pair in pairs_1:
    pt1, pt2 = world_points[pair[0]], world_points[pair[1]]
    x = [pt1[0] / pt1[3], pt2[0] / pt2[3]]
    y = [pt1[1] / pt1[3], pt2[1] / pt2[3]]
    z = [pt1[2] / pt1[3], pt2[2] / pt2[3]]
    ax.plot(x, y, z, c='r')
#     if pair == [0, 6] or pair == [1, 7] or pair == [2, 8] or pair == [3, 9] or pair == [4, 10] or pair == [5, 11] or pair == [
#         6, 12] or pair == [7, 13] or pair == [8, 14] or pair == [9, 15] or pair == [10, 16] or pair == [11, 17]:
    dist_1.append(np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2 + (z[0] - z[1]) ** 2))

for pair in pairs_2:
    pt1, pt2 = world_points[pair[0]], world_points[pair[1]]
    x = [pt1[0] / pt1[3], pt2[0] / pt2[3]]
    y = [pt1[1] / pt1[3], pt2[1] / pt2[3]]
    z = [pt1[2] / pt1[3], pt2[2] / pt2[3]]
    ax.plot(x, y, z, c='r')
#     if pair == [0, 6] or pair == [1, 7] or pair == [2, 8] or pair == [3, 9] or pair == [4, 10] or pair == [5, 11] or pair == [
#         6, 12] or pair == [7, 13] or pair == [8, 14] or pair == [9, 15] or pair == [10, 16] or pair == [11, 17]:
    dist_2.append(np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2 + (z[0] - z[1]) ** 2))

for pair in pairs_3:
    pt1, pt2 = world_points[pair[0]], world_points[pair[1]]
    x = [pt1[0] / pt1[3], pt2[0] / pt2[3]]
    y = [pt1[1] / pt1[3], pt2[1] / pt2[3]]
    z = [pt1[2] / pt1[3], pt2[2] / pt2[3]]
    ax.plot(x, y, z, c='r')
#     if pair == [0, 6] or pair == [1, 7] or pair == [2, 8] or pair == [3, 9] or pair == [4, 10] or pair == [5, 11] or pair == [
#         6, 12] or pair == [7, 13] or pair == [8, 14] or pair == [9, 15] or pair == [10, 16] or pair == [11, 17]:
    dist_3.append(np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2 + (z[0] - z[1]) ** 2))

plt.show()
dist_1 = np.array(dist_1, dtype=float)
dist_2 = np.array(dist_2, dtype=float)
dist_3 = np.array(dist_3, dtype=float)
#----------------------------------------------------------------------------------



#7.按照比例尺放大
#----------------------------------------------------------------------------------
dist_1 = dist_1*(length/np.mean(dist_1))/10
dist_2 = dist_2*(width/np.mean(dist_2))/10
dist_3 = dist_3*(height/np.mean(dist_3))/10
#----------------------------------------------------------------------------------




print("detected length")
print(dist_1)
print("detected width")
print(dist_2)
print("detected height")
print(dist_3)
# # print(np.mean(dist))
# # print(np.std(dist, ddof=1))
