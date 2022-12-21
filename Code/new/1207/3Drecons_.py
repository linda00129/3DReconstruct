import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pylab import *
import sift as sift
from identify import *

def get_foot(start_point, end_point, point_a):
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

pts_left = np.array(sift.pts_left, dtype=float)

pts_right = np.array(sift.pts_right, dtype=float)

# im = cv2.imread("matches.png")
# w = im.shape[1] / 2

# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     global pts_left, pts_right
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if (x > w):
#             pts_right = np.append(pts_right, np.array([[x - w, y]], dtype=float), axis=0)
#             xy = "%d,%d" % (x - w, y)
#         else:
#             pts_left = np.append(pts_left, np.array([[x, y]], dtype=float), axis=0)
#             xy = "%d,%d" % (x, y)
#         cv2.circle(im, (x, y), 2, (0, 0, 255))
#         cv2.putText(im, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255))
#         cv2.imshow("image", im)

# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# while(1):
#     cv2.imshow("image", im)
#     key = cv2.waitKey(5) & 0xFF
#     if key == ord('q'):
#         break
# cv2.destroyAllWindows()

#2.1获取左相机的坐标点
img1=cv2.imread(sift.queryImagePath)
 
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global pts_left
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        pts_left = np.append(pts_left, [[x,y]], axis=0)
        cv2.circle(img1, (x, y), 2, (0, 0, 255))
        cv2.putText(img1, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255))
        cv2.imshow("image", img1)
        
cv2.namedWindow("image")
cv2.moveWindow("image", 0, 0)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow("image", img1)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()


# 2.2获取右相机的坐标点
img2=cv2.imread(sift.trainingImagePath)
 
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global pts_right
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        pts_right = np.append(pts_right, [[x,y]], axis=0)
        cv2.circle(img2, (x, y), 2, (0, 0, 255))
        cv2.putText(img2, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255))
        cv2.imshow("image", img2)
        
cv2.namedWindow("image")
cv2.moveWindow("image", 0, 0)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow("image", img2)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

'''for i in range(len(pts_left)):
    l_r = F.T @ np.append(pts_left[i], 1)
    start_pt = np.array([0, (-l_r[2] / l_r[1])], dtype=np.float)
    end_pt = np.array([1520, (-1520 * l_r[0] / l_r[1] - l_r[2] / l_r[1])], dtype=np.float)
    pts_right[i] = get_foot(start_pt, end_pt, pts_right[i])
'''
# Matrices for Picture/blue3,4
# proj_mats_left = np.array([[1455.5, 0.9612, 856.7579, 0], [0, 1454.7, 893.6307, 0], [0, 0, 1, 0]], dtype=float)
# proj_mats_right = np.array([[1.45035032e+03, -1.45382906e+01, 7.64411949e+02,
#                              7.46021517e+04],
#                             [1.68462464e+01, 1.44986615e+03, 8.86185755e+02,
#                              -1.46994658e+02],
#                             [3.62950000e-04, 5.76190000e-04, 1.00000000e+00,
#                              2.11600000e-01]], dtype=float)

# Matrices for NewPicture/0807,0901
# proj_mats_left = np.array([[842.3037, 1.1647, 720.7515, 0], [0, 842.6949, 518.2422, 0], [0, 0, 1, 0]])
# proj_mats_right = np.array([
#     [8.54836113e+02, -5.07523434e+00, 6.80629342e+02, -4.61477934e+04], 
#     [5.62137355e+00,  8.45859320e+02, 5.27496332e+02,  4.10971997e+02],
#     [8.30000000e-03, -4.10000000e-03, 1.00000000e+00,  2.49430000e+00]])

# 1023
proj_mats_left = np.array([
[855.883631, 0.677484, 711.240074, 0.000000],
[0.000000, 855.455451, 526.926009, 0.000000],
[0.000000, 0.000000, 1.000000, 0.000000]], dtype=float)

proj_mats_right = np.array([
[848.796471, 1.507855, 719.681853, 51341.905931],
[-5.975813, 855.508490, 526.805999, -1071.760056],
[-0.009904, 0.000101, 0.999951, -1.741791]], dtype=float)

world_points = cv2.triangulatePoints(proj_mats_left, proj_mats_right, pts_left.T, pts_right.T).T

WorPoints=[];
for i in range(len(world_points)):
    x,y,z,w=world_points[i];
    WorPoints.append([x/w,y/w,z/w]);
# print(WorPoints)

WorPoints=array(WorPoints);
N = int(input('please input the total number of fastenings: '))
cluster=KMeans(n_clusters=N);
cluster=cluster.fit(WorPoints);
centtrod=cluster.cluster_centers_;

# print(centtrod.tolist())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for pt in centtrod:
    # if abs(pt[0])>1500 or abs(pt[1])>1500 or abs(pt[2])>1500:
    #     print("*************") 
    #     print(pt);
    #     continue;
    ax.scatter(pt[0], pt[1], pt[2], c=None, depthshade=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

n1 = int(input('please input number of fastenings (width): '))
n2 = int(input('please input number of fastenings (height): '))
test = IdentifyClass(centtrod, n1=n1, n2=n2, sigma=25)
test.identify2()
test.check()

def sq(x): 
    return x*x;

def dist(x, y):
    if (len(x)!=len(y)):
        raise Exception("Dist")
    ret = 0
    for i in range(len(x)):
        ret += sq(x[i]-y[i])
    return sqrt(ret)

# dis=[]
# for pt1 in centtrod:
#     if (abs(pt1[0])>1500 or abs(pt1[1])>1500 or abs(pt1[2])>1500): 
#         continue;
#     for pt2 in centtrod:
#         if (abs(pt2[0])>1500 or abs(pt2[1])>1500 or abs(pt2[2])>1500 or (pt1[0] == pt2[0] and pt1[1] == pt2[1] and pt1[2] == pt2[2])): 
#             continue;
#         dis.append([dist(pt1, pt2)])
# dis=np.array(dis)

# cluster1=KMeans(n_clusters=6);
# cluster1=cluster1.fit(dis);
# centtrod1=cluster1.cluster_centers_;
# centtrod1=array(centtrod1);

# print(centtrod1);

# for length in centtrod1:
#     print("---"*10)
#     for pt1 in centtrod:
#         if (abs(pt1[0])>1500 or abs(pt1[1])>1500 or abs(pt1[2])>1500): 
#             continue;
#         for pt2 in centtrod:
#             if (abs(pt2[0])>1500 or abs(pt2[1])>1500 or abs(pt2[2])>1500 or (pt1[0] == pt2[0] and pt1[1] == pt2[1] and pt1[2] == pt2[2]) ): 
#                 continue;
#             if abs(dist(pt1, pt2)-length)<100:
#                 print(dist(pt1, pt2));

