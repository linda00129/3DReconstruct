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

# 1023
# proj_mats_left = np.array([
# [857.000134, 0.338953, 714.088501, 0.000000],
# [0.000000, 856.418862, 527.909923, 0.000000],
# [0.000000, 0.000000, 1.000000, 0.000000]], dtype=float)

# proj_mats_right = np.array([
# [852.080163, -0.131339, 719.952159, 52068.998017],
# [-3.336481, 856.283714, 528.118569, -614.795274],
# [-0.006862, -0.000256, 0.999976, -1.166534]], dtype=float)

# 1127_1
# proj_mats_left = np.array([
# [853.036139, 0.604054, 718.916147, 0.000000],
# [0.000000, 853.152709, 524.997015, 0.000000],
# [0.000000, 0.000000, 1.000000, 0.000000]], dtype=float)

# proj_mats_right = np.array([
# [841.743210, 0.778889, 732.106008, 49743.113997],
# [-9.391729, 852.400423, 526.133751, -1125.499960],
# [-0.015562, -0.001430, 0.999878, -1.695399]], dtype=float)

#1127_2
proj_mats_left = np.array([
[854.345691, -0.543883, 717.278815, 0.000000],
[0.000000, 853.400718, 531.495797, 0.000000],
[0.000000, 0.000000, 1.000000, 0.000000]], dtype=float)

proj_mats_right = np.array([
[846.407744, -1.941970, 726.626392, 51159.529747],
[-6.477397, 851.910622, 533.841646, -406.405531],
[-0.010995, -0.002797, 0.999936, 0.156549]], dtype=float)

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

print(centtrod.tolist())

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

width, height, length, Scaling = 0, 0, 0, False
res = input('Scaling to known distance? (y)/n: ')
if (res != 'n'):
    width = float(input('please input width: '))
    height = float(input('please input height: '))
    length = float(input('please input length: '))
    Scaling = True

iden = IdentifyClass(A=centtrod, n1=n1, n2=n2, sigma=50)
iden.identify2()
iden.check(scaling=Scaling, lx=length, ly=height, lz=width)
