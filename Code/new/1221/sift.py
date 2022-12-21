import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import combined

DISTANCE_RATIO = 0.6

def judgeCol(r,g,b):
    # return b-r>50 and b-g>50 #blue
    # return g<=50 and b<=50 and r<=50 # black
    return g-r > 50 and g-b > 50 # green

# # not red
# lower = np.array([10,43,46])
# upper = np.array([156,255,255])

#picname = "1027_2_1"
picname = combined.picname
# picname="2_10"

pathL="PicIn/"+picname+"-1.jpg"
pathR="PicIn/"+picname+"-2.jpg"
pathFL="PicOut/outpFix-"+picname+"-1.jpg"
pathFR="PicOut/outpFix-"+picname+"-2.jpg"
picL = cv2.imread(pathL,0)
picR = cv2.imread(pathR,0)
imL = Image.open(pathL)
imR = Image.open(pathR)
imFL = Image.open(pathFL)
imFR = Image.open(pathFR)
l = picL.shape[0]
w = picL.shape[1]

sift = cv2.SIFT_create()
# maskL=cv2.inRange(picL, lower, upper)
# maskR=cv2.inRange(picR, lower, upper)
(kp1, des1) = sift.detectAndCompute(picL, None)
(kp2, des2) = sift.detectAndCompute(picR, None)

# —————— match ——————
# 最近邻近似
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# 筛选点对
pts_left=[]
pts_right=[]

res = input('Ratio test? (y)/n: ')
if (res != 'n'):
    matches = flann.knnMatch(des1, des2, k=2)
    print(len(matches))
    matchesMask = [[0 for j in range(2)] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        x, y = kp1[m.queryIdx].pt
        p, q = kp2[m.trainIdx].pt
        r, g, b = imFL.getpixel((int(x), int(y)))
        r2, g2, b2 = imFR.getpixel((int(p), int(q)))
        if (m.distance < DISTANCE_RATIO * n.distance) and judgeCol(r, g, b) and judgeCol(r2, g2, b2) and (abs(int(q-y)) < l*0.015 and abs(int(p-x)) < w*0.1):
                matchesMask[i] = [1, 0]
                pts_left.append([int(x),int(y)]);
                pts_right.append([int(p),int(q)]);
else:
    K = int(input('K = '))
    matches = flann.knnMatch(des1, des2, k=K)
    print(len(matches))
    matchesMask = [[0 for j in range(K)] for i in range(len(matches))]

    for i in range(len(matches)):
        for j, m in enumerate(matches[i]):
            x, y = kp1[m.queryIdx].pt
            p, q = kp2[m.trainIdx].pt
            r, g, b = imFL.getpixel((int(x), int(y)))
            r2, g2, b2 = imFR.getpixel((int(p), int(q)))
            if judgeCol(r, g, b) and judgeCol(r2, g2, b2) and abs(int(q-y)) < l*0.015 and abs(int(p-x)) < w*0.1:
                matchesMask[i][j] = 1
                pts_left.append([int(x),int(y)]);
                pts_right.append([int(p),int(q)]);
                break


pts_left=np.array(pts_left)
pts_right=np.array(pts_right)
print(len(pts_left),len(pts_right))
drawParams = dict(matchColor=(255, 255, 255),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMask,
                  flags=0
                  )
resultImage = cv2.drawMatchesKnn(picL, kp1, picR, kp2, matches, None, **drawParams)
plt.xticks([]), plt.yticks([])
plt.imshow(resultImage), plt.show()
cv2.imwrite("PicOut/outpSift-"+picname+".jpg", resultImage)


# transfer value
queryImagePath=pathL
trainingImagePath=pathR
