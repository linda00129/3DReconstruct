import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import combined

# 换色部分（绿）rgb范围判断 需要更改可自行搜索
def judgeCol(r,g,b):
    # return b-r>50 and b-g>50 #blue
    # return g<=50 and b<=50 and r<=50 # black
    return g-r > 50 and g-b > 50 # green

def judgeBlank(r,g,b):
    return r<50 and g<50 and b<50

# 读取原图及fix后图片
picname = combined.picname
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
(kp1, des1) = sift.detectAndCompute(picL, None)
(kp2, des2) = sift.detectAndCompute(picR, None)

# —————— match ——————
# 最近邻近似
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
# 寻找k个最近邻
K = int(input('K = '))
matches = flann.knnMatch(des1, des2, k=K)
print('点对数: ', len(matches))

matchesMask = [[0 for j in range(K)] for i in range(len(matches))]

# 筛选点对
pts_left=[]
pts_right=[]

for i in range(len(matches)):
    for j, m in enumerate(matches[i]):
        x, y = kp1[m.queryIdx].pt
        p, q = kp2[m.trainIdx].pt
        r, g, b = imFL.getpixel((int(x), int(y)))
        r2, g2, b2 = imFR.getpixel((int(p), int(q)))
        if ((judgeCol(r, g, b) and (judgeCol(r2, g2, b2) or judgeBlank (r2, g2, b2))) or (judgeCol(r2, g2, b2) and (judgeCol(r, g, b) or judgeBlank (r, g, b)))) and abs(int(q-y)) < l*0.01 and abs(int(p-x)) < w*0.1:
            matchesMask[i][j] = 1
            pts_left.append([int(x),int(y)])
            pts_right.append([int(p),int(q)])
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
queryImagePath=pathFL
trainingImagePath=pathFR