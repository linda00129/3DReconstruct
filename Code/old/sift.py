import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

#Orange is just a random name.
def if_orange(color):
    orange = [3, 24, 151]
    blue = [10, 0, 212]
    diff = abs(int(color[2]) - int(orange[2])) * 0.30 + abs(int(color[1]) - int(orange[1])) * 0.35 + \
           abs(int(color[0]) - int(orange[0])) * 0.35
    threshold = 30
    return diff <= threshold

def if_orange2(color):
    # orange = [24, 24, 160]
    # blue = [10, 0, 212]
    # diff = abs(int(color[2]) - int(orange[2])) * 0.30 + abs(int(color[1]) - int(orange[1])) * 0.35 + \
    #        abs(int(color[0]) - int(orange[0])) * 0.35
    # threshold = 50
    # if (diff <= threshold):
    #     print('color = ', color, 'diff = ', diff)
    # return diff <= threshold
    # return color[0] <= 100 and color[1] <= 100 and color[2] >= 140
    return color[2] - color[0] > 50 and color[2] - color[1] > 50

queryImagePath = 'NewPicture/0807_1.jpg'
trainingImagePath = 'NewPicture/0807_2.jpg'

queryImage = cv2.imread(queryImagePath, 0)
im = Image.open(queryImagePath)
trainingImage = cv2.imread(trainingImagePath, 0)
im2 = Image.open(trainingImagePath)

l = queryImage.shape[0]
w = queryImage.shape[1]

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(queryImage, None)
kp2, des2 = sift.detectAndCompute(trainingImage, None)

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# K越大，匹配对越多，匹配质量越低
K = int(input('K = '))
matches = flann.knnMatch(des1, des2, k=K)

print(len(matches))

matchesMask = [[0 for j in range(K)] for i in range(len(matches))]

pts_left=[]
pts_right=[]

# with open("out_left.txt","w") as f:
#     for i, (m, n) in enumerate(matches):
#         x, y = kp1[m.queryIdx].pt
#         p, q = kp2[m.trainIdx].pt
#         r, g, b = im.getpixel((int(x), int(y)))
#         r2, g2, b2 = im2.getpixel((int(p), int(q)))
#         if (m.distance < 1 * n.distance) and (if_orange((r, g, b))) and (abs(int(q-y)) < l*0.05 and abs(int(p-x)) < w*0.2):
#                 matchesMask[i] = [1, 0]
#                 pts_left.append([int(x),int(y)]);
#                 pts_right.append([int(p),int(q)]);

with open("out_left.txt","w") as f:
    for i in range(len(matches)):
        for j, m in enumerate(matches[i]):
            x, y = kp1[m.queryIdx].pt
            p, q = kp2[m.trainIdx].pt
            r, g, b = im.getpixel((int(x), int(y)))
            r2, g2, b2 = im2.getpixel((int(p), int(q)))
            if if_orange2((r, g, b)) and if_orange2((r2, g2, b2)) and abs(int(q-y)) < l*0.015 and abs(int(p-x)) < w*0.1:
                matchesMask[i][j] = 1
                pts_left.append([int(x),int(y)]);
                pts_right.append([int(p),int(q)]);
                break

# pts_right=[];
# with open("out_right.txt","w") as f:
#     for i, (m, n) in enumerate(matches):
#         x, y = kp1[m.queryIdx].pt
#         p, q = kp2[m.trainIdx].pt
#         r, g, b = im.getpixel((int(x), int(y)))
#         r2, g2, b2 = im2.getpixel((int(p), int(q)))
#         # if (m.distance < 1.0 * n.distance) & (if_orange((r, g, b))):
#         if m.distance < 0.6 * n.distance and if_orange((r, g, b)) and if_orange((r, g, b)):
#             # if (abs(int(q-y)) < 500) and (abs(int(p-x)) < 500):
#                 matchesMask[i] = [1, 0]
#                 pts_right.append([int(p),int(q)]);

pts_left=np.array(pts_left);
pts_right=np.array(pts_right);
print(len(pts_left),len(pts_right));
with open("out_left.txt","w") as f:
    print(pts_left, file=f)
with open("out_right.txt","w") as f:
    print(pts_right, file=f)
drawParams = dict(matchColor=(255, 255, 255),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMask,
                  flags=0
                  )
resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)
plt.xticks([]), plt.yticks([])
plt.imshow(resultImage), plt.show()
cv2.imwrite("matches.png", resultImage)