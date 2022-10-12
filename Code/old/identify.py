import copy
from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt
import random
import math

A_test = [[  15.07804127,   20.64613755,  798.64647406],
 [-416.95409997, -183.90445827,  838.31546744],
 [-322.10424623,  205.88471794,  904.37973532],
 [ 220.29957998, -188.90744652,  678.53884932],
 [ 220.895611  ,   26.39474355,  707.92663104],
 [-339.08415024,  -38.28505614, 1025.77181478],
 [  41.99679344, -268.21212532,  893.46161022],
 [-166.64695384,   38.39469584,  851.9256966 ],
 [ 219.53016438,  214.62273227,  747.98255223],
 [-396.63801532, -272.78249996, 1000.94856641],
 [-163.21498252, -270.04465963,  956.1732426 ],
 [  40.6835142 ,  216.82941311,  815.98446871],
 [-201.61647592, -188.93956712,  811.93315582],
 [-109.58704138,  155.01056329, 1003.98818596],
 [ 252.00541205, -279.30221954,  828.92033596],
 [-358.4791915 ,   28.77077779,  877.83764671],
 [  -7.8235181 , -196.27696086,  758.39256473],
 [ 249.77586464,  -38.17515011,  850.96300591],
 [  63.48187726,  -23.96567398,  920.82541527],
 [-130.04040865,  -35.14901703,  992.63082557],
 [ 246.7548469 ,  154.11122081,  835.98206096],
 [-294.24254917,  135.63286834, 1052.78887142],
 [  65.13040472,  153.21258364,  937.75077021],
 [-127.10922958,  209.53045199,  872.91181089]]

def norm(u):
    ret = 0
    for x in u:
        ret += x*x
    return math.sqrt(ret)

def getPlaneAxis(u):
    v = np.array([u[0], u[1], -1]) / norm([u[0], u[1], -1])
    x = [1, 0, 0] - np.dot([1, 0, 0], v)*v
    x /= norm(x)
    y = np.cross(x, v)
    return [x, y]

def getLineAxis(u):
    v = np.array([1, u[0]])
    return v/norm(v)

def slope(u, v):
    if (v[0] == u[0]):
        return 1e9
    return (v[1] - u[1])/(v[0] - u[0])

class IdentifyClass:
    '''
    n1 = 2

    identify: ransac + ransac + sort
        n1 < n2 < n3
    
    identify2: ransac + rotation + sort
        n1 < n2 and n1 < n3
        n2为y方向 n3为x方向
        id[i][j][k]: z, y, x
    '''

    def __init__(self, A = A_test, n1 = 2, n2 = 3, sigma = 50):
        self.A = np.array(A)
        self.N = len(A)
        self.n1 = n1
        self.n2 = n2
        self.n3 = int(self.N / self.n1 / self.n2)
        self.sigma = sigma

    def ransacPlane(self, Id, sigma=50):
        best_a, best_b, best_c = 0, 0, 0
        maxTot = 0
        ret = []
        iters = 1000

        while iters > 0:
            iters -= 1
            sample_id = random.sample(range(len(Id)), 3)
            [x1, y1, z1] = self.A[Id[sample_id[0]]]
            [x2, y2, z2] = self.A[Id[sample_id[1]]]
            [x3, y3, z3] = self.A[Id[sample_id[2]]]
            a = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1)
            b = (z2-z1)*(x3-x1) - (x2-x1)*(z3-z1)
            c = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
            d = a*x1 + b*y1 + c*z1 # ax + by + cz = d
            s = norm([a, b, c])
            
            tot = 0
            inliner = []
            for i in Id:
                if abs(a*self.A[i][0] + b*self.A[i][1] + c*self.A[i][2] - d) < sigma * s:
                    tot += 1
                    inliner.append(i)

            if (tot > maxTot):
                maxTot = tot
                best_a, best_b, best_c, best_d = a, b, c, d
                ret = inliner
            if (tot * self.n1 == self.N):
                break
        return ret

    def leastSqPlane(self, Id):
        A = np.zeros((3,3))
        for i in Id:
            A[0][0] += self.A[i][0]**2
            A[0][1] += self.A[i][0]*self.A[i][1]
            A[0][2] += self.A[i][0]
            A[1][1] += self.A[i][1]**2
            A[1][2] += self.A[i][1]
        A[1][0] = A[0][1]
        A[2][0] = A[0][2]
        A[2][1] = A[1][2]
        A[2][2] = len(Id)

        B = np.zeros(3)
        for i in Id:
            B[0] += self.A[i][0]*self.A[i][2]
            B[1] += self.A[i][1]*self.A[i][2]
            B[2] += self.A[i][2]
        A_inv = np.linalg.inv(A)
        ret = np.dot(A_inv, B) # z = ret[0]*x + ret[1]*y +ret[2]
        # print("LeastSqPlane = ", ret)
        return ret

    def ransacLine(self, Id, sigma=30):
        best_a, best_b = 0, 0
        maxTot = 0
        ret = []
        iters = 10000

        while iters > 0:
            iters -= 1
            sample_id = random.sample(range(len(Id)), 2)
            [x1, y1] = self.A_2d[Id[sample_id[0]]]
            [x2, y2] = self.A_2d[Id[sample_id[1]]]
            a = (y2 - y1) / (x2 - x1)
            b = (x2*y1 - x1*y2) / (x2 - x1)
            s = norm([a, 1])
            
            tot = 0
            inliner = []
            for i in Id:
                if abs(a*self.A_2d[i][0] + b - self.A_2d[i][1]) < sigma * s:
                    tot += 1
                    inliner.append(i)

            if (tot > maxTot):
                maxTot = tot
                best_a, best_b = a, b
                ret = inliner
            if (tot * self.n1 * self.n2 == self.N):
                break
        return ret

    def leastSqLine(self, Id):
        A = np.zeros((2,2))
        for i in Id:
            A[0][0] += self.A_2d[i][0]**2
            A[0][1] += self.A_2d[i][0]
            A[1][0] += self.A_2d[i][0]
        A[1][1] = len(Id)

        B = np.zeros(2)
        for i in Id:
            B[0] += self.A_2d[i][0]*self.A_2d[i][1]
            B[1] += self.A_2d[i][1]
        A_inv = np.linalg.inv(A)
        ret = np.dot(A_inv, B) # y = ret[0]*x + ret[1]
        # print("LeastSqLine = ", ret)
        return ret

    def identify(self):
        self.planeId = np.zeros(self.N, dtype=np.int32)
        self.lineId = np.zeros(self.N, dtype=np.int32)
        self.plane = np.zeros((self.n1, 3))
        self.line = np.zeros((self.n1, self.n2, 2))
        self.A_2d = np.zeros((self.N, 2))
        self.A_1d = np.zeros(self.N)
        self.planePointsId = np.zeros((self.n1, self.n2*self.n3), dtype=np.int32)
        self.linePointsId = np.zeros((self.n1, self.n2, self.n3), dtype=np.int32)
        self.rankPlane = np.zeros(self.n1, dtype=np.int32)
        self.rankLine = np.zeros((self.n1, self.n2), dtype=np.int32)
        self.rankPoint = np.zeros(self.N, dtype=np.int32)
        self.id = np.zeros((self.n1, self.n2, self.n3), dtype=np.int32)

        Id1 = np.array(list(range(self.N)), dtype=np.int32)
        for i in range(self.n1):
            self.planePointsId[i] = self.ransacPlane(Id1, sigma=50)
            for j in self.planePointsId[i]:
                self.planeId[j] = i
            Id1 = np.setdiff1d(Id1, self.planePointsId[i])
            self.plane[i] = self.leastSqPlane(self.planePointsId[i])

            x_axis, y_axis = getPlaneAxis(self.plane[i])
            for j in self.planePointsId[i]:
                y = [self.A[j][0], self.A[j][1], self.A[j][2] - self.plane[i][2]]
                self.A_2d[j] = [np.dot(y, x_axis), np.dot(y, y_axis)]
            # print("Plane: ", self.planePointsId[i])
            
            Id2 = copy.copy(self.planePointsId[i])
            for j in range(self.n2):
                self.linePointsId[i][j] = self.ransacLine(Id2, sigma=30)
                # print('Line: ', self.linePointsId[i][j])
                for k in self.linePointsId[i][j]:
                    self.lineId[k] = j
                Id2 = np.setdiff1d(Id2, self.linePointsId[i][j])
                self.line[i][j] = self.leastSqLine(self.linePointsId[i][j])

                axis = getLineAxis(self.line[i][j])
                for k in self.linePointsId[i][j]:
                    self.A_1d[k] = np.dot([self.A[k][0], self.A[k][1] - self.line[i][j][1]], axis)
                
        t1 = [[self.plane[i][2], i] for i in range(self.n1)]
        t1.sort(key = lambda x: x[0])
        for i in range(self.n1):
            self.rankPlane[t1[i][1]] = i
        
        for i in range(self.n1):
            t2 = [[self.line[i][j][1], j] for j in range(self.n2)]
            t2.sort(key = lambda x: x[0])
            for j in range(self.n2):
                self.rankLine[i][t2[j][1]] = j
            
            for j in range(self.n2):
                t3 = [[self.A_1d[self.linePointsId[i][j][k]], k] for k in range(self.n3)]
                t3.sort(key = lambda x: x[0])
                for k in range(self.n3):
                    self.rankPoint[self.linePointsId[i][j][t3[k][1]]] = k
        
        for i in range(self.N):
            p = int(self.planeId[i])
            l = int(self.lineId[i])
            self.id[self.rankPlane[p]][self.rankLine[p][l]][self.rankPoint[i]] = i
    
    def cmp(self, a, b):
        x0 = np.array(self.A_2d[self.convexId[0]])
        t = np.cross(np.array(self.A_2d[a])-x0, np.array(self.A_2d[b])-x0)
        if t < 0 or (t == 0 and self.A_2d[a][0] > self.A_2d[b][0]):
            return 1
        else:
            return -1

    def identify2(self):
        while True:
            self.planeId = np.zeros(self.N, dtype=np.int32)
            self.plane = np.zeros((self.n1, 3))
            self.A_2d = np.zeros((self.N, 2))
            self.planePointsId = np.zeros((self.n1, self.n2*self.n3), dtype=np.int32)
            self.rankPlane = np.zeros(self.n1, dtype=np.int32)
            self.rankX = np.zeros(self.N, dtype=np.int32)
            self.rankY = np.zeros(self.N, dtype=np.int32)
            self.convexId = np.array([], dtype=np.int32)
            self.A_2dRot = np.zeros((self.N, 2))
            self.id = np.zeros((self.n1, self.n2, self.n3), dtype=np.int32)

            Id1 = np.array(list(range(self.N)), dtype=np.int32)
            for i in range(self.n1):
                cnt = 0
                tmp = self.ransacPlane(Id1, self.sigma)
                while len(tmp) != self.n2 * self.n3 and cnt < 100:
                    tmp = self.ransacPlane(Id1, self.sigma)
                    cnt += 1
                self.planePointsId[i] = tmp

                for j in self.planePointsId[i]:
                    self.planeId[j] = i
                Id1 = np.setdiff1d(Id1, self.planePointsId[i])
                self.plane[i] = self.leastSqPlane(self.planePointsId[i])

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for j in range(self.N):
                    ax.scatter(self.A[j][0], self.A[j][1], self.A[j][2], c=None, depthshade=True)
                X, Y = np.meshgrid(np.linspace(-300, 300, 101), np.linspace(-300, 300, 101))
                Z = self.plane[i][0]*X + self.plane[i][1]*Y + self.plane[i][2]
                ax.plot_surface(X, Y, Z, alpha=0.5)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                plt.show()

                x_axis, y_axis = getPlaneAxis(self.plane[i])
                minY = 1e9
                minId = 0
                t1 = []
                for j in self.planePointsId[i]:
                    t1.append(j)
                    y = [self.A[j][0], self.A[j][1], self.A[j][2] - self.plane[i][2]]
                    self.A_2d[j] = [np.dot(y, x_axis), np.dot(y, y_axis)]
                    if (self.A_2d[j][1] < minY):
                        minY = self.A_2d[j][1]
                        minId = j
                
                self.convexId = np.array([minId], dtype=np.int32)
                t1.remove(self.convexId[0])
                t1.sort(key = cmp_to_key(self.cmp))

                cnt = 0
                for j in t1:
                    while cnt > 0 and np.cross(self.A_2d[self.convexId[cnt]] - self.A_2d[self.convexId[cnt-1]], self.A_2d[j] - self.A_2d[self.convexId[cnt]]) <= 0:
                        self.convexId = np.delete(self.convexId, cnt)
                        cnt -= 1
                    self.convexId = np.append(self.convexId, j)
                    cnt += 1

                s = []
                x0 = self.A_2d[self.convexId[0]]
                for j in self.convexId[1:]:
                    s.append(math.atan((self.A_2d[j][1] - x0[1]) / (self.A_2d[j][0] - x0[0])))
                    x0 = self.A_2d[j]
                s.append(math.atan((self.A_2d[self.convexId[0]][1] - x0[1]) / (self.A_2d[self.convexId[0]][0] - x0[0])))

                minS = min(s)
                maxS = max(s)
                sum1, sum2 = 0, 0
                cnt1, cnt2 = 0, 0
                for j in s:
                    if j < (maxS + minS) / 2:
                        if maxS - j > 5/6*math.pi: # 摆烂了
                            cnt2 += 1
                            sum2 += j + math.pi
                        else:
                            cnt1 += 1
                            sum1 += j
                    else:
                        cnt2 += 1
                        sum2 += j
                r1 = sum1 / cnt1
                r2 = sum2 / cnt2
                # print(s)
                # print(r1, r2)

                if abs(r1) > abs(r2):
                    r1, r2 = r2, r1
                sin1, sin2 = math.sin(r1), math.sin(r2)
                cos1, cos2 = math.cos(r1), math.cos(r2)

                for j in self.planePointsId[i]:
                    # self.A_2dRot[j][0] = math.cos(r)*self.A_2d[j][0] + math.sin(r)*self.A_2d[j][1]
                    # self.A_2dRot[j][1] = - math.sin(r)*self.A_2d[j][0] + math.cos(r)*self.A_2d[j][1]
                    self.A_2dRot[j][0] = (sin2*self.A_2d[j][0] - cos2*self.A_2d[j][1]) / (sin2*cos1 - cos2*sin1)
                    self.A_2dRot[j][1] = (sin1*self.A_2d[j][0] - cos1*self.A_2d[j][1]) / (sin1*cos2 - cos1*sin2)
                
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                for j in self.planePointsId[i]:
                    ax1.scatter(self.A_2d[j][0], self.A_2d[j][1])
                    ax2.scatter(self.A_2dRot[j][0], self.A_2dRot[j][1])
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                plt.show()

                rank1 = list(range(self.n2 * self.n3))
                rank2 = list(range(self.n2 * self.n3))
                rank1.sort(key = lambda x: self.A_2dRot[self.planePointsId[i][x]][0])
                rank2.sort(key = lambda x: self.A_2dRot[self.planePointsId[i][x]][1])
                for j, k in enumerate(rank1):
                    self.rankX[self.planePointsId[i][k]] = int(j / self.n2)
                for j, k in enumerate(rank2):
                    self.rankY[self.planePointsId[i][k]] = int(j / self.n3)

            t1 = [[self.plane[i][2], i] for i in range(self.n1)]
            t1.sort(key = lambda x: x[0])
            for i in range(self.n1):
                self.rankPlane[t1[i][1]] = i
            for i in range(self.N):
                p = self.planeId[i]
                self.id[self.rankPlane[p]][self.rankY[i]][self.rankX[i]] = i

            res = input('保留此结果? (y)/n: ')
            if (res != 'n'):
                break

    def check(self, lx = (0, 0, 1e9), ly = (0, 0, 1e9), lz = (0, 0, 1e9)):
        lenX = np.zeros((self.n1, self.n2, self.n3 - 1))
        lenY = np.zeros((self.n1, self.n2 - 1, self.n3))
        lenZ = np.zeros((self.n1 - 1, self.n2, self.n3))
        for i in range(self.n1):
            for j in range(self.n2):
                for k in range(self.n3):
                    print(i, j, k, self.A[self.id[i][j][k]])
                    if i < self.n1 - 1:
                        lenZ[i][j][k] = norm(self.A[self.id[i+1][j][k]] - self.A[self.id[i][j][k]])
                    if j < self.n2 - 1:
                        lenY[i][j][k] = norm(self.A[self.id[i][j+1][k]] - self.A[self.id[i][j][k]])
                    if k < self.n3 - 1:
                        lenX[i][j][k] = norm(self.A[self.id[i][j][k+1]] - self.A[self.id[i][j][k]])
        print()
        print('前后')
        print(lenZ)
        # print('expected = ', lz[1])
        print('mean = ', np.mean(lenZ))
        print('std = ', np.std(lenZ))
        # print('max = ', np.max(lenZ))
        # print('min = ', np.min(lenZ))
        # for i in range(self.n1 - 1):
        #     for j in range(self.n2):
        #         for k in range(self.n3):
        #             if (lenZ[i][j][k] < lz[0]):
        #                 print(i, j, k, lenZ[i][j][k], '过短')
        #             if (lenZ[i][j][k] > lz[2]):
        #                 print(i, j, k, lenZ[i][j][k], '过长')

        print()
        print('竖直')
        print(lenY)
        # print('expected = ', ly[1])
        print('mean = ', np.mean(lenY))
        print('std = ', np.std(lenY))
        # print('max = ', np.max(lenY))
        # print('min = ', np.min(lenY))
        # for i in range(self.n1):
        #     for j in range(self.n2 - 1):
        #         for k in range(self.n3):
        #             if (lenY[i][j][k] < ly[0]):
        #                 print(i, j, k, lenY[i][j][k], '过短')
        #             if (lenY[i][j][k] > ly[2]):
        #                 print(i, j, k, lenY[i][j][k], '过长')

        print()
        print('水平')
        print(lenX)
        # print('expected = ', lx[1])
        print('mean = ', np.mean(lenX))
        print('std = ', np.std(lenX))
        # print('max = ', np.max(lenX))
        # print('min = ', np.min(lenX))
        # for i in range(self.n1):
        #     for j in range(self.n2):
        #         for k in range(self.n3 - 1):
        #             if (lenX[i][j][k] < lx[0]):
        #                 print(i, j, k, lenX[i][j][k], '过短')
        #             if (lenX[i][j][k] > lx[2]):
        #                 print(i, j, k, lenX[i][j][k], '过长')



# test = IdentifyClass()
# test.identify2()
# test.check([200, 235, 270], [220, 250, 280], [120, 140, 160])