import random

import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import random
import time

def imageToArr(imgPath):
    imgArrTemp = np.array(Image.open(imgPath))
    imgArr = np.array([[0]*len(imgArrTemp[0])]*len(imgArrTemp))
    for i in range(len(imgArrTemp)):
        for j in range(len(imgArrTemp[0])):
            imgArr[i][j] = sum(imgArrTemp[i][j])/3
    return imgArr

def getWs(imgPath, wlen):
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, wlen)
    corners = np.int0(corners)
    output = []
    img = imageToArr(imgPath)
    imgShow = cv2.imread(imgPath)
    for corner in corners:
        y,x = corner.ravel()
        w = [ [0]*wlen for i in range(wlen)]
        cv2.rectangle(imgShow, (y-wlen//2, x-wlen//2), (y+wlen//2, x+wlen//2), (255, 0, 0), 1)
        for i in range(wlen):
            for j in range(wlen):
                w[i][j] = img[x + i - wlen//2][y + j - wlen//2]
        output.append((x,y,w))

    plt.imshow(imgShow), plt.savefig("fig" + imgPath)
    return output

def genFeatDescr(img1,img2):
    imgShow1 = cv2.imread(img1)
    imgShow2 = cv2.imread(img2)
    sift = cv2.SIFT_create()
    gray1 = cv2.cvtColor(imgShow1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(imgShow2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    return (kp1,des1),(kp2,des2)

def features(imgPath1, imgPath2, wlen):
    data1, data2 = genFeatDescr(imgPath1, imgPath2)
    imgShow1 = cv2.imread(imgPath1)
    imgShow2 = cv2.imread(imgPath2)
    kp1 = data1[0]
    des1 = data1[1]
    kp2 = data2[0]
    des2 = data2[1]
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # knn_matches = matcher.knnMatch(des1, des2, 2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.4
    good_matches = []
    for i in range(min(len(des1), len(des2))):
        m,n = knn_matches[i]
        if m.distance < ratio_thresh * n.distance:
            output = (int(kp1[n.queryIdx].pt[0]),int(kp1[n.queryIdx].pt[1]), int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1]))
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)

            cv2.circle(imgShow1, (output[0], output[1]), 5, (r,g,b), 2)
            cv2.circle(imgShow2, (output[2], output[3]), 5, (r,g,b), 2)
            good_matches.append(output)
    cv2.imwrite("tes1.png", imgShow1)
    cv2.imwrite("tes2.png", imgShow2)
    return good_matches

def flowMap(fts, w, h, squareLen, flMap):
    data = []
    for ft in fts:
        mag = np.abs(ft[0]-ft[2])+np.abs(ft[1]-ft[3])
        data.append((ft[0], ft[1], mag))
    img = np.array([np.array([0 for i in range(w)]) for j in range(h)])
    for datum in data:
        for i in range(squareLen):
            if len(img) <= datum[1] + i:
                break
            for j in range(squareLen):
                if len(img[i]) <= datum[0] + j:
                    break
                img[datum[1]+i][datum[0]+j] = datum[2]
    np.save(flMap, img)
    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(suppress=True)
    # print(np.load(flMap))


def lucus(img1, img2, flMap):
    ft = features(img1, img2, 15)
    img = cv2.imread(img1)
    flowMap(ft, len(img[0]), len(img), 2, flMap)

def createClowMaps(startInd, endInd):
    writeIter = 0
    for i in range(startInd, endInd+1):
        num = str(i)
        file1 = str("0"*(6-len(num))) + num
        path1 = "image_2/" + file1 + ".png"
        path2 = "image_3/" + file1 + ".png"
        print(path1)
        print(path2)
        print("______________")
        lucus(path1, path2, "FlowMap/flowMapV" + str(writeIter) + ".npy")
        writeIter += 1

def compare(disp, groundTruth, k):
    d = np.load(disp)
    gt = np.load(groundTruth)
    num = 0
    den = 0
    for x in range(len(d)):
        for y in range(len(d[x])):
            pxd = d[x][y]
            pxgt = gt[x][y]
            if not pxd == 0 and not pxgt <= 0:
                # print(str(pxd) + " " + str(pxgt))
                if (pxd - pxgt)**2 > k**2:
                    num += 1
                den += 1
    return num, den

def compareAll(start, end, k):
    num = 0
    den = 0
    for i in range(start, end+1):
        number = str(i)
        file1 = str("0"*(6-len(number))) + number
        gt = "GroundTruth/" + file1 + ".npy"
        disp = "FlowMap/flowMapV" + number + ".npy"
        tempNum, tempDen = compare(disp, gt, k)
        num += tempNum
        den += tempDen
        print(str(i) + "  " + str(num/den))
    print(num/den)
    return num/den


if __name__ == '__main__':
    compareAll(0, 7480, 1)