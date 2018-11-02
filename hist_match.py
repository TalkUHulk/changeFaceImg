import cv2 as cv
import numpy as np

def histMatch_core(src,dst,mask = None):
    srcHist = [0] * 256
    dstHist = [0] * 256
    srcProb = [.0] * 256; # 源图像各个灰度概率
    dstProb = [.0] * 256; # 目标图像各个灰度概率


    for h in range(src.shape[0]):
         for w in range(src.shape[1]):
             if mask is None:
                 srcHist[int(src[h,w])] += 1
                 dstHist[int(dst[h,w])] += 1
             else:
                 if mask[h,w] > 0:
                     srcHist[int(src[h, w])] += 1
                     dstHist[int(dst[h, w])] += 1


    resloution = src.shape[0] * src.shape[1]

    if mask is not None:
        resloution = 0
        for h in range(mask.shape[0]):
            for w in range(mask.shape[1]):
                if mask[h, w] > 0:
                    resloution += 1

    for i in range(256):
        srcProb[i] = srcHist[i] / resloution
        dstProb[i] = dstHist[i] / resloution

     # 直方图均衡化
    srcMap = [0] * 256
    dstMap = [0] * 256

    # 累积概率
    for i in range(256):
        srcTmp = .0
        dstTmp = .0
        for j in range(i + 1):
            srcTmp += srcProb[j]
            dstTmp += dstProb[j]

        srcMapTmp = srcTmp * 255 + .5
        dstMapTmp = dstTmp * 255 + .5
        srcMap[i] = srcMapTmp if srcMapTmp <= 255.0 else 255.0
        dstMap[i] = dstMapTmp if dstMapTmp <= 255.0 else 255.0

    matchMap = [0] * 256
    for i in range(256):
        pixel = 0
        pixel_2 = 0
        num = 0 # 可能出现一对多
        cur = int(srcMap[i])
        for j in range(256):
            tmp = int(dstMap[j])
            if cur == tmp:
                pixel += j
                num += 1
            elif cur < tmp: # 概率累计函数 递增
                pixel_2 = j
                break

        matchMap[i] = int(pixel / num) if num > 0 else int(pixel_2)

    newImg = np.zeros(src.shape[:2], dtype=np.uint8)
    for h in range(src.shape[0]):
        for w in range(src.shape[1]):
            if mask is None:
                newImg[h,w] = matchMap[src[h,w]]
            else:
                if mask[h,w] > 0:
                    newImg[h, w] = matchMap[src[h, w]]
                else:
                    newImg[h, w] = src[h, w]

    return newImg



# src1 src2 mask must have the same size
def histMatch(src1,src2,mask = None,dst = None):

    sB,sG,sR = cv.split(src1)
    dB,dG,dR = cv.split(src2)

    if mask.shape[2] > 1:
        rM,gM,bM = cv.split(mask)
        nB = histMatch_core(sB, dB, rM)
        nG = histMatch_core(sG, dG, gM)
        nR = histMatch_core(sR, dR, bM)
    else:
        nB = histMatch_core(sB,dB,mask)
        nG = histMatch_core(sG,dG,mask)
        nR = histMatch_core(sR,dR,mask)

    newImg = cv.merge([nB,nG,nR])

    if dst is not None:
        dst = newImg

    return newImg