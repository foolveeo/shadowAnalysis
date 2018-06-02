# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:22:09 2018

@author: Fulvio Bertolini
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def computeSunSkyRatioRGB(imgLit, imgShadow, ao, normals, sunDirection):
    
    zero = np.zeros((3), np.float)
    
    
    rgbLit = np.zeros((3), np.float)
    normalsLit = np.zeros((3), np.float)
    aoLit = 0
    nbrLit = 0
    
    rgbShadow = np.zeros((3), np.float)
    aoShadow = 0
    nbrShadow = 0
    
    for i in range(imgLit.shape[0]):
        for j in range(imgLit.shape[1]):
            if(not np.array_equal(imgLit[i,j], zero)):
                rgbLit = np.add(imgLit[i,j], rgbLit)
                normalsLit = np.add(normals[i,j], normalsLit)
                aoLit += ao[i,j]
                nbrLit += 1
            if(not np.array_equal(imgShadow[i,j], zero)):
                rgbShadow = np.add(imgShadow[i,j], rgbShadow)
                aoShadow += ao[i,j]
                nbrShadow += 1
                
    
    averageLit = np.divide(rgbLit, nbrLit)
    averageNormal = np.divide(normalsLit, nbrLit)
    averageAoLit = aoLit / nbrLit    
    
    averageShadow = np.divide(rgbShadow, nbrShadow)
    averageAoShadow = aoShadow / nbrShadow
    
    C = np.divide(averageShadow, averageLit)
    dotSunDir = np.dot(averageNormal, sunDirection)
    sunSkyRatioNumerator = averageAoShadow - np.multiply(C, averageAoLit)
    sunSkyRatioDenominator = np.multiply(C, dotSunDir)
    
    sunSkyRatio = np.divide(sunSkyRatioNumerator, sunSkyRatioDenominator)
    radianceFactor = 2 * np.cos(np.deg2rad(0.53/2))
    sunSkyRatio = np.divide(sunSkyRatio, radianceFactor)
    return sunSkyRatio
    
    
    

def computeAverageRatioChannel(imgLit, imgShadow, ch):
    
    cumulativeLit = 0
    cumulativeShadow = 0
    
    nbrLit = 0
    nbrShadow = 0
    
    for i in range(imgLit.shape[0]):
        for j in range(imgLit.shape[1]):
            if(imgLit[i,j,ch] != 0):
                cumulativeLit += imgLit[i,j,ch]
                nbrLit += 1
            if(imgLit[i,j,ch] != 0):
                cumulativeShadow += imgShadow[i,j,ch]
                nbrShadow += 1
                
    averageLit = cumulativeLit / nbrLit
    averageShadow = cumulativeShadow / nbrShadow
    
    return (averageLit / averageShadow)
    
    
def computeBrightnessRatio(imgRGB, litMask, shadowMask, ao, normals, sunDir, label):
    
    
    
    img_shadow = cv2.bitwise_and(imgRGB, imgRGB, mask=shadowMask)
    img_lit = cv2.bitwise_and(imgRGB, imgRGB, mask=litMask)
#
#    img_shadowHLS = cv2.cvtColor(img_shadow, cv2.COLOR_RGB2HLS)
#    img_litHLS = cv2.cvtColor(img_lit, cv2.COLOR_RGB2HLS)
#    
#    img_shadowYUV = cv2.cvtColor(img_shadow, cv2.COLOR_RGB2YUV)
#    img_litYUV = cv2.cvtColor(img_lit, cv2.COLOR_RGB2HSV)
#    
#    img_shadowLUV = cv2.cvtColor(img_shadow, cv2.COLOR_RGB2LUV)
#    img_litLUV = cv2.cvtColor(img_lit, cv2.COLOR_RGB2LUV)
#    
#    
    sunSkyRatioRGB = computeSunSkyRatioRGB(img_lit, img_shadow, ao, normals, sunDir)
    
#    sunSkyRatioHLS = computeAverageRatioChannel(img_litHLS, img_shadowHLS, 1)
#    sunSkyRatioYUV = computeAverageRatioChannel(img_litYUV, img_shadowYUV, 0)
#    sunSkyRatioLUV = computeAverageRatioChannel(img_litLUV, img_shadowLUV, 0)
#    print(label + ":")
    print("RGB ratios: ", sunSkyRatioRGB)
#    print("YUV ratio: ", sunSkyRatioYUV)
#    print("HLS ratios: ", sunSkyRatioHLS)
#    print("LUV ratios: ", sunSkyRatioLUV)    
#    


redBalanced1 = cv2.imread("./camerasession/DSC00012.JPG", 1)
redBalanced1 = cv2.cvtColor(redBalanced1, cv2.COLOR_BGR2RGB)

redBalanced2 = cv2.imread("./camerasession/DSC00015.JPG", 1)
redBalanced2 = cv2.cvtColor(redBalanced2, cv2.COLOR_BGR2RGB)

redBalanced3 = cv2.imread("./camerasession/DSC00019.JPG", 1)
redBalanced3 = cv2.cvtColor(redBalanced3, cv2.COLOR_BGR2RGB)

blueBalanced1 = cv2.imread("./camerasession/DSC00013.JPG", 1)
blueBalanced2 = cv2.imread("./camerasession/DSC00014.JPG", 1)
blueBalanced3 = cv2.imread("./camerasession/DSC00020.JPG", 1)

blueBalanced1 = cv2.cvtColor(blueBalanced1, cv2.COLOR_BGR2RGB)
blueBalanced2 = cv2.cvtColor(blueBalanced2, cv2.COLOR_BGR2RGB)
blueBalanced3 = cv2.cvtColor(blueBalanced3, cv2.COLOR_BGR2RGB)

greenBalanced1 = cv2.imread("./camerasession/DSC00011.JPG", 1)
greenBalanced2 = cv2.imread("./camerasession/DSC00016.JPG", 1)
greenBalanced3 = cv2.imread("./camerasession/DSC00021.JPG", 1)

greenBalanced1 = cv2.cvtColor(greenBalanced1, cv2.COLOR_BGR2RGB)
greenBalanced2 = cv2.cvtColor(greenBalanced2, cv2.COLOR_BGR2RGB)
greenBalanced3 = cv2.cvtColor(greenBalanced3, cv2.COLOR_BGR2RGB)

whiteBalanced1 = cv2.imread("./camerasession/DSC00010.JPG", 1)
whiteBalanced2 = cv2.imread("./camerasession/DSC00017.JPG", 1)
whiteBalanced3 = cv2.imread("./camerasession/DSC00018.JPG", 1)

whiteBalanced1 = cv2.cvtColor(whiteBalanced1, cv2.COLOR_BGR2RGB)
whiteBalanced2 = cv2.cvtColor(whiteBalanced2, cv2.COLOR_BGR2RGB)
whiteBalanced3 = cv2.cvtColor(whiteBalanced3, cv2.COLOR_BGR2RGB)

shadowMask1 = cv2.imread("./camerasession/masks/shadow_DSC00013.JPG", 0)
shadowMask2 = cv2.imread("./camerasession/masks/shadow_DSC00015.JPG", 0)
shadowMask3 = cv2.imread("./camerasession/masks/shadow_DSC00017.JPG", 0)

litMask1 = cv2.imread("./camerasession/masks/lit_DSC00013.JPG", 0)
litMask2 = cv2.imread("./camerasession/masks/lit_DSC00015.JPG", 0)
litMask3 = cv2.imread("./camerasession/masks/lit_DSC00017.JPG", 0)

shadowMask1 = cv2.resize(shadowMask1, (redBalanced1.shape[1], redBalanced1.shape[0]) )
shadowMask2 = cv2.resize(shadowMask2, (redBalanced2.shape[1], redBalanced2.shape[0]) )
shadowMask3 = cv2.resize(shadowMask3, (redBalanced3.shape[1], redBalanced3.shape[0]) )
litMask1 = cv2.resize(litMask1, (redBalanced1.shape[1], redBalanced1.shape[0]) )
litMask2 = cv2.resize(litMask2, (redBalanced2.shape[1], redBalanced2.shape[0]) )
litMask3 = cv2.resize(litMask2, (redBalanced3.shape[1], redBalanced3.shape[0]) )


ao = np.ones((whiteBalanced1.shape), np.float)
normals = np.ones((whiteBalanced1.shape), np.float)
sunDir = np.ones((3), np.float)


#computeBrightnessRatio(redBalanced1, litMask1, shadowMask1, ao, normals, sunDir, "redBalanced1")
#computeBrightnessRatio(greenBalanced1, litMask1, shadowMask1, ao, normals, sunDir, "greenBalanced1")
#computeBrightnessRatio(blueBalanced1, litMask1, shadowMask1, ao, normals, sunDir, "blueBalanced1")
computeBrightnessRatio(whiteBalanced1, litMask1, shadowMask1, ao, normals, sunDir, "whiteBalanced1")
#print()
#print()
#
#computeBrightnessRatio(redBalanced2, litMask2, shadowMask2, ao, normals, sunDir, "redBalanced2")
#computeBrightnessRatio(greenBalanced2, litMask2, shadowMask2, ao, normals, sunDir, "greenBalanced2")
#computeBrightnessRatio(blueBalanced2, litMask2, shadowMask2, ao, normals, sunDir, "blueBalanced1")
#computeBrightnessRatio(whiteBalanced2, litMask2, shadowMask2, ao, normals, sunDir, "whiteBalanced1")
#print()
#print()
#computeBrightnessRatio(redBalanced3, litMask3, shadowMask3, ao, normals, sunDir, "redBalanced3")
#computeBrightnessRatio(greenBalanced3, litMask3, shadowMask3, ao, normals, sunDir, "greenBalanced3")
#computeBrightnessRatio(blueBalanced3, litMask3, shadowMask3, ao, normals, sunDir, "blueBalanced3")
#computeBrightnessRatio(whiteBalanced3, litMask3, shadowMask3, ao, normals, sunDir, "whiteBalanced3")
#print()
#print()