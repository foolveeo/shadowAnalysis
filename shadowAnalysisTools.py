# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 15:39:59 2018

@author: Fulvio Bertolini
"""
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.signal as signal


def computeRadiances(Ea, Es):
    
    radSky = Ea / np.pi
    
    d = 0.53
    sunMultiplier = 1 / (2 * np.pi * (1 - np.cos(np.deg2rad(d/2))))
    radSun = Es * sunMultiplier
    return radSky, radSun
    

def compute_ao_average(rawFeatures_filtered):
    ao_avg = np.mean(rawFeatures_filtered[:,7])
    return ao_avg
    
    
def computeIrradiancesFulvio(k, sunSkyRatioR, sunSkyRatioB):
    Esun = np.ones((3). np.float)
    Esky = np.ones((3). np.float)

    Esky[1] = 1
    Esun[1] = Esky[1] * k
    
    
def rgb_to_chRchB(r,g,b):
    chR = r/g
    chB = b/g
    
    return (chR, chB)

def compute_k_g(shadowG, litG, aoShadow, aoLit, ns):
    k_g1 = (litG * aoShadow) / (shadowG * ns)
    k_g2 = aoLit/ns
    k_g = k_g1 - k_g2
    
    return k_g
    

def compute_k_rgb(shadowRGB, litRGB, aoShadow, aoLit, ns):
    k_r = compute_k_g(shadowRGB[0], litRGB[0], aoShadow, aoLit, ns)
    k_g = compute_k_g(shadowRGB[1], litRGB[1], aoShadow, aoLit, ns)
    k_b = compute_k_g(shadowRGB[2], litRGB[2], aoShadow, aoLit, ns)
    
    print("k_r: ", k_r, "k_g: ", k_g, "k_b: ", k_b)
    k = (k_r + k_g + k_b) / 3.0
    
    return k
    

def computeRedBlueRatio(rbShadow, rbLit, aoLit, ns, k):
    rRatio = rbLit[0] / rbShadow[0]
    bRatio = rbLit[1] / rbShadow[1]
    multiplier = (aoLit + (ns*k)) / ns
    rRatio = rRatio * multiplier
    bRatio = bRatio * multiplier
    
    redRatio = rRatio - (aoLit / ns)
    blueRatio = bRatio - (aoLit / ns)
    
    return redRatio, blueRatio

def compute_k(rawFeatures):
    averageG = np.mean(rawFeatures[:,1])
    return ((np.pi * averageG) / 0.3)


def computeIrradiances(C, aoShadow, aoLit, nsLit, nsShadow, k, aoAvg, nsAvg):

    vecRatio = nsAvg / nsLit
    d1 = np.multiply(vecRatio, (np.subtract(np.divide(aoLit, C), aoShadow)))
    denominatorSky = np.add(d1, aoAvg)
    numerator = np.subtract(aoLit, np.multiply(C, aoShadow))
    denominatorSun = np.multiply(nsLit, C)
    frac = np.divide(numerator, denominatorSun)
    
    Ea = np.divide(k, denominatorSky)
    Es = np.multiply(Ea, frac)
    
    #d1 = vecRatio * ((ao_t1[x,y] / C) - ao_t0[x,y])
    #denominatorSky = aoWhiteBal + d1
    #numerator = ao_t1[x,y] - (C * ao_t0[x,y])
    #denominatorSun = np.dot(normals_t1[x,y], sunDir) * C
    #if(denominatorSun != 0.0 and denominatorSky != 0.0):
    #frac = ( numerator / denominatorSun )
    #newMask[x,y,ch] = 255;
    #Ea[x,y,ch] = k / denominatorSky
    #Es[x,y,ch] = Ea[x,y,ch] * frac
    
    
    return Ea, Es
def computeMeans(rawFeaturesShadow, rawFeaturesLit, processedFeaturesShadow, processedFeaturesLit):
    
    rgbShadow = meanRGB(rawFeaturesShadow)
    rgbLit = meanRGB(rawFeaturesLit)
    nsShadow = meanNS(processedFeaturesShadow)
    nsLit = meanNS(processedFeaturesLit)
    aoShadow = meanAO(rawFeaturesShadow)
    aoLit = meanAO(rawFeaturesLit)
    aoLit = meanAO(rawFeaturesShadow)
    
    return (rgbShadow, rgbLit, nsShadow, nsLit, aoShadow, aoLit)
    
def meanRGB(rawFeatures):
    rgbCumulative = np.zeros((3), np.float)
    for i in range(rawFeatures.shape[0]):
        for ch in range(3):    
            rgbCumulative[ch] = np.add(rgbCumulative[ch], rawFeatures[i,ch])
        
    rgbCumulative = np.divide(rgbCumulative, rawFeatures.shape[0])
    return rgbCumulative


def meanAO(rawFeatures):
    aoCumulative = 0
    for i in range(rawFeatures.shape[0]):   
        aoCumulative += rawFeatures[i,7]
        
    aoCumulative /= rawFeatures.shape[0]
    return aoCumulative


def meanNS(processedFeatures):
    
    nsCumulative = 0
    for i in range(processedFeatures.shape[0]):   
        nsCumulative += processedFeatures[i,0]
        
    nsCumulative /= processedFeatures.shape[0]
    return nsCumulative

        
def getPixelFeatures(x, y, rawFeatures, widthImg, sunDir):
    
    ## NOT tested yet
    rawFeaturesIndex = (x // widthImg) * widthImg + (y % widthImg)
    rawFeaturesElement = rawFeatures[rawFeaturesIndex]
    
    processedFeaturesElement = fillProcessedFeatureElement(rawFeaturesElement, sunDir)
    
    return rawFeaturesElement, processedFeaturesElement
    
def plotLogRLogB(processedFeaturesShadow, processedFeaturesLit):
    
    plt.figure()
    plt.plot(processedFeaturesShadow[:,1],processedFeaturesShadow[:,2], 'bx')
    plt.plot(processedFeaturesLit[:,1], processedFeaturesLit[:,2], 'ro')
    plt.title("logR - logB shadows (blue) and lit (red)")
    

def rgb_to_logRlogB(rgbImg):
    rows, cols, _ = rgbImg.shape
    
    logR = np.zeros((rows, cols), np.float)
    logB = np.zeros((rows, cols), np.float)
    
    for x in range(0,rows):
        for y in range(0,cols):
            if  (rgbImg[x,y,1] != 0):
               
                r = float(rgbImg[x,y,0]) / float(255)
                g = float(rgbImg[x,y,1]) / float(255)
                b = float(rgbImg[x,y,2]) / float(255)
                
                #red in log space
                logR[x,y] = np.log(r / g)
                
                #blue in log space
                logB[x,y] = np.log(b / g)
                
    return logR, logB

def bgr_to_chR_chG(bgrImg):
    rows, cols, _ = bgrImg.shape
    
    chR = np.zeros((rows, cols), np.float)
    chG = np.zeros((rows, cols), np.float)
    chB = np.zeros((rows, cols), np.float)
    
    for x in range(0,rows):
        for y in range(0,cols):
            if  (bgrImg[x,y,1] != 0):
               
                r = float(bgrImg[x,y,2]) / float(255)
                g = float(bgrImg[x,y,1]) / float(255)
                b = float(bgrImg[x,y,0]) / float(255)
                
                # r chromaticity componen
                chR[x,y] = (r / (r+g+b))
                
                #g chromaticity component
                chG[x,y] = (g / (r+g+b))
                
                chB[x,y] = (b / (r+g+b))
                
    return chR, chB, chG



def loadColorImages(folder):
    colorBGR = cv2.imread(folder + "/color.png", 1)
    colorRGB = cv2.cvtColor(colorBGR, cv2.COLOR_BGR2RGB)
    
    shadowMask = cv2.imread(folder + "/shadowMask.png", 0)
    litMask = cv2.imread(folder + "/litMask.png", 0)
    
    colorRGB_float = colorRGB.astype(float)
    colorRGB_float = np.divide(colorRGB_float, 254.0)
    
    colorRGB_shadowMask = cv2.bitwise_and(colorRGB_float, colorRGB_float, mask = shadowMask)
    colorRGB_litMask = cv2.bitwise_and(colorRGB_float, colorRGB_float, mask = litMask)    
   
    return colorRGB_float, colorRGB_shadowMask, colorRGB_litMask




def loadNormalsImages(folder):
    normalsBGR = cv2.imread(folder + "/normals.png", 1)
    normalsRGB = cv2.cvtColor(normalsBGR, cv2.COLOR_BGR2RGB)
    
    shadowMask = cv2.imread(folder + "/shadowMask.png", 0)
    litMask = cv2.imread(folder + "/litMask.png", 0)
    
    
    normalsRGB_float = normalsRGB.astype(float)
    normalsRGB_float = np.divide(normalsRGB_float, 255.0)
    
    
    normalsRGB_float = np.multiply(normalsRGB_float, 2.0)
    normalsRGB_float = np.subtract(normalsRGB_float, 1)
    normalsRGB_float = np.clip(normalsRGB_float, 0, 1)
    
    normalsRGB_shadowMask = cv2.bitwise_and(normalsRGB_float, normalsRGB_float, mask = shadowMask)
    normalsRGB_litMask = cv2.bitwise_and(normalsRGB_float, normalsRGB_float, mask = litMask)    
    
    return normalsRGB_float, normalsRGB_shadowMask, normalsRGB_litMask

    
def loadDepthImages(folder):
    depth = cv2.imread(folder + "/depth.png", 0)
    
    shadowMask = cv2.imread(folder + "/shadowMask.png", 0)
    litMask = cv2.imread(folder + "/litMask.png", 0)
    
    depth_float = depth.astype(float)
    depth_float = np.divide(depth_float, 255.0)
    depth_shadowMask = cv2.bitwise_and(depth_float, depth_float, mask = shadowMask)
    depth_litMask = cv2.bitwise_and(depth_float, depth_float, mask = litMask)    
    
    return depth_float, depth_shadowMask, depth_litMask

def loadAoImages(folder):
    ao = cv2.imread(folder + "/ao.png", 0)
    
    shadowMask = cv2.imread(folder + "/shadowMask.png", 0)
    litMask = cv2.imread(folder + "/litMask.png", 0)
    
    ao_float = ao.astype(float)
    ao_float = np.divide(ao_float, 255.0)
    
    ao_shadowMask = cv2.bitwise_and(ao_float, ao_float, mask = shadowMask)
    ao_litMask = cv2.bitwise_and(ao_float, ao_float, mask = litMask)    
    
    return ao, ao_shadowMask, ao_litMask

def loadShadowMapImages(folder):
    shadowMap = cv2.imread(folder + "/shadowMap.png", 0)
    
    shadowMask = cv2.imread(folder + "/shadowMask.png", 0)
    litMask = cv2.imread(folder + "/litMask.png", 0)
    
    shadowMap_shadowMask = cv2.bitwise_and(shadowMap, shadowMap, mask = shadowMask)
    shadowMap_litMask = cv2.bitwise_and(shadowMap, shadowMap, mask = litMask)    
    
    
    return shadowMap, shadowMap_shadowMask, shadowMap_litMask



def loadSunDir(folder):
    sunDirFile = open(folder + "/sunDir.txt", 'r')
    sunDirFileLines = sunDirFile.readlines()
    sunDirString = sunDirFileLines[0]
    
    sunDirStringValues = sunDirString.split('?')
    sunDir = np.zeros((3), np.float)
    
    sunDir[0] = np.float(sunDirStringValues[0])
    sunDir[1] = np.float(sunDirStringValues[1])
    sunDir[2] = np.float(sunDirStringValues[2])
    
    return sunDir

def processFeatures(rawFeatures, sunDir):
# Raw Features:
    # 0: R
    # 1: G
    # 2: B
    # 3: normal X
    # 4: normal Y
    # 5: normal Z
    # 6: depth
    # 7: ao
    # 8: shadowMap
    
# Processed Features:
    # 0: dot(n,s) = n.s
    # 1: log(R/G) = logR
    # 2: log(B/G) = logB
    # 3: (R / R+G+B) = chR
    # 4: (G / R+G+B) = chG
    # 5: (B / R+G+B) = chB
    # 6: up direction
    processedFeatures = np.zeros((rawFeatures.shape[0], 7), np.float)
    
    for i in range(rawFeatures.shape[0]):
        processedFeatures[i] = fillProcessedFeatureElement(rawFeatures[i], sunDir)
        
    return processedFeatures

def fillProcessedFeatureElement(rawFeaturesElement, sunDir):
    
    processedFeaturesElement = np.zeros((7), np.float)
    processedFeaturesElement[0] = np.dot(sunDir, rawFeaturesElement[3:6])
    processedFeaturesElement[1] = rgb_to_logR(rawFeaturesElement[0:3])
    processedFeaturesElement[2] = rgb_to_logB(rawFeaturesElement[0:3])
    processedFeaturesElement[3] = rgb_to_chR(rawFeaturesElement[0:3])
    processedFeaturesElement[4] = rgb_to_chG(rawFeaturesElement[0:3])
    processedFeaturesElement[5] = rgb_to_chB(rawFeaturesElement[0:3])
    processedFeaturesElement[6] = 0
    
    return processedFeaturesElement

def rgb_to_logR(rgb):
    return np.log(rgb[0] / rgb[1])
     
def rgb_to_logB(rgb):
    return np.log(rgb[2] / rgb[1])

def rgb_to_chR(rgb):
    return (3 * rgb[0]) / (rgb[0] + rgb[1] + rgb[2])

def rgb_to_chG(rgb):
    return (3 * rgb[1]) / (rgb[0] + rgb[1] + rgb[2])

def rgb_to_chB(rgb):
    return (3 * rgb[2]) / (rgb[0] + rgb[1] + rgb[2])
    
def removeSaturated(rawFeatures):
    newFeatures =  np.zeros((rawFeatures.shape), np.float)
    indexNewFeatures = 0
    
    for i in range(rawFeatures.shape[0]):
        if not checkSaturatedRawFeature(rawFeatures[i]):
            newFeatures[indexNewFeatures] = rawFeatures[i]
            indexNewFeatures += 1
            
    newFeatures = newFeatures[0:indexNewFeatures, :]
    return newFeatures
   
def checkSaturatedRawFeature(rawFeature):
    
    # check if rgb was 255 or 0,
    # rgb are divided by 254 before this check, so 1 corrisponds to a 254 pixel value
    if(rawFeature[0] == 0 or rawFeature[0] > 1 or 
       rawFeature[1] == 0 or rawFeature[1] > 1 or 
       rawFeature[2] == 0 or rawFeature[2] > 1):
        return True
    
    # check if normal values are available
    if(rawFeature[3] == 0 and rawFeature[4] == 0 and rawFeature[5] == 0):
        return True
    if(rawFeature[6] == 0):
        return True
    
def makeArray3(image3channel):
    features1 = makeArray(image3channel[:,:,0])
    features2 = makeArray(image3channel[:,:,1])
    features3 = makeArray(image3channel[:,:,2])
    
    features = np.zeros((features1.shape[0],3))
    
    features[:,0] = features1
    features[:,1] = features2
    features[:,2] = features3
    
    return features
    
def makeArray(image1channel):
    array = image1channel.ravel()
    return array

def concatenateRawFeatures(colorRGB_array, normalsRGB_array, depth_array, ao_array, shadowMap_array):
    ## we create a matrix of features for each pixel, we will filter masked values later
    # we also need to get rid of saturated values
    # features indices:
    # 0: R
    # 1: G
    # 2: B
    # 3: normal X
    # 4: normal Y
    # 5: normal Z
    # 6: depth
    # 7: ao
    # 8: shadowMap
    rawFeatures = np.zeros((depth_array.shape[0], 9), np.float)
    
    for i in range(depth_array.shape[0]):
        rawFeatures[i,0] = colorRGB_array[i,0]
        rawFeatures[i,1] = colorRGB_array[i,1]
        rawFeatures[i,2] = colorRGB_array[i,2]
        rawFeatures[i,3] = normalsRGB_array[i,0]
        rawFeatures[i,4] = normalsRGB_array[i,1]
        rawFeatures[i,5] = normalsRGB_array[i,2]
        rawFeatures[i,6] = depth_array[i]
        rawFeatures[i,7] = ao_array[i]
        rawFeatures[i,8] = shadowMap_array[i]
        
    return rawFeatures
