# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:21:08 2018

@author: Fulvio Bertolini
"""

import cv2
import os
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import matplotlib.mlab as mlab
import scipy.signal as signal

def draw_ellipse(position, covariance, ax=None, **kwargs): 
    """Draw an ellipse with a given position and covariance""" 
    ax = ax or plt.gca()
        
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0])) 
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
        # Draw the Ellipse
    
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
    


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
    print("max values in logR and logB: ", np.max(logR), " ", np.max(logB))
    print("min values in logR and logB: ", np.min(logR), " ", np.min(logB))
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



def loadImages(folder):
    colorBGR = cv2.imread(folder + "/color.png", 1)
    colorRGB = cv2.cvtColor(colorBGR, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(folder + "/depth.png", 0)
    normalsBGR = cv2.imread(folder + "/normals.png", 1)
    normalsRGB = cv2.cvtColor(normalsBGR, cv2.COLOR_BGR2RGB)
    ao = cv2.imread(folder + "/ao.png", 0)
    shadowMap = cv2.imread(folder + "/shadowMap.png", 0)
    shadowMask = cv2.imread(folder + "/shadowMask.png", 0)
    litMask = cv2.imread(folder + "/litMask.png", 0)
    
    
    colorRGB_shadowMask = cv2.bitwise_and(colorRGB,colorRGB,mask = shadowMask)
    colorRGB_litMask = cv2.bitwise_and(colorRGB,colorRGB,mask = litMask)    
    
    depth_shadowMask = cv2.bitwise_and(depth,depth,mask = shadowMask)
    depth_litMask = cv2.bitwise_and(depth,depth,mask = litMask)    
    
    normalsRGB_shadowMask = cv2.bitwise_and(normalsRGB,normalsRGB,mask = shadowMask)
    normalsRGB_litMask = cv2.bitwise_and(normalsRGB,normalsRGB,mask = litMask)    
    
    ao_shadowMask = cv2.bitwise_and(ao,ao,mask = shadowMask)
    ao_litMask = cv2.bitwise_and(ao,ao,mask = litMask)    
    
#    shadowMap_shadowMask = cv2.bitwise_and(shadowMap,shadowMap,mask = shadowMask)
#    shadowMap_litMask = cv2.bitwise_and(shadowMap,shadowMap,mask = litMask)    
    chR, chG, chB = bgr_to_chR_chG(colorBGR)
    chR_shadowMask = cv2.bitwise_and(chR,chR,mask = shadowMask)
    chR_litMask = cv2.bitwise_and(chR,chR,mask = litMask)
    
    chG_shadowMask = cv2.bitwise_and(chG,chG,mask = shadowMask)
    chG_litMask = cv2.bitwise_and(chG,chG,mask = litMask)
    
    chB_shadowMask = cv2.bitwise_and(chB,chB,mask = shadowMask)
    chB_litMask = cv2.bitwise_and(chB,chB,mask = litMask)
    return   colorRGB, colorRGB_shadowMask, colorRGB_litMask,  depth_shadowMask, depth_litMask, normalsRGB_shadowMask, normalsRGB_litMask, ao_shadowMask, ao_litMask, chR_shadowMask, chR_litMask, chG_shadowMask, chG_litMask, chB_shadowMask, chB_litMask
    



sessionID = input("Enter session ID: ")
sessionPath = "../Sessions/" + sessionID + "/"
if os.path.exists(sessionPath):
    frameNbr = input("Enter number frame: ")
    folder = sessionPath + "singleFrames/" + str(frameNbr)
    
    
    colorRGB, colorRGB_shadowMask, colorRGB_litMask, depth_shadowMask, depth_litMask, normalsRGB_shadowMask, normalsRGB_litMask,  ao_shadowMask, ao_litMask, chR_shadowMask, chR_litMask, chG_shadowMask, chG_litMask, chB_shadowMask, chB_litMask = loadImages(folder)
    
    logR_shadowMask, logB_shadowMask = rgb_to_logRlogB(colorRGB_shadowMask)
    logR_litMask, logB_litMask = rgb_to_logRlogB(colorRGB_litMask)
    
#    plt.figure()
#    plt.plot(logR_shadowMask.ravel(), logB_shadowMask.ravel(), 'o')
#    plt.plot(logR_litMask.ravel(), logB_litMask.ravel(), 'x')
#    plt.figure()
#    plt.plot(chR_shadowMask.ravel(), chB_shadowMask.ravel(), 'o')
#    plt.plot(chR_litMask.ravel(), chB_litMask.ravel(), 'x')
#
#    
    logR_litMask_array = logR_litMask.ravel()
    logB_litMask_array = logB_litMask.ravel()
    logR_shadowMask_array = logR_shadowMask.ravel()
    logB_shadowMask_array = logB_shadowMask.ravel()
    
    
    litIndices =  np.nonzero(logR_litMask_array)
    shadowIndices =  np.nonzero(logR_shadowMask_array)
    
    litFeatures = np.zeros((litIndices[0].shape[0],2), np.float)
    shadowFeatures = np.zeros((shadowIndices[0].shape[0],2), np.float)
    
    for i in range(litIndices[0].shape[0]):
        litFeatures[i,0] = logR_litMask_array[litIndices[0][i]]
        litFeatures[i,1] = logB_litMask_array[litIndices[0][i]]
        
    for i in range(shadowIndices[0].shape[0]):
        shadowFeatures[i,0] = logR_shadowMask_array[shadowIndices[0][i]]
        shadowFeatures[i,1] = logB_shadowMask_array[shadowIndices[0][i]]

#    print(logR_litMask_array.shape)
#    for i in range(logR_litMask.shape[0]):
#        for j in range(logR_litMask.shape[1]):
#            if(logR_litMask[i,j] != 0 and logB_litMask[i,j] != 0):
#                
#                
    
#    shadowFeatures[:,0] = logR_shadowMask.ravel()
#    shadowFeatures[:,1] = logB_shadowMask.ravel()
#    litFeatures[:,0] = logR_litMask.ravel()
#    litFeatures[:,1] = logB_litMask.ravel()
#    
    gmmShadow = GaussianMixture(1).fit(shadowFeatures)
    probShadow = gmmShadow.predict_proba(shadowFeatures)
    meansShadow = gmmShadow.means_
    covariancesShadow = gmmShadow.covariances_
    sigmasShadow = np.sqrt(covariancesShadow)
    
    gmmLit = GaussianMixture(1).fit(litFeatures)
    probLit = gmmLit.predict_proba(litFeatures)
    meansLit = gmmLit.means_
    covariancesLit = gmmLit.covariances_
    sigmasLit = np.sqrt(covariancesLit)
    
    
    plt.figure()
    plt.plot(shadowFeatures[:,0],shadowFeatures[:,1], 'bx')
    plt.plot(litFeatures[:,0], litFeatures[:,1], 'ro')
    plt.plot(meansLit[0,0],meansLit[0,1], 'k*' )
    plt.plot(meansShadow[0,0], meansShadow[0,1], 'g*')
    
    junkRGB = np.copy(colorRGB)
    
    for i in range(logR_litMask.shape[0]):
        for j in range(logR_litMask.shape[0]):
            if(logR_litMask[i,j] > 0):
                junkRGB[i,j,:] = 255
            if(logR_litMask[i,j] < 0):
                junkRGB[i,j,:] = 0
    
    for i in range(logR_shadowMask.shape[0]):
        for j in range(logR_shadowMask.shape[0]):
            if(logR_shadowMask[i,j] > 0):
                junkRGB[i,j,:] = 255
            if(logR_shadowMask[i,j] < 0):
                junkRGB[i,j,:] = 0
                
      
    junkRGB2 = np.copy(colorRGB)
        
    for i in range(logB_litMask.shape[0]):
        for j in range(logB_litMask.shape[0]):
            if(logB_litMask[i,j] > 0):
                junkRGB2[i,j,:] = 255
            if(logB_litMask[i,j] < 0):
                junkRGB2[i,j,:] = 0
    
    for i in range(logB_shadowMask.shape[0]):
        for j in range(logB_shadowMask.shape[0]):
            if(logB_shadowMask[i,j] > 0):
                junkRGB2[i,j,:] = 255
            if(logB_shadowMask[i,j] < 0):
                junkRGB2[i,j,:] = 0
                
        
    plt.figure()
    plt.imshow(junkRGB2)
    plt.title("white = logB > 0")
    
        
    plt.figure()
    plt.imshow(junkRGB)
    plt.title("white = logR > 0")
    
    
#    draw_ellipse(meansShadow[1], covariancesShadow)
#    
#    
#    
#    
#    plt.figure()
#    plt.imshow(colorRGB_shadowMask)
#    plt.title("colorRGB_shadowMask")
#    
#    plt.figure()
#    plt.imshow(colorRGB_litMask)
#    plt.title("colorRGB_litMask")
#    
#    plt.figure()
#    plt.imshow(depth_shadowMask, cmap='gray')
#    plt.title("depth_shadowMask")
#   
#    plt.figure()
#    plt.imshow(depth_litMask, cmap='gray')
#    plt.title("depth_litMask")
#    
#    plt.figure()
#    plt.imshow(normalsRGB_shadowMask)
#    plt.title("normalsRGB_shadowMask")
#    
#
#    plt.figure()
#    plt.imshow(normalsRGB_litMask)
#    plt.title("normalsRGB_litMask")
#    
#    
#    
#    plt.figure()
#    plt.imshow(ao_shadowMask, cmap='gray')
#    plt.title("ao_shadowMask")
#        
#    plt.figure()
#    plt.imshow(ao_litMask, cmap='gray')
#    plt.title("ao_litMask")
#    
#    
#    
#    plt.figure()
#    plt.imshow(chR_shadowMask)
#    plt.title("chR_shadowMask")
#    
#    
#    
#    
#    plt.figure()
#    plt.imshow(chR_litMask)
#    plt.title("chR_litMask")
#    
#    
#    
#    
#    plt.figure()
#    plt.imshow(chG_shadowMask)
#    plt.title("chG_shadowMask")
#    
#    plt.figure()
#    plt.imshow(chG_litMask)
#    plt.title("chG_litMask")
#    
#    
#    
#    
#    plt.figure()
#    plt.imshow(chB_shadowMask)
#    plt.title("chB_shadowMask")
#    
#    plt.figure()
#    plt.imshow(chB_litMask)
#    plt.title("chB_litMask")
#    
    
#   
#    plt.figure()
#    plt.hist(chR_shadowMask.ravel(),255,[1,256])
#    plt.title("rosso shadow")
#    plt.figure()
#    plt.hist(chG_shadowMask.ravel(),255,[1,256])
#    plt.title("verde shadow")
#    plt.figure()
#    plt.hist(chB_shadowMask.ravel(),255,[1,256])
#    plt.title("blu shadow")
#    
#    plt.figure()
#    plt.hist(chR_litMask.ravel(),255,[1,256])
#    plt.title("rosso lit")
#    plt.figure()
#    plt.hist(chG_litMask.ravel(),255,[1,256])
#    plt.title("verde lit")
#    plt.figure()
#    plt.hist(chB_litMask.ravel(),255,[1,256])
#    plt.title("blu lit")
    
#    plt.figure()
#    plt.plot(chR_litMask.ravel(), chB_litMask.ravel(), 'o')
#    plt.plot(chR_shadowMask.ravel(), chB_shadowMask.ravel(), 'x')
#    
#
#    plt.figure()
#    plt.plot(np.divide(chR_litMask.ravel(), chG_litMask.ravel()), np.divide(chB_litMask.ravel(), chG_litMask.ravel()), 'o')
#    plt.plot(np.divide(chR_shadowMask.ravel(), chG_shadowMask.ravel()), np.divide(chB_shadowMask.ravel(), chG_shadowMask.ravel()), 'x')
#    
#    
    
    
    