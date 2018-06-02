# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:21:08 2018

@author: Fulvio Bertolini
"""

import cv2
import os
import numpy as np
import shutil


def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,isShadowMask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if isShadowMask == True:
                cv2.circle(mask,(x,y),10,(255,0,0),-1)
            else:
                cv2.circle(mask,(x,y),10,(0,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if isShadowMask == True:
            cv2.circle(mask,(x,y),10,(255,0,0),-1)
        else:
            cv2.circle(mask,(x,y),10,(0,255,255),-1)

sessionID = input("Enter session ID: ")
sessionPath = "../Sessions/" + sessionID + "/"
if os.path.exists(sessionPath):
    frameNbr = input("Enter number frame: ")
    imagesFolder = sessionPath + "images/"
 
    colorPath = imagesFolder + "color/rgb_" + str(frameNbr) + ".png"
    depthPath = imagesFolder + "depth/depth_" + str(frameNbr) + ".png"
    normalPath = imagesFolder + "normals/normals_" + str(frameNbr) + ".png"
    shadowPath = imagesFolder + "shadows/shadows_" + str(frameNbr) + ".png"
    sunDirPath = sessionPath + "sunDirection/sunDirection_" + str(frameNbr) + ".txt"
    
    bgrImg = cv2.imread(colorPath, 1)
    depthImg = cv2.imread(depthPath, 0)
    normalImg = cv2.imread(normalPath, 1)
    shadowImg = cv2.imread(shadowPath, 1)
    
    sunDirFile = open(sunDirPath, "r")
    sunDirString = sunDirFile.readlines()
    sunDirFile.close()
    
    x, y, z = sunDirString[0].split('?')
    
    x = np.float(x)
    y = np.float(y)
    z = np.float(z)
    
    sunDir = np.zeros((3), np.float)
    sunDir[0] = x
    sunDir[1] = y
    sunDir[2] = z
    
    
    
    key = ''
    while key != 113:  # for 'q' key
        cv2.imshow("Color Image", bgrImg)
        cv2.imshow("Depth Image", depthImg)
        cv2.imshow("Normal Image", normalImg)
        cv2.imshow("Shadow Image", shadowImg)
        key = cv2.waitKey(5)
        
    cv2.destroyAllWindows()
    
    
    mask = cv2.imread(colorPath, 1)
    
    isShadowMask = True
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    
    while(1):
        cv2.imshow('image',mask)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            isShadowMask = not isShadowMask
        elif k == 13:
            break
    
    cv2.destroyAllWindows()
    
    shadowMask = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    litMask = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    yellow = np.array([0, 255, 255], np.uint8)
    blue = np.array([255, 0, 0], np.uint8)
    
    
    for i in range(mask.shape[0]):    
        for j in range(mask.shape[1]):
            if(np.array_equal(mask[i,j], yellow)):
                litMask[i,j] = 255
            if(np.array_equal(mask[i,j], blue)):
                shadowMask[i,j] = 255
                
    singleFrameFolderPath = sessionPath + "singleFrames/" + str(frameNbr)
    os.makedirs(singleFrameFolderPath)
    
    ambient_occlusion = shadowImg[:,:,0] 
    shadowMap = shadowImg[:,:,2]



    bgrImg_shadowMask = cv2.bitwise_and(bgrImg,bgrImg,mask = shadowMask)
    bgrImg_litMask = cv2.bitwise_and(bgrImg,bgrImg,mask = litMask) 
    
    depthImg_shadowMask = cv2.bitwise_and(depthImg, depthImg,mask = shadowMask)
    depthImg_litMask = cv2.bitwise_and(depthImg, depthImg,mask = litMask)   

    normalImg_shadowMask = cv2.bitwise_and(normalImg, normalImg,mask = shadowMask)
    normalImg_litMask = cv2.bitwise_and(normalImg, normalImg,mask = litMask)   
    
    ambient_occlusion_shadowMask = cv2.bitwise_and(ambient_occlusion, ambient_occlusion,mask = shadowMask)
    ambient_occlusion_litMask = cv2.bitwise_and(ambient_occlusion, ambient_occlusion,mask = litMask)   
    
    shadowMap_shadowMask = cv2.bitwise_and(shadowMap,shadowMap ,mask = shadowMask)
    shadowMap_litMask = cv2.bitwise_and(shadowMap, shadowMap,mask = litMask)   
    

    shutil.copy2(sunDirPath, singleFrameFolderPath + '/sunDir.txt')
    
    cv2.imwrite(singleFrameFolderPath + "/color.png", bgrImg)
    cv2.imwrite(singleFrameFolderPath + "/depth.png", depthImg)
    cv2.imwrite(singleFrameFolderPath + "/normals.png", normalImg)
    cv2.imwrite(singleFrameFolderPath + "/ao.png", ambient_occlusion)
    cv2.imwrite(singleFrameFolderPath + "/shadowMap.png", shadowMap)
    cv2.imwrite(singleFrameFolderPath + "/shadowMask.png", shadowMask)
    cv2.imwrite(singleFrameFolderPath + "/litMask.png", litMask)
    
    cv2.imwrite(singleFrameFolderPath + "/color_shadowMask.png", bgrImg_shadowMask)
    cv2.imwrite(singleFrameFolderPath + "/depth_shadowMask.png", depthImg_shadowMask)
    cv2.imwrite(singleFrameFolderPath + "/normals_shadowMask.png", normalImg_shadowMask)
    cv2.imwrite(singleFrameFolderPath + "/ao_shadowMask.png", ambient_occlusion_shadowMask)
    cv2.imwrite(singleFrameFolderPath + "/shadowMap_shadowMask.png", shadowMap_shadowMask)
    
    cv2.imwrite(singleFrameFolderPath + "/color_litMask.png", bgrImg_litMask)
    cv2.imwrite(singleFrameFolderPath + "/depth_litMask.png", depthImg_litMask)
    cv2.imwrite(singleFrameFolderPath + "/normals_litMask.png", normalImg_litMask)
    cv2.imwrite(singleFrameFolderPath + "/ao_litMask.png", ambient_occlusion_litMask)
    cv2.imwrite(singleFrameFolderPath + "/shadowMap_litMask.png", shadowMap_litMask)
    
    
else:
    print("Directory with session ID doesn't exist!")
    os.makedirs(sessionPath)
    
    zedFolder = sessionPath + "ZED/"
    iPhoneFolder = sessionPath + "iPhone/"
    imagesFolder = sessionPath + "images/"
    