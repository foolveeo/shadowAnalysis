 # -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:17:49 2018

@author: Fulvio Bertolini
"""
import os
import cv2
import numpy as np
import ntpath
import matplotlib.pyplot as plt

def rgb_to_chRchB(rgbImg):
    rows, cols, _ = rgbImg.shape
    
    chR = np.zeros((rows, cols), np.float)
    chB = np.zeros((rows, cols), np.float)
    
    for x in range(0,rows):
        for y in range(0,cols):
            if  (rgbImg[x,y,1] != 0):
               
                r = float(rgbImg[x,y,0]) / float(255)
                g = float(rgbImg[x,y,1]) / float(255)
                b = float(rgbImg[x,y,2]) / float(255)
                
                if(g == 0):
                    g = (1 / 255)
                
                #red in log space
                chR[x,y] = r / g
                chB[x,y] = b / g
                 
    return chR, chB



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

def bgr_to_logRlogB(bgrImg):
    rows, cols, _ = bgrImg.shape
    
    logR = np.zeros((rows, cols), np.float)
    logB = np.zeros((rows, cols), np.float)
    
    for x in range(0,rows):
        for y in range(0,cols):
            if  (bgrImg[x,y,1] != 0):
               
                r = float(bgrImg[x,y,2]) / float(255)
                g = float(bgrImg[x,y,1]) / float(255)
                b = float(bgrImg[x,y,1]) / float(255)
                
                #red in log space
                logR[x,y] = np.log(r / g)
                
                #blue in log space
                logB[x,y] = np.log(b / g)
    return logR, logB

def createFeatures(img, shadowMask, litMask):
    
    img_shadow = cv2.bitwise_and(img, img, mask=shadowMask)
    img_lit = cv2.bitwise_and(img, img, mask=litMask)


   
    


    shadowLogR, shadowLogB = rgb_to_chRchB(img_shadow)
    litLogR, litLogB = rgb_to_chRchB(img_lit)
    
    shadowLogR_array = shadowLogR.ravel()
    shadowLogB_array = shadowLogB.ravel()
    
    litLogR_array = litLogR.ravel()
    litLogB_array = litLogB.ravel()
    
    shadowIndices =  np.nonzero(shadowLogR_array)
    litIndices =  np.nonzero(litLogR_array)
    
    litFeatures = np.zeros((litIndices[0].shape[0],2), np.float)
    shadowFeatures = np.zeros((shadowIndices[0].shape[0],2), np.float)
    
    for i in range(litIndices[0].shape[0]):
        litFeatures[i,0] = litLogR_array[litIndices[0][i]]
        litFeatures[i,1] = litLogB_array[litIndices[0][i]]
        
    for i in range(shadowIndices[0].shape[0]):
        shadowFeatures[i,0] = shadowLogR_array[shadowIndices[0][i]]
        shadowFeatures[i,1] = shadowLogB_array[shadowIndices[0][i]]

    return (shadowFeatures, litFeatures)
    
    
def plot(redBalanced, greenBalanced, blueBalanced, whiteBalanced, shadowMask, litMask, label):
        
    redBalanceShadowFeatures, redBalanceLitFeatures = createFeatures(redBalanced, shadowMask, litMask)
    greenBalanceShadowFeatures, greenBalanceLitFeatures = createFeatures(greenBalanced, shadowMask, litMask)
    blueBalanceShadowFeatures, blueBalanceLitFeatures = createFeatures(blueBalanced, shadowMask, litMask)
    whiteBalanceShadowFeatures, whiteBalanceLitFeatures = createFeatures(whiteBalanced, shadowMask, litMask)
    
    
    plt.figure()
    plt.plot(redBalanceShadowFeatures[:,0],redBalanceShadowFeatures[:,1], 'rx')
    plt.plot(redBalanceLitFeatures[:,0], redBalanceLitFeatures[:,1], 'ro')
    plt.title("redBalance " + label)
    
    plt.figure()
    plt.plot(greenBalanceShadowFeatures[:,0],greenBalanceShadowFeatures[:,1], 'gx')
    plt.plot(greenBalanceLitFeatures[:,0], greenBalanceLitFeatures[:,1], 'go')
    plt.title("greenBalance " + label)
    
    plt.figure()
    plt.plot(blueBalanceShadowFeatures[:,0],blueBalanceShadowFeatures[:,1], 'bx')
    plt.plot(blueBalanceLitFeatures[:,0], blueBalanceLitFeatures[:,1], 'bo')
    plt.title("blueBalance " + label)
    
    plt.figure()
    plt.plot(whiteBalanceShadowFeatures[:,0],whiteBalanceShadowFeatures[:,1], 'kx')
    plt.plot(whiteBalanceLitFeatures[:,0], whiteBalanceLitFeatures[:,1], 'ko')
    plt.title("whiteBalance " + label)
    
    plt.figure()
    plt.plot(redBalanceShadowFeatures[:,0],redBalanceShadowFeatures[:,1], 'rx')
    plt.plot(redBalanceLitFeatures[:,0], redBalanceLitFeatures[:,1], 'ro')
    plt.plot(greenBalanceShadowFeatures[:,0],greenBalanceShadowFeatures[:,1], 'gx')
    plt.plot(greenBalanceLitFeatures[:,0], greenBalanceLitFeatures[:,1], 'go')
    plt.plot(blueBalanceShadowFeatures[:,0],blueBalanceShadowFeatures[:,1], 'bx')
    plt.plot(blueBalanceLitFeatures[:,0], blueBalanceLitFeatures[:,1], 'bo')
    plt.plot(whiteBalanceShadowFeatures[:,0],whiteBalanceShadowFeatures[:,1], 'kx')
    plt.plot(whiteBalanceLitFeatures[:,0], whiteBalanceLitFeatures[:,1], 'ko')
    plt.title(label)
    

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,isShadowMask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if isShadowMask == True:
                cv2.circle(mask,(x,y),25,(255,0,0),-1)
            else:
                cv2.circle(mask,(x,y),25,(0,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if isShadowMask == True:
            cv2.circle(mask,(x,y),25,(255,0,0),-1)
        else:
            cv2.circle(mask,(x,y),25,(0,255,255),-1)
            
            
            
photoFolder = "./camerasession/"
#os.makedirs(photoFolder + "masks")
#cv2.namedWindow('image')
#cv2.setMouseCallback('image',draw_circle)
#yellow = np.array([0, 255, 255], np.uint8)
#blue = np.array([255, 0, 0], np.uint8)
#drawing = False
#for filename in os.listdir(photoFolder):
#    img = cv2.imread(photoFolder + filename, 1)
#    mask = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
#    
#    isShadowMask = True
#    while(1):
#        cv2.imshow("image", mask)
#            
#        k = cv2.waitKey(1) & 0xFF
#        if k == ord('m'):
#            isShadowMask = not isShadowMask
#        elif k == 13:
#            break
#    shadowMask = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
#    litMask = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
#    for i in range(mask.shape[0]):    
#        for j in range(mask.shape[1]):
#            if(np.array_equal(mask[i,j], yellow)):
#                litMask[i,j] = 255
#            if(np.array_equal(mask[i,j], blue)):
#                shadowMask[i,j] = 255
#                
#    cv2.imwrite("./camerasession/masks/" + filename, mask)
#    
#    
#    
#yellow = np.array([0, 255, 255], np.uint8)
#blue = np.array([254, 0, 0], np.uint8)
# 
#files = os.listdir(photoFolder + "masks/")
#for filename in files:
#    img = cv2.imread(photoFolder + "masks/" + filename, 1)
#    shadowMask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
#    litMask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
#
#    
#    for i in range(img.shape[0]):    
#        for j in range(img.shape[1]):
#            if(np.array_equal(img[i,j], yellow)):
#                litMask[i,j] = 255
#            if(np.array_equal(img[i,j], blue)):
#                shadowMask[i,j] = 255
#    cv2.imwrite(photoFolder + "masks/lit_" + filename, litMask)
#    cv2.imwrite(photoFolder + "masks/shadow_" + filename, shadowMask)
#    
#    
##    
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

plot(redBalanced1, greenBalanced1, blueBalanced1, whiteBalanced1, shadowMask1, litMask1, "uno")
plot(redBalanced2, greenBalanced2, blueBalanced2, whiteBalanced2, shadowMask2, litMask2, "due")
plot(redBalanced3, greenBalanced3, blueBalanced3, whiteBalanced3, shadowMask3, litMask3, "tre")
                  