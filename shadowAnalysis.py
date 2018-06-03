# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:21:08 2018

@author: Fulvio Bertolini
"""

import os
import shadowAnalysisTools as tools

        
    
sessionID = "bankpark"
frameNbr = 135
if(sessionID == ""):
    sessionID = input("Enter session ID: ")

sessionPath = "../Sessions/" + sessionID + "/"
if os.path.exists(sessionPath):
    if(frameNbr == 0):
        frameNbr = input("Enter number frame: ")
    folder = sessionPath + "singleFrames/" + str(frameNbr)
    
    
    colorRGB, colorRGB_shadowMask, colorRGB_litMask = tools.loadColorImages(folder)
    normalsRGB, normalsRGB_shadowMask, normalsRGB_litMask =  tools.loadNormalsImages(folder)
    depth, depth_shadowMask, depth_litMask = tools.loadDepthImages(folder)
    ao, ao_shadowMask, ao_litMask = tools.loadAoImages(folder)
    shadowMap, shadowMap_shadowMask, shadowMap_litMask = tools.loadShadowMapImages(folder)
    sunDir = tools.loadSunDir(folder)
    
#    
    # Raw features indices
    # 0: R
    # 1: G
    # 2: B
    # 3: normal X
    # 4: normal Y
    # 5: normal Z
    # 6: depth
    # 7: ao
    # 8: shadowMap
    rawFeatures = tools.concatenateRawFeatures(tools.makeArray3(colorRGB), 
                        											 tools.makeArray3(normalsRGB), 
                                               tools.makeArray(depth),
                                               tools.makeArray(ao), 
                    											     tools.makeArray(shadowMap))
    
    rawFeatures_shadowMask = tools.concatenateRawFeatures(tools.makeArray3(colorRGB_shadowMask), 
                                                          tools.makeArray3(normalsRGB_shadowMask),
                                                          tools.makeArray(depth_shadowMask), 
                                                          tools.makeArray(ao_shadowMask), 
                                                          tools.makeArray(shadowMap_shadowMask))
    
    rawFeatures_litMask = tools.concatenateRawFeatures(tools.makeArray3(colorRGB_litMask),
                                                       tools.makeArray3(normalsRGB_litMask),
                                                       tools.makeArray(depth_litMask),
                                                       tools.makeArray(ao_litMask), 
                                                       tools.makeArray(shadowMap_litMask))
    
    
    rawFeatures_filtered = tools.removeSaturated(rawFeatures)
    rawFeatures_shadowMask_filtered = tools.removeSaturated(rawFeatures_shadowMask)
    rawFeatures_litMask_filtered = tools.removeSaturated(rawFeatures_litMask)
    
    processedFeatures = tools.processFeatures(rawFeatures_filtered, sunDir)
    processedFeatures_shadowMask = tools.processFeatures(rawFeatures_shadowMask_filtered, sunDir)
    processedFeatures_litMask = tools.processFeatures(rawFeatures_litMask_filtered, sunDir)
    
    tools.plotLogRLogB(processedFeatures_shadowMask, processedFeatures_litMask)
    
  
#    gmmShadow = GaussianMixture(1).fit(shadowFeatures)
#    probShadow = gmmShadow.predict_proba(shadowFeatures)
#    meansShadow = gmmShadow.means_
#    covariancesShadow = gmmShadow.covariances_
#    sigmasShadow = np.sqrt(covariancesShadow)
#    
#    gmmLit = GaussianMixture(1).fit(litFeatures)
#    probLit = gmmLit.predict_proba(litFeatures)
#    meansLit = gmmLit.means_
#    covariancesLit = gmmLit.covariances_
#    sigmasLit = np.sqrt(covariancesLit)
#    
#    
#    plt.figure()
#    plt.plot(shadowFeatures[:,0],shadowFeatures[:,1], 'bx')
#    plt.plot(litFeatures[:,0], litFeatures[:,1], 'ro')
#    plt.plot(meansLit[0,0],meansLit[0,1], 'k*' )
#    plt.plot(meansShadow[0,0], meansShadow[0,1], 'g*')
#    
#    junkRGB = np.copy(colorRGB)
#    
#    for i in range(logR_litMask.shape[0]):
#        for j in range(logR_litMask.shape[0]):
#            if(logR_litMask[i,j] > 0):
#                junkRGB[i,j,:] = 255
#            if(logR_litMask[i,j] < 0):
#                junkRGB[i,j,:] = 0
#    
#    for i in range(logR_shadowMask.shape[0]):
#        for j in range(logR_shadowMask.shape[0]):
#            if(logR_shadowMask[i,j] > 0):
#                junkRGB[i,j,:] = 255
#            if(logR_shadowMask[i,j] < 0):
#                junkRGB[i,j,:] = 0
#                
#      
#    junkRGB2 = np.copy(colorRGB)
#        
#    for i in range(logB_litMask.shape[0]):
#        for j in range(logB_litMask.shape[0]):
#            if(logB_litMask[i,j] > 0):
#                junkRGB2[i,j,:] = 255
#            if(logB_litMask[i,j] < 0):
#                junkRGB2[i,j,:] = 0
#    
#    for i in range(logB_shadowMask.shape[0]):
#        for j in range(logB_shadowMask.shape[0]):
#            if(logB_shadowMask[i,j] > 0):
#                junkRGB2[i,j,:] = 255
#            if(logB_shadowMask[i,j] < 0):
#                junkRGB2[i,j,:] = 0
#                
#        
#    plt.figure()
#    plt.imshow(junkRGB2)
#    plt.title("white = logB > 0")
#    
#        
#    plt.figure()
#    plt.imshow(junkRGB)
#    plt.title("white = logR > 0")
#    
    
