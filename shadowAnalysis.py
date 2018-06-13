# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:21:08 2018

@author: Fulvio Bertolini
"""

import os
import matplotlib.pyplot as plt
import shadowAnalysisTools as tools

        
    
sessionID = "banktree"
frameNbr = 155
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
    # tools.savePlotLogRLogB(processedFeatures_shadowMask, processedFeatures_litMask, folder + "/logR-logB.png")
    plt.figure()
    plt.imshow(normalsRGB)
    plt.title("normalRGB")
    
    
    
    
    
    
    
    
    
    
    