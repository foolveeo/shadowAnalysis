# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:21:08 2018

@author: Fulvio Bertolini
"""

import os
import matplotlib.pyplot as plt
import shadowAnalysisTools as tools
import numpy as np


        
    
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
    ao = np.multiply(ao, 5)
    ao_shadowMask = np.multiply(ao_shadowMask, 5)
    ao_litMask = np.multiply(ao_litMask, 5)
    
    shadowMap, shadowMap_shadowMask, shadowMap_litMask = tools.loadShadowMapImages(folder)
    sunDir = tools.loadSunDir(folder)
    
    plt.figure()
    plt.imshow(ao)
    
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
    
    rgbShadow, rgbLit, nsShadow, nsLit, aoShadow, aoLit = tools.computeMeans(rawFeatures_shadowMask_filtered, rawFeatures_litMask_filtered, processedFeatures_shadowMask, processedFeatures_litMask)
    
    
     # my approach
    rbShadow = tools.rgb_to_chRchB(rgbShadow[0], rgbShadow[1], rgbShadow[2])
    rbLit = tools.rgb_to_chRchB(rgbLit[0], rgbLit[1], rgbLit[2])
    k_fulvio = tools.compute_k_rgb(rgbShadow, rgbLit, aoShadow, aoLit, nsLit)
    k_fulvio_g  = tools.compute_k_g(rgbShadow[1], rgbLit[1], aoShadow, aoLit, nsLit)
    
    sunSkyRatioR, sunSkyRatioB = tools.computeRedBlueRatio(rbShadow, rbLit, aoLit, nsLit, k_fulvio)
#    
    print("k fulvio", k_fulvio)
    print("k fulvio green", k_fulvio_g)
    
    print("redSunSkyRatio according to Fulvio: ", sunSkyRatioR)
    print("blueSunSkyRatio according to Fulvio: ", sunSkyRatioB)
    
    
    
    # Claus approach
    k = tools.compute_k(rawFeatures_filtered)
    C = np.divide(rgbShadow, rgbLit)
    ao_avg = tools.compute_ao_average(rawFeatures_shadowMask_filtered)
    Ea, Es = tools.computeIrradiances(C, aoShadow, aoLit, nsLit, nsShadow, k_fulvio, ao_avg, nsShadow)
    La, Ls = tools.computeRadiances(Ea, Es)
    
    
    sunColor = np.multiply( np.ones((100,100,3), np.float), Ls)
    skyColor = np.multiply( np.ones((100,100,3), np.float), La)
    
    
    sunColor = np.divide(sunColor, np.max(sunColor))
    skyColor = np.divide(skyColor, np.max(skyColor))
    plt.figure()
    plt.subplot(121)
    plt.imshow(skyColor)
    plt.subplot(122)
    plt.imshow(sunColor)
    
    print("Ea = ", Ea)
    print("Es = ", Es)
    print("La = ", La)
    print("Ls = ", Ls)
    print("k Claus", k)
    
    print("redSunSkyRatio according to Claus: ", (Es[1] / Ea[0]))
    print("blueSunSkyRatio according to Claus: ", (Es[2] / Ea[2]))
    
    
   
#    
#    Esky, ESun = tools.computeIrradiancesFulvio(k, sunSkyRatioR, sunSkyRatioB)
   
    
    
    