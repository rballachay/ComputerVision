#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:55:19 2020

@author: RileyBallachay
"""
import cProfile
from Import_and_convolute import SemanticSegmentation

path = '/Volumes/Seagate Backup Plus Drive/FIJIPath'
sim = SemanticSegmentation(path)

pixels = sim.predict()


filternames = ['ori', 'gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7',
    
                       'gaus8','gaus9','gaus10','gaus11','gaus12','gaus13','gaus14','gaus15',
    
                       'gaus16','gaus17','gaus18','laplace','sobel_x1','sobel_y1',
    
              'sobel_x3','sobel_y3','sobel_x5','sobel_y5','sobel_x7','sobel_y7','canny',
    
              'eroded_11','eroded_5','eroded_3','adapthresh','otsuthresh','gradient','xcoords',
    
              'ycoords','euclidean','meadian','scharr','prewitt','roberts','gabor','meijering',
    
              'sato','frangi','hessian','MP Add','MP Mean','MP STDev','MP Med','MP Max',
    
              'MP Min']