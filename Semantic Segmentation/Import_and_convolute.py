#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 09:28:45 2020

@author: RileyBallachay
"""
"""

The import and preprocess script is meant for importing masks

and data, then running the data through convolutional filters. 

The regions that aren't contained within either mask are dropped from 

the dataframe. The masks were manually built in FIJI. 

"""
# Authors: Riley Ballachay <riley.ballachay@gmail.com>

# Most Recent Date: 2020-02-18

# Importing all the necessary libraries
import pandas as pd
import numpy as np
import cv2
import os
import timeit
from skimage.filters import (wiener,gaussian,median,sobel,scharr,
                             prewitt,roberts,gabor,meijering,
                             sato,frangi,hessian)
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from skimage.transform import rotate
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
    
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )

"""
Can add the following to logging.basicConfig to dump to file
    filename="test.log",

Sample Debugging statements

    self.name = name
    self.price = price
    logging.debug("Pizza created: {} (${})".format(self.name, self.price))
"""
    
    
class SemanticSegmentation:  

    
    def __init__(self,path):
        self.path = path
        self.maskpaths = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            if "Image" in dirpath:
                self.imagepath = dirpath
            if "Texture" in dirpath:
                self.maskpaths.append(dirpath)
        try:
            print("The Image Path is:\n",self.imagepath)
        except:
            print("The name of the folder for images needs to be 'Image', Please correct and retry")
        try:
            print("The Mask Paths Are:\n",self.maskpaths[:])
            nummasks = len(self.maskpaths)
            print("Total Number of Texture Classes: ", nummasks)
        except:
            print("The name of the folder for masks needs to contain 'Texture', Please correct and retry")
            
        self.imagefiles = [os.path.join(self.imagepath,f) for f in os.listdir(self.imagepath) if f.endswith('.jpg')]
        numImages = len(self.imagefiles)
        print("There are %i images" % numImages)
        
        maskPathMatrix = []
        for maskpath in self.maskpaths:
            fileList = [os.path.join(maskpath,f) for f in os.listdir(self.imagepath) if f.endswith('.jpg')]
            maskPathMatrix.append(fileList)
        
        maskPathMatrix = [list(x) for x in zip(*maskPathMatrix)]
        print(maskPathMatrix)
        
        self.allMasks=[]
        for maskList in maskPathMatrix:
            mask = maskList[0]
            currentMask = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
            currentMask = np.where(currentMask>5,1,0)
            maskStack = np.reshape(currentMask,currentMask.shape + (1,))
            for mask in maskList[1:]:
                maskVal = int(maskList.index(mask)) + 1
                currentMask = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
                currentMask = np.where(currentMask>5,maskVal,0)
                currentMask = np.reshape(currentMask,currentMask.shape + (1,))
                maskStack = np.concatenate((maskStack,currentMask),axis=2)
            finalMask = np.amax(maskStack,axis=2)
            self.allMasks.append(finalMask)
          
        maskRavel = self.allMasks[0].ravel()
        for mask in self.allMasks[1:]:
            maskRavel.extend(mask.ravel())
            
        self.imageDF,self.filterNames,self.imageShape = self.__get_image_df()
        length,width = self.imageDF.shape
        maskDF = np.zeros((length,width+1))
        maskDF[:,:-1] = self.imageDF
        maskDF[:,-1] = maskRavel
        
        nonZeroIndices = np.nonzero(maskRavel)
        
        self.labelledPixels = maskDF[nonZeroIndices]


    def returner(self,):
        return self.labelledPixels
            
    
    # This is a python implementation of the function
    # designed in Weka to highlight 'membrane-like'
    # features in the image. It was extremely effective 
    # in the Weka classifier, so I added it here
    
    def __membrane_projection(self,image):
    
        # Defining the convolutional kernel
        kernel = np.zeros((19,19))
        kernel[:,9] = 1/19   
    
        # Defining rotations for the kernel    
        rotations = np.arange(0,180,6).astype(np.uint8)
    
        # Iterating through the rotation angles 
        # and applying the rotated kernel to the
        # image using OpenCV  
        rotation=rotations[0]
        tempkernel = rotate(kernel,rotation)
        tempimage = cv2.filter2D(image,-1,tempkernel)[:, :, np.newaxis]
        imagestack = tempimage
        
        for rotation in rotations[1:]:
            tempkernel = rotate(kernel,rotation)
            tempimage = cv2.filter2D(image,-1,tempkernel)[:, :, np.newaxis]
            imagestack = np.append(imagestack,tempimage,axis=2)
    
        
    
        # The four different methods of Z-stacking the images 
        # from Weka documentation
        summed = np.sum(imagestack,axis=2)
        meaned = np.mean(imagestack,axis=2)
        stdved = np.std(imagestack,axis=2)
        medied = np.median(imagestack,axis=2)
        maxed = np.amax(imagestack,axis=2)
        mined = np.amin(imagestack,axis=2)    
    
    
        logging.debug("membrane_project: Highlight membrane-like features in image.")
    
        # returns each of these as an image
    
        return (summed,meaned,stdved,medied,maxed,mined)
    
    
    # This function returns the euclidean distance between an array 
    # of x and y values and the center of the image on both the
    # Alpha 2 unit and the rig
    
    def __euclidean_distance(self,x,y):
        
        x0 = np.ceil(len(x)/2)
        y0 = np.ceil(len(x)/2)
    
        distances = np.sqrt((x0-x)**2+(y0-y)**2)
    
        logging.debug("euclidian_distance: Return euclidian distance between an array & center of image.")
    
        return distances

    
    
    # Main function for processing images and returning 
    # a stack of images which are filtered using various
    # kernerls and methods
    
    def __process_imgfile(self,x):
    
        # A series of filters which are stacked as features for each pixel
        # These filters are in no way verified
        # More analysis on what features are important is essential
        # Read in the image 
        ori = cv2.imread(x,cv2.IMREAD_GRAYSCALE)
        imageShape = ori.shape
        
        # Blur the image as a preprocessing step
        #ori = cv2.GaussianBlur(ori,(37,37),cv2.BORDER_DEFAULT)
    
        # This portion of the code is intended to retrieve both the coordinates
        # of each pixel and the euclidean distance of each 
        value = ori.flatten()
        ycoords,xcoords = np.indices(ori.shape).reshape(-1,len(value))
        euclidean = self.__euclidean_distance(xcoords,ycoords)
    
        # Create list and gaussian filter specifics
        GAUSSIAN = []
        gaussianwidths = np.arange(3,39,2,dtype=np.uint8)
    
        # Iterate through specified gaussian filter dimensions
        for width in gaussianwidths:
            gauss = cv2.GaussianBlur(ori,(width,width),cv2.BORDER_DEFAULT)
            GAUSSIAN.append(gauss)
        
        # Blur the image prior to Laplace operation
        # then equalize histogram after
        blur = cv2.GaussianBlur(ori,(5,5),0)
        laplace = cv2.Laplacian(blur,cv2.CV_8U)
        laplace = cv2.equalizeHist(laplace)
    
        # Run sobel filter with 4 different kernel size
        # in both the x and y direction
        sobel_x1 = cv2.Sobel(ori,cv2.CV_8U, 1, 0,ksize=1)
        sobel_y1 = cv2.Sobel(ori,cv2.CV_8U, 0, 1,ksize=1)
        sobel_x3 = cv2.Sobel(ori,cv2.CV_8U, 1, 0,ksize=3)
        sobel_y3 = cv2.Sobel(ori,cv2.CV_8U, 0, 1,ksize=3)
        sobel_x5 = cv2.Sobel(ori,cv2.CV_8U, 1, 0,ksize=5)
        sobel_y5 = cv2.Sobel(ori,cv2.CV_8U, 0, 1,ksize=5)
        sobel_x7 = cv2.Sobel(ori,cv2.CV_8U, 1, 0,ksize=7)
        sobel_y7 = cv2.Sobel(ori,cv2.CV_8U, 0, 1,ksize=7)
    
        # Canny edge detection filter
        canny = cv2.Canny(blur,50,100)
    
        # Erode the image with different kernel sizes
        kernel11 = np.ones((11,11),np.uint8)
        eroded_11 = cv2.erode(ori,kernel11,iterations = 2)
        kernel5 = np.ones((5,5),np.uint8)
        eroded_5 = cv2.erode(ori,kernel5,iterations = 2)
        kernel3 = np.ones((3,3),np.uint8)
        eroded_3 = cv2.erode(ori,kernel3,iterations = 2)
    
        # Defining blurred image and kernel for adaptive thresholding
        blur = cv2.GaussianBlur(ori,(5,5),0)  
        kernel = np.ones((5,5),np.uint8)
    
        # Perform adaptive thresholding
        adapthresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
        adapthresh = cv2.morphologyEx(adapthresh, cv2.MORPH_OPEN,kernel)
    
        # Perform gradient analysis on original image
        gradient = cv2.morphologyEx(ori, cv2.MORPH_GRADIENT, kernel)
    
        # Perform otsu thresholding 
        ret,otsuthresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
        # Median, scharr, prewitt, gabor, meijering,sato,frangi,hessian and roberts filters
        # all from skimage
        median_ = np.array(median(ori))
        scharr_ = np.array(scharr(ori))
        prewitt_ = np.array(prewitt(ori))
        gabor_,_ = np.array(gabor(ori,frequency=20))
        meijering_ = np.array(meijering(ori))
        sato_ = np.array(sato(ori))
        frangi_ = np.array(frangi(ori))
        hessian_ = np.array(hessian(ori))
        roberts_ = np.array(roberts(ori))
    
        # Calls the membrane_projection function
        Memprojs = self.__membrane_projection(ori)
    
        # Creates a list of images to be returned by the function (this has to be 
        # in the same order as the filternames)
        filteredimages = [ori, GAUSSIAN[0],GAUSSIAN[1],GAUSSIAN[2],GAUSSIAN[3],
              GAUSSIAN[4],GAUSSIAN[5],GAUSSIAN[6],GAUSSIAN[7],GAUSSIAN[8],
              GAUSSIAN[9],GAUSSIAN[10],GAUSSIAN[11],GAUSSIAN[12],GAUSSIAN[13],
              GAUSSIAN[14],GAUSSIAN[15],GAUSSIAN[16],GAUSSIAN[17],laplace,
              sobel_x1,sobel_y1,sobel_x3,sobel_y3,sobel_x5,sobel_y5,sobel_x7,
              sobel_y7,canny,eroded_11,eroded_5,eroded_3,adapthresh,otsuthresh,
              gradient,median_,scharr_,prewitt_,roberts_,
              gabor_,meijering_,sato_,frangi_,hessian_,Memprojs[0],Memprojs[1],
              Memprojs[2],Memprojs[3],Memprojs[4],Memprojs[5]]
    
        # The filternames which correspond to each of the filtered images from above          
        filternames = ['ori', 'gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7',
                       'gaus8','gaus9','gaus10','gaus11','gaus12','gaus13','gaus14','gaus15',
                       'gaus16','gaus17','gaus18','laplace','sobel_x1','sobel_y1',
              'sobel_x3','sobel_y3','sobel_x5','sobel_y5','sobel_x7','sobel_y7','canny',
              'eroded_11','eroded_5','eroded_3','adapthresh','otsuthresh','gradient','meadian',
              'scharr','prewitt','roberts','gabor','meijering',
              'sato','frangi','hessian','MP Add','MP Mean','MP STDev','MP Med','MP Max',
              'MP Min']
    
        logging.debug("process_imgfile: Process images and return stack of filtered image.")    
    
        return filteredimages,filternames,imageShape
    
      
    
    # The path and unit of the images in the path are passed to this function   
    # and the filtered images as flattened 1-D arrays, the names of the filters
    # the paths of all the images in the path and subdirectories are returned
    def __get_image_df(self):
        
        # Process all the image files and append to one master dataframe
        imagestack,filternames,imageShape = self.__process_imgfile(self.imagefiles[0])
        tempDF = np.zeros((imagestack[0].size,len(imagestack)))
        for i in range(0,len(imagestack)):
                tempDF[:,i] = imagestack[i].ravel()
        
        imageDF = tempDF

        for path in self.imagefiles[1:]:
            image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            imagestack,_ = self.__process_imgfile(image)
            tempDF = np.zeros((image.size,len(imagestack)))
            for i in range(0,len(imagestack)):
                tempDF[:,i] = imagestack[i].ravel()
    

            imageDF = np.append(imageDF,tempDF,axis=0)
    
    
        logging.debug("getimagedf: Return names of filters and image paths.")    
    
        return imageDF,filternames,imageShape
    
    
    def fit(self,model=DecisionTreeClassifier()):
        self.filterNames.append('class')
        imagePD = pd.DataFrame(self.labelledPixels,columns=self.filterNames)
        X = imagePD.drop('class',axis=1)
        Y = imagePD['class']
        
        kfold = KFold(n_splits=4,shuffle=True)
        scores = []
        
        # Splitting data into training and testing sets
        for train, test in kfold.split(X,Y):
            X_train = X.iloc[train]
            Y_train = Y.iloc[train]
            
            X_test = np.array(X.iloc[test])
            Y_test = np.array(Y.iloc[test])
            
            length,width = np.shape(X_test)
            # Creating an instance of the model
            # You can use pruning to decrease the complexity of the tree and increase training speed
            startime = timeit.default_timer()
            self.model.fit(X_train,Y_train)
            #importances['Weight']=model.feature_importances_
            elapsed = timeit.default_timer() - startime
            print("%.2f seconds elapsed\n" % elapsed)
            
            # Predict the output of the testing dataset
            predicted_labels = model.predict(X_test)
            
            # Printing out the confusion matrix and score for the labels
            print(confusion_matrix(Y_test,predicted_labels))
            print(f1_score(Y_test,predicted_labels, average='macro'))
            scores.append(f1_score(Y_test,predicted_labels, average='macro'))
        
        print("The average score for 5 folds is: ")
        print(np.average(scores))
        
        return
    
    def predict(self,model=ExtraTreesClassifier()):
        imagePD = pd.DataFrame(self.labelledPixels,columns=self.filterNames+['class'])
        X = imagePD.drop('class',axis=1)
        Y = imagePD['class']
        
        model.fit(X,Y)
        
        predictions = model.predict(self.imageDF)
        
        length,width = self.imageShape
        runLength = length*width
        
        masks = []
        for i in range(0, len(predictions), runLength):
            predictedMask = predictions[i:runLength].reshape(length,width)
            plt.figure(dpi=120)
            imgplot = plt.imshow(predictedMask)
            plt.axis('off')
            plt.show()  
            masks.append(predictedMask)
            
        return masks






