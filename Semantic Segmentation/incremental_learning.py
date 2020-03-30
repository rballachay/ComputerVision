#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 20:55:25 2020

@author: RileyBallachay
"""
import timeit
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from skimage.io import imread
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.tree import _tree
from sklearn.neural_network import MLPClassifier


# Image processing for the df
def process_imgfile(x):
  ori = cv2.imread(x,cv2.IMREAD_GRAYSCALE)
  ori = cv2.equalizeHist(ori)
  gaus_32 = cv2.GaussianBlur(ori,(31,31),cv2.BORDER_DEFAULT)
  gaus_16 = cv2.GaussianBlur(ori,(17,17),cv2.BORDER_DEFAULT)
  gaus_8 = cv2.GaussianBlur(ori,(9,9),cv2.BORDER_DEFAULT)
  gaus_2 = cv2.GaussianBlur(ori,(3,3),cv2.BORDER_DEFAULT)
  blur = cv2.GaussianBlur(ori,(3,3),0)
  laplace = cv2.Laplacian(blur,cv2.CV_8U )
  laplace = cv2.equalizeHist(laplace)
  sobel_x1 = cv2.Sobel(ori,cv2.CV_8U, 1, 0,ksize=1)
  sobel_y1 = cv2.Sobel(ori,cv2.CV_8U, 0, 1,ksize=1)
  sobel_x3 = cv2.Sobel(ori,cv2.CV_8U, 1, 0,ksize=3)
  sobel_y3 = cv2.Sobel(ori,cv2.CV_8U, 0, 1,ksize=3)
  sobel_x5 = cv2.Sobel(ori,cv2.CV_8U, 1, 0,ksize=5)
  sobel_y5 = cv2.Sobel(ori,cv2.CV_8U, 0, 1,ksize=5)
  sobel_x7 = cv2.Sobel(ori,cv2.CV_8U, 1, 0,ksize=7)
  sobel_y7 = cv2.Sobel(ori,cv2.CV_8U, 0, 1,ksize=7)
  canny = cv2.Canny(gaus_2,50,100)
  kernel11 = np.ones((11,11),np.uint8)
  eroded_11 = cv2.erode(ori,kernel11,iterations = 2)
  kernel5 = np.ones((5,5),np.uint8)
  eroded_5 = cv2.erode(ori,kernel5,iterations = 2)
  kernel3 = np.ones((3,3),np.uint8)
  eroded_3 = cv2.erode(ori,kernel3,iterations = 2)
  blur = cv2.GaussianBlur(ori,(5,5),0)
  adapthresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
  kernel = np.ones((5,5),np.uint8)
  adapthresh = cv2.morphologyEx(adapthresh, cv2.MORPH_OPEN,kernel)
  gradient = cv2.morphologyEx(ori, cv2.MORPH_GRADIENT, kernel)
  ret,otsuthresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  
  #img = cv2.resize(img, (64, 64))
  #img = str(img)
  return (ori, gaus_32,gaus_16,gaus_8,gaus_2,laplace,sobel_x1,sobel_y1,
          sobel_x3,sobel_y3,sobel_x5,sobel_y5,sobel_x7,sobel_y7,canny,
          eroded_11,eroded_5,eroded_3,adapthresh,otsuthresh,gradient)

def change(arrays):
    pixelcoords = np.zeros((2056,2464))
    for array in arrays:
        x,y,xlen,ylen = array
        for i in range(0,xlen):
            for j in range(0,ylen):
                pixelcoords[y+j,x+i]  = 1
    return pixelcoords

background_path = '/Users/RileyBallachay/Desktop/Image_2_colonies.jpg'
background_mask = cv2.imread(background_path,cv2.IMREAD_GRAYSCALE)
background_mask[background_mask>0] = 1

foreground_path = '/Users/RileyBallachay/Desktop/Image_2_background.jpg'
foreground_mask = cv2.imread(foreground_path,cv2.IMREAD_GRAYSCALE)
foreground_mask[foreground_mask>0] = 1

path = '/Users/RileyBallachay/Downloads/p1 end 20190702 pVS-178/Image_2.jpg'

filteredimages = process_imgfile(path)

length = len(filteredimages)
height = int(2056*2464)
#imagedata =  np.arange(0,2046*2464,2046*2464)
imagedata=np.zeros((height,length))
itercount=0
for image in filteredimages:
    flattened = np.concatenate(image).ravel()
    imagedata[:,itercount] = flattened
    itercount+=1
    
background_bool = background_mask.ravel()
background_bool = [1 if a_ > 0 else a_ for a_ in background_bool]
background_bool = np.reshape(background_bool,(int(2056*2464), 1))

foreground_bool = foreground_mask.ravel()
foreground_bool = [1 if a_ > 0 else a_ for a_ in foreground_bool]
foreground_bool = np.reshape(foreground_bool,(int(2056*2464),1))

imagedata = np.append(imagedata,background_bool,axis=1)
imagedata = np.append(imagedata,foreground_bool,axis=1)

columnTitles = ['ori', 'gaus_32','gaus_16','gaus_8','gaus_2','laplace','sobel_x1',
                'sobel_y1','sobel_x3','sobel_y3','sobel_x5','sobel_y5','sobel_x7','sobel_y7','canny',
'eroded_11','eroded_5','eroded_3','adapthresh','otsuthresh','gradient','background_bool','foreground_bool']

image_pd = pd.DataFrame(imagedata,columns=columnTitles)

image_pd = image_pd[(image_pd.background_bool!=0) | (image_pd.foreground_bool!=0)]

image_pd['class'] = image_pd.background_bool
image_pd = image_pd.drop('background_bool',axis=1)
image_pd = image_pd.drop('foreground_bool',axis=1)

X = image_pd.drop('class',axis=1)
Y = image_pd['class']


kfold = KFold(n_splits=5,shuffle=True)
scores = []

    
# Splitting data into training and testing sets
for train, test in kfold.split(X,Y):
    X_train = X.iloc[train]
    Y_train = Y.iloc[train]
    
    X_test = np.array(X.iloc[test])
    Y_test = np.array(Y.iloc[test])
    # Creating an instance of the model
    # You can use pruning to decrease the complexity of the tree and increase training speed
    startime = timeit.default_timer()
    model = RandomForestClassifier()
    model.fit(X_train,Y_train)
    elapsed = timeit.default_timer() - startime
    print("%.2f seconds elapsed\n" % elapsed)
    
    # Predict the output of the testing dataset
    predicted_labels = model.predict(X_test)
    
    # Dictionary to map categorical variables into numerical, for precision score
    d = {0:"Colony",1:"Background"}
    
    # Mapping using the dictionary
    predicted_labels = [d[x] for x in predicted_labels]
    target_labels = [d[x] for x in Y_test]
    
    # Printing out the confusion matrix and score for the labels
    print(confusion_matrix(target_labels,predicted_labels))
    print(metrics.precision_score(target_labels,predicted_labels, average='macro'))
    scores.append(metrics.precision_score(target_labels,predicted_labels, average='macro'))

print("The average score for 5 folds is: ")
print(np.average(scores))