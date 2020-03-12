"""
Created on Thu Mar 12 12:02:44 2020

@author: RileyBallachay

This script was created as a general use tool to copy files 
with parent directory name to a new folder as quickly as possible.
"""


# Import libraries
from shutil import copyfile
import os

# Define root directory, containing a few levels of subdirectories
rootdir = 'rootdir'
outputdir = 'outputdir'

# Walk through subdirectories, find all JPEG files withot blah.jpg 
in the name and copy over to output directory with parent name

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('.jpg') and 'blah.jpg' not in file:
            sourceName = os.path.join(subdir,file)
            destinationName = os.path.join(outputdir,(os.path.split(subdir)[-1] + ' ' + file))
            copyfile(sourceName,destinationName)
