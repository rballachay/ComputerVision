"""
Created on Tue Mar 17 09:06:02 2020

@author: RileyBallachay
"""

import glob

# Path of CSV Files to merge
interesting_files = glob.glob("Source Path/**/*.csv") 

# Don't save header for each imported CSV file
header_saved = False

# Loop over every CSV file in source path and merge into output CSV file
with open("Output.csv", "w") as fout:
    for filename in interesting_files:
        with open(filename) as fin:
            header = next(fin)
            if not 'Summary' in filename:
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
                
