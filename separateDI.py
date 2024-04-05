"""
Takes all N6ESMR TIF images in source directory and splits embedded
data images (DIs) in to separate png stored in save directory. User have
to specify whether positives or negatives are to be processed.
"""

# -- File info -- #
__author__ = 'Emil Haaber Tellefsen'
__contributors__ = ''
__contact__ = 's201201@dtu.dk'
__version__ = '1.2'
__date__ = '05-04-2024'

# -- Built-in modules -- #
import os
from glob import glob

# -- Third-part modules -- #
import numpy as np
from skimage import io
from tqdm import tqdm

# -- Proprietary modules -- #
import projectFunctions as pf



#######################################################################
#User Inputs
#######################################################################

positive = False  #Choose whether negative of positive TIFs are to be processed 



#######################################################################
#Code
#######################################################################

# defining directory for TIFs and save directory for DI
if positive:
    source_dir = "./data/positives"
    save_dir   = "./outputs/positives/separated"   
    err_report_name = "separateDI_positives.err"
else:
    source_dir = "./data/negatives"
    save_dir   = "./outputs/negatives/separated"
    err_report_name = "separateDI_negatives.err"


# making list of images and define error report in case of processing error
files = glob(source_dir + '/*.tif')
err_report = ''

for file in tqdm(files):
    filename = os.path.basename(file)

    try:
        #read image and remove edges
        img = io.imread(file)
        img = img[300:-150,150:-150]
        
        #invert image if positive
        if positive:
            img = 255-img

        # finding and removing upper and lower edge of DI
        img = pf.vertical_fit(img)

        #splitting DI
        spl_images = pf.split_images_gradline(img)

        #saving all DIs
        for j in range(len(spl_images)):
            io.imsave(save_dir + "/" + filename[:-4] + "_"+str(j+1)+".png", spl_images[j])
    
    except:
        #Recording unsuccesful splits
        err_report = err_report + filename + "\n"
        pass

#writing error report
text_file = open('./reports/' + err_report_name, "w")
text_file.write(err_report)
text_file.close()