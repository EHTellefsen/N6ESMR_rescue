"""
Takes all N6ESMR data images (DI) in source directory and alligns them to a predefined
set of reference points using the left axis line within each image.
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
from skimage import io, img_as_float, img_as_ubyte
from skimage.filters import gaussian
from tqdm import tqdm

# -- Proprietary modules -- #
import projectFunctions as pf



#######################################################################
#User Inputs
#######################################################################

positive = True  #Choose whether negative of positive TIFs are to be processed 



#######################################################################
#Code
#######################################################################

# defining directory for separated DI and save directory alligned DI
if positive:
    source_dir = "./outputs/positives/separated"
    save_dir   = "./outputs/positives/alligned"   
    err_report_name = "allignDI_positives.err"
else:
    source_dir = "./outputs/negatives/separated"
    save_dir   = "./outputs/negatives/alligned"
    err_report_name = "allignDI_negatives.err"


# making list of images and define error report in case of processing error
files = glob(source_dir + '/*.png')
err_report = ''

#defining reference points for registration
tp_points = np.array([[218,339],[218,1560]])


for file in tqdm(files):
    filename = os.path.basename(file)

    try:
        #read image, convert to float and remove edges
        img = io.imread(file)
        img = img_as_float(img[:,20:-20])

        #sampling 100 points from left side of DI
        points = pf.significant_peaks(img)

        #Filtering away points not on axis line using Heirachical clustering
        points_filtered = pf.remove_outliers(points,maxDistance=5,scaleFactor=30)

        #Computing OLS for remaining points
        w = pf.linear_fit(points_filtered)

        #Using OLS line to find axis line tips
        top,bottom=pf.line_nodes(gaussian(img,2), points_filtered, w)

        #Using tips to allign images
        img_out=pf.tfm_to_template(img,np.array([top,bottom]),tp_points)
        
        # Limiting DI to 2000x2000, converting DI to Uint8 and saving
        img_out = img_as_ubyte(img_out[:2000,:2000])
        io.imsave(save_dir + "/" + filename, img_out)

    except:
        #Recording errors
        err_report = err_report + filename + "\n"
        pass

#writing error report
text_file = open('./reports/' + err_report_name, "w")
text_file.write(err_report)
text_file.close()