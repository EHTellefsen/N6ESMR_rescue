# -- File info -- #
__author__ = 'Emil Haaber Tellefsen'
__contributors__ = ''
__contact__ = 's201201@dtu.dk'
__version__ = '1.2'
__date__ = '05-04-2024'

# -- Built-in modules -- #
import warnings

# -- Third-part modules -- #
import numpy as np
import datetime

from skimage import img_as_ubyte, img_as_float
from skimage.filters import prewitt_h, prewitt_v
from skimage.measure import profile_line
from skimage.transform import SimilarityTransform, warp

from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
from scipy.interpolate import InterpolatedUnivariateSpline

import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#######################################################################
# misc
#######################################################################

def find_derivative(signal):
    """
    FIND_GRADIENT finds the gradient of a 1d signal using:
    df(i) = f(i+1) - f(i-1)

    usage: gradient = find_gradient(signal)

    Input   signal          1darray: array with signal
    Output  gradient        1darray: array with gradient

    Author: Emil Haaber Tellefsen, 2023
    """

    n = len(signal)
    ds=np.zeros(n)
    for i in range(1,n-1):
        ds[i]=signal[i+1]-signal[i-1]
    return ds

def second_derivative(signal):
    """
    SECOND_DERIVATIVE finds the second derivative of a 1d signal using:
    ddf(i) = f(i+2) - 2*f(i) + f(i-2)

    usage: ddf = second_derivative(signal)

    Input   signal          1darray: array with signal
    Output  ddf             1darray: array with second derivative

    Author: Emil Haaber Tellefsen, 2023
    """

    n = len(signal)
    ddf = np.zeros(n)
    for i in range(2,n-2):
        ddf[i] = signal[i+2] - 2*signal[i] + signal[i-2]
    return ddf

def find_first_peak(signal, location, significance=2, minThresh=0):
    """
    FIND_FIRST_PEAK finds the first or last significant peak in a signal, where "significance" 
    denotes numbers of standard deviations above signal mean. In cases where no significant 
    peaks can be found, NaN is returned.

    usage: peak = find_first_peak(signal, location)
           peak = find_first_peak(signal, location, significance)
           peak = find_first_peak(signal, location, significance, minThresh)

    Input   signal          1darray: array with signal
    Input   location        string: 'start' for first peak, 'end' for last peak
    Input   significance    scalar: number of standard deviations above mean the peak can be 
                            found at (default is 2)
    Input   minThresh       scalar: defines a minimum distance from mean peak needs to be at, 
                            to filter out noise in cases where standard deviation is low
    Output  peak            scalar: location of first peak

    Author: Emil Haaber Tellefsen, 2023
    """

    #finding mean and standard deviation of signal
    avg = np.mean(signal)
    std = np.std(signal)

    # start peak
    if location == 'start':
        i = 0
        while i < len(signal):
            # proceducere when significant level is found (25 is chosen arbitrarily)
            if signal[i] >= avg + significance*std and signal[i] >=avg + minThresh:
                # finding max of next 25 values
                loc = i+np.where(signal[i:i+25]==np.max(signal[i:i+25]))[0].squeeze()

                # finding unique maximum
                if np.isscalar(loc) == True:
                    return loc
                else:
                    return np.min(loc)
            i+=1
    # end peak
    elif location == 'end':
        i = len(signal)-1
        while i >= 0:
            # proceducere when significant level is found (25 is chosen arbitrarily)
            if signal[i] >= avg + significance*std and signal[i] >=avg + minThresh:
                # finding max of next 25 values
                loc = i-25+np.where(signal[i-25:i]==np.max(signal[i-25:i]))[0].squeeze()
                
                # finding unique maximum
                if np.isscalar(loc) == True:
                    return loc
                else:
                    return np.max(loc)
            i-=1
    
    return float('nan')

def getPix(multiIm, maskIm):
    """
    Extracting pixels from multispectral image
    
    function [clPix, row, col] = getPix(multiIm, maskIm) 
    
    Input
      multiIm - multispectral image of size r x c x l
      maskIm - binary mask with ones at pixels that should be extracted (use
          layers from annotationIm from the loadMulti function)
    
    Output
      clPix - pixels in n x l matrix
      row - row pixels indicies
      col - column pixels indicies
    
    Anders Nymark Christensen - DTU, 2018 01 30
    Adapted from
    Anders Lindbjerg Dahl - DTU, January 2013
    """
    
    nMask = maskIm.sum()
    
    r, c = np.where(maskIm == 1)
    
    clPix = np.zeros(int(nMask))
    clPix = multiIm[r,c]    
    
    return [clPix, r, c]



#######################################################################
# splitting image frames
#######################################################################

def vertical_fit(img):
    """
    VERTICAL_FIT finds the top and bottom of a polaroid in the NIMBUS 6 
    ESMR dataset, so that image fits to the top and bottom of the individual 
    frames. Function works using a horizontal prewitt filter. It is necessary 
    to remove significant edges before using this.

    usage: img_out = vertical_fit(img,polaroidType)

    Input   img             (:,:)ndarray image from NIMBUS 6 ESMR dataset
    Output  img_out         (:,:)ndarray image where top and bottom has been 
                            fit to frame

    Author: Emil Haaber Tellefsen, 2023
    """

    # calculating summed prewitt filter
    img_prewitt = prewitt_h(img)
    h_sum = np.sum(img_prewitt,axis=1)

    #finding peaks
    peaks_h = find_peaks(abs(h_sum),distance=500)[0]

    #find top and bottom depending on polaroidType
    top = np.min(peaks_h)
    bottom = np.max(peaks_h)
    
    #fitting image
    img = img[top:bottom,:]
    return img

def split_images_gradline(img):
    """
    SPLIT_IMAGES_GRADLINE seperates frames in NIMBUS 6 ESMR images that has 
    already been processed using VERTICAL_FIT. Function works through applying
    vertical prewitt filter to a band of 40 pixels starting 60 pixels above the
    bottom of image, and then using significant peaks as splitting parameters.
    Frames cut over in original images is ignored.

    usage: frames, startpeak, endpeak = split_images_gradline(img)

    Input   img             (:,:)ndarray: image from NIMBUS 6 ESMR dataset 
                            that has been vertically fit using VERTICAL_FIT.
    Output  frames          List of (:,:)ndarrays: images split to individual 
                            frames.

    Author: Emil Haaber Tellefsen, 2023
    """

    #finding dimelsions and level to compute gradient
    m=img.shape[1]
    section = img_as_float(img[(img.shape[0] - 100):(img.shape[0] - 60),:])
    section_prewitt = prewitt_v(section)
    v_sum=img_as_float(np.sum(section_prewitt,axis=0))

    #finding gradient peaks
    peaks = find_peaks(abs(v_sum),height=np.mean(abs(v_sum))+5*np.std(abs(v_sum)),distance=100)[0]
    peakType = np.sign(v_sum[peaks])

    #making list of frames and puts frames in whenever peakType matches the start of and image. Frames cut over is ignored
    outputImages=[]
    for i in range(peakType.size-1):
        if peakType[i]==1:
            sub_img = img[:,peaks[i]:peaks[i+1]]
            outputImages.append(sub_img)
    
    return outputImages



#######################################################################    
# Alligning frames
#######################################################################

def significant_peaks(img, order=2):
    """
    SIGNIFICANT_PEAKS samples 100 equally spaced rows in an image and finds the 
    first significant derivative peak in each, comming from the left. It can do 
    this using both first and second order derivatives. Only points where a
    significant peak is found is returned.

    usage: points =  significant_peaks(img)
           points =  significant_peaks(img,order) 

    Input   img          (:,:)ndarray: float image coresponding to extracted image in 
                         N6ESMR dataset.
            order        scalar (1 or 2): denotes the order of derivative used for 
                         finding peaks (default is 2).
    Output  points       (:,:2)ndarray: array of point coordinates of peaks found.

    Author: Emil Haaber Tellefsen, 2023
    """    

    n = np.shape(img)[0]
    lnum = 200
    direction='start'
    
    # Finding sample y-values using equal interval
    interval = int(n/lnum)
    yrange = np.linspace(interval, interval*lnum, lnum)

    # Allocating vector for found peak locations
    vpeaks=np.zeros(lnum)

    # Itterating through selected rows and locate first peak
    index = -1
    for i in range(lnum):
        index = index + interval 
        if order == 1:
            sig = find_derivative(img[index,:])
        elif order == 2:
            sig = second_derivative(img[index,:])

        vpeaks[i] = find_first_peak(sig,location=direction,significance=3,minThresh=0.1)
    
    # Removing results for row rows where no peak has been found
    yrange=yrange[np.isnan(vpeaks)==False]
    vpeaks=vpeaks[np.isnan(vpeaks)==False]

    # Converting results to (:,:2)ndarray
    points = np.vstack((vpeaks,yrange)).T

    return points

def remove_outliers(points, maxDistance, scaleFactor):
    """
    REMOVE_OUTLIERS removes points that does not fit the group of points on graph
    line. Function uses single linkage heirachical clustering, where relative 
    direction weight can be scaled. 

    usage: points_filtered =  remove_outliers(points, maxDistance, scaleFactor)

    Input  points            (:,:2)ndarray: array of point coordinates of peaks found
    Input  maxDistance       scalar: maximum euclidean distance points are allowed to 
                             be apart using single linkage and taking scaleFactor into 
                             account.
    Input  scaleFactor       scalar: value describing how much more weight should be 
                             applied to horizontal distance vs. vertical distance.
    Output points_filtered   (:,:2)ndarray: array of point coordinates where points 
                             not in graph line is filtered away.

    Author: Emil Haaber Tellefsen, 2023
    """    

    X=np.copy(points)

    # Deviding by scaleFactor to decrease importance of vertical distance
    X[:,1]=X[:,1]/scaleFactor

    # Defining linkage and calculating clusters satisfying maxDistance criteria
    Z = linkage(X, method='single', metric='euclidean')
    cls = fcluster(Z, criterion='distance', t=maxDistance)-1

    # Finding largest cluster
    counts = np.zeros(np.max(cls)+1)
    for i in cls:
        counts[i]+= 1
    largestClass = np.where(counts==np.max(counts))[0].squeeze()

    # removing all points not within largest cluster
    points_filtered = points[cls == largestClass,:]

    return points_filtered

def linear_fit(points):
    """
    LINEAR_FIT computes the coefficients for a linear fit 'f(x)= a + b*x', using normal 
    equation and coordinate points given.

    usage: w = linear_fit(points)

    Input  points      (:,:2)ndarray: array of point coordinates
    Output w           (2,)ndarray: coefficients for linear fit

    Author: Emil Haaber Tellefsen, 2023
    """

    X = np.vstack((np.ones(len(points[:,1])),points[:,1])).T
    XtX = X.T @ X
    Xty = X.T @ points[:,0]

    w = np.linalg.solve(XtX,Xty).squeeze()

    return w

def line_nodes(img, points, w):
    """
    LINE_NODES finds the beginning and end of a fitted line on an image, using first and last 
    points as preliminary ends, and a gradient transformation to calculate it exact.

    usage: top,bottom = line_nodes(img, points, w)

    Input  img      (:,:)ndarray: float image coresponding to frame in NIMBUS 6 ESMR dataset
    Input  points   (:,:2)ndarray: array of point coordinates
    Input  w        (2,)ndarray: coefficients for linear fit
    Output top      (2,)ndarray: coordinates for top peak
    Output bottom   (2,)ndarray: coordinates for top peak

    Author: Emil Haaber Tellefsen, 2023
    """

    # making prelimnary start and end of line using leftmost and rightmost point
    line_start = np.min(points[:,1])-60
    line_end = np.max(points[:,1])+60

    # making profile line between points using weights from linear fit
    pl = profile_line(img,[line_start, w[0] + line_start*w[1]],[line_end, w[0] + line_end*w[1]])
    
    # finding first and last significant peak of profile line
    sig = abs(find_derivative(pl))
    top = find_first_peak(sig,'start',2)
    bottom = find_first_peak(sig,'end',2)

    # finding coordinates from peaks
    top_tfm = line_start + top
    bottom_tfm = line_start + bottom
    top_x = w[0]+w[1]*top_tfm
    bottom_x = w[0]+w[1]*bottom_tfm
    return [int(top_x),int(top_tfm)],[int(bottom_x),int(bottom_tfm)]

def tfm_to_template(img, src_points, tp_points):
    """
    TFM_TO_TEMPLATE uses a similarity transform and some key points to warp an image 
    in order to match a template defined by coresponding reference points.

    usage: img_warped = tfm_to_template(img, src_points, tp_points)

    Input  img                  (:,:)ndarray: float image.
           src_points           (:,2)ndarray: key points in image.
           tp_points            (:,2)ndarray: corresponding keypoints in template image.
    Output img_warped           (:,:)ndarray: transformed float image.

    Author: Emil Haaber Tellefsen, 2023
    """ 

    tform = SimilarityTransform()

    tform.estimate(src_points,tp_points)
    img_warped = warp(img,tform.inverse)
    return img_warped



#######################################################################
# Light correction
#######################################################################

def BGMask():
    """
    BGMASK defines a mask that covers all sections in an N6ESMR image that given correct
    processing should not contain any data-information.

    usage: backgroundMask = BGMask()

    Output  backgroundMask  (:2000,:2000)ndarray: binary array at same size as extracted
                            images, covering guaranteed background pixels.

    Author: Emil Haaber Tellefsen, 2023
    """ 

    backgroundMask=np.zeros((2000,2000),dtype=int)
    backgroundMask[0:100,:]=1
    backgroundMask[260:325,:]=1
    backgroundMask[:,:100]=1
    backgroundMask[:,-100:]=1
    backgroundMask[-165:,:]=1
    backgroundMask[200:260,1220:]=1
    backgroundMask[200:260,:900]=1
    backgroundMask[1600:,:180]=1
    return backgroundMask

def ImproveBGMask(BGMask, nodesL, nodesR):
    """
    IMPROVEBGMASK uses information in N6ESMR image regarding location of ticks
    in order to find pieces of background not covered by text, thus improving 
    background mask.

    usage: BGMask_improved = ImproveBGMask(BGMask,nodesL,nodesR)

    Input   BGMask              (:2000,:2000)ndarray: binary array at same size as extracted
                                images, covering guaranteed background pixels.
            nodesL              (:,)ndarray: row number for ticks of left axis line
            nodesR              (:,)ndarray: row number for ticks of right axis line
    Output  BGMask_improved     (:2000,:2000)ndarray: binary array corresponding to BGMask, but
                                with new pieces of mask added.
    Author: Emil Haaber Tellefsen, 2023
    """     

    #preallocating new mask
    addMask = np.zeros(np.shape(BGMask),dtype=int)

    # adding piece of mask above first left tick
    addMask[270:nodesL[0]-50,100:200]=1
    addMask[270:nodesL[0]-50,240:310]=1

    # itterating through remaining left ticks
    for i in range(0,len(nodesL)-1):
        addMask[nodesL[i]+50:nodesL[i+1]-50,100:200]=1
        addMask[nodesL[i]+85:nodesL[i+1]-35,240:310]=1

    # adding piece of mask above first right tick
    addMask[270:nodesR[0]-50,1000:1080]=1
    addMask[270:nodesR[0]-50,1130:1200]=1

    # itterating through remaining left ticks
    for i in range(0,len(nodesR)-1):
        addMask[nodesR[i]+50:nodesR[i+1]-50,1000:1080]=1
        addMask[nodesR[i]+85:nodesR[i+1]-35,1130:1200]=1
    
    # adding new mask pieces to existing mask
    BGMask[addMask==1]=1
    return BGMask

def polynomial_detrend(img_f, background):
    """
    POLYNOMIAL_DETREND uses the background pixels of a N6ESMR image to fit a 2D polynomial over 
    for the image in the form:
        f(x,y) = w0 + w1*x + w2*y + w3*x^2 + w4*y^2
    This image is then subtracted from the original image as to detrend said image.

    usage: img_f_fit = polynomial_detrend(img_f, BGMask)

    Input   img_f               (:2000,:2000)ndarray: float version of an aligned N6ESMR image
            BGMask              (:2000,:2000)ndarray: binary array at same size as extracted
                                images, covering background pixels.
    Output  img_f_detrended     (:2000,:2000)ndarray: detrended version of img_f

    Author: Emil Haaber Tellefsen, 2023
    """     

    # extracting pixels
    bg, r, c = getPix(img_f,background)

    #removing bg edges cut durring alignment
    uncutbg = bg!=0
    bg=bg[uncutbg]
    r=r[uncutbg]
    c=c[uncutbg]

    # fitting polynomial
    x = np.array([r,c]).T

    def func(x,a,b,c,d,e):
        y = a + b*x[:,0] + c*x[:,0]**2 +  d*x[:,1] + e*x[:,1]**2
        return y

    k = curve_fit(func,xdata=x,ydata=bg)[0]

    # applying fit to matrix
    rc = np.mgrid[0:2000,0:2000]
    fit = k[0] + k[1]*rc[0] + k[2]*rc[0]**2  + k[3]*rc[1] + k[4]*rc[1]**2

    # removing trend from image
    img_f_fit = img_f - fit
    img_f_fit -= np.min(img_f_fit)

    return img_f_fit



#######################################################################
# extraction, morphing, and combining swaths in images
#######################################################################

def extract_bands(img_f):
    """
    EXTRACT_BANDS finds and returns all SCSP in the input image, using predefined edge boundaries. It also investigates whether any data is present
    in the right half of the image, as there is not always any data available in this section, and finds out where swaths start location is. Furthermore,
    the returned bands is resized to match original swath geometry.

    usage: bands, lpeaks, rpeaks = extract_bands(img_f)

    Input   img_f               (:2000,:2000)ndarray: float version of an aligned N6ESMR image
    Output  bands               (:20,(:,:71)ndarray)list: list of all extracted SCSP found
            lpeaks              (2,)ndarray: top and bottom pixel location for left SCPSs
            rpeaks              (2,)ndarray: top and bottom pixel location for right SCPSs

    Author: Emil Haaber Tellefsen, 2023
    """

    #defining SCSP location and transformation to satellite geometry
    bandstarts = np.array([317, 385, 453, 520, 587, 655, 723, 791, 860, 929, 1210, 1279, 1348, 1416, 1484, 1552, 1620, 1687, 1754, 1821])
    to_resampled = 759/(1620-339)

    #Extracting band estimate and transforming
    bands=[]
    for i in range(0,len(bandstarts)):
        band = img_f[339:1620,bandstarts[i]:(bandstarts[i]+62)]
        band = zoom(band,(0.59210,1.14516))
        bands.append(band)
    
    # find left SCSP ends
    testband=bands[7]
    db=find_derivative(testband[:,35])
    firstPeak = find_first_peak(-db,'start',4,minThresh=0.1) 
    lastPeak = find_first_peak(db,'end',4,minThresh=0.1)  

    # Checking if left peaks are available
    if np.isnan(firstPeak) or np.isnan(lastPeak):
        lpeaks = 0
        for i in range(0,10):
            bands[i]=0
    else:
        lpeaks = [int(339+firstPeak/to_resampled),int(339+lastPeak/to_resampled)]
        for i in range(0,10):
            bands[i]=bands[i][(firstPeak-4):lastPeak,:]    

    # find right SCSP ends
    testband=bands[17]
    db=find_derivative(testband[:,35])
    firstPeak = find_first_peak(-db,'start',4) 
    lastPeak = find_first_peak(db,'end',4)  

    # Checking if right peaks are available
    if np.isnan(firstPeak) or np.isnan(lastPeak):
        rpeaks = 0
        for i in range(10,20):
            bands[i]=0
    else:
        rpeaks = [int(339+firstPeak/to_resampled),int(339+lastPeak/to_resampled)]
        for i in range(10,20):
            bands[i]=bands[i][(firstPeak-4):lastPeak,:]


    return bands, lpeaks, rpeaks

def combine_bands(bands1, bands2):
    """
    COMBINE_BANDS takes 2 SCSP lists as input, and stitches them together. It takes into account the curvature of the swath, 
    and returns the combined swath.

    usage: bands_out = combine_bands(band1,band2)

    Input   band1       (:,(:,:71)ndarray)list: SCSP from the left part of the DI 
            band2       (:,(:,:71)ndarray)list: SCSP from the right part of the DI                       
    Output  bands_out    (:10,:,:71)ndarray: Array with merged SCSP i.e. Array of SCS 

    Author: Emil Haaber Tellefsen, 2023
    """
    width = 71
    cms = 4
    len_b1 = np.shape(bands1[0])[0]
    len_b2 = np.shape(bands2[0])[0]
    len_t = len_b1+len_b2-cms
    # defining curvarture to mask ends
    bands_out = np.zeros((len(bands1),len_t,width))
    curve = [0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,0]
    curveMask_upper = np.zeros((cms,len(curve)))
    curveMask_lower = np.zeros((cms,len(curve)))

    for i in range(0,len(curve)):
        curveMask_upper[curve[i]:,i]=1
        curveMask_lower[:curve[i],i]=1
    
    # putting bands into 2 arrays of similar dimensions
    for i in range(0,len(bands1)):
        band1_out = np.zeros((len_t,width))
        band2_out = np.zeros((len_t,width))
        band1_out[:len_b1 , :] = bands1[i]
        band2_out[len_t-len_b2: , :] = bands2[i]

        #removing top and bottom curve of band1
        band1_out[:cms,:] *= curveMask_upper
        band1_out[len_b1-cms:len_b1 , :] *= curveMask_lower

        #removing top and bottom curve of band2
        band2_out[len_t-len_b2 : len_t-len_b2+cms , :] *= curveMask_upper
        band2_out[-cms:,:]*=curveMask_lower

        bands_out[i,:,:] = band1_out + band2_out
    
    bands_out[bands_out==0]=float('nan')

    return bands_out

def trim_ends(bands):
    """
    TRIM_ENDS removes pixels at the end of the swath that are supposed to be outside of swath due to curvature of swath

    usage: bands_out = trim_ends(bands)
    Input   bands       (:10,:,:71)ndarray: Array with merged SCSP i.e. Array of SCS              
    Output  bands       (:10,:,:71)ndarray: Array with merged SCSP i.e. Array of SCS with curved ends replaced by NaN

    Author: Emil Haaber Tellefsen, 2023
    """
    width = 71
    cms = 4

    #Defining curve and mask
    curve = [0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,0]
    curveMask_upper = np.zeros((np.shape(bands)[0],cms,width))
    curveMask_lower = np.zeros((np.shape(bands)[0],cms,width))

    #Replaces curved end area with NaN
    for i in range(0,width):
        curveMask_upper[:,curve[i]:,i]=1
        curveMask_lower[:,:curve[i],i]=1
    bands[:,:cms,:] *= curveMask_upper
    bands[:,-cms:,:]*=curveMask_lower
    bands[bands==0]=float('nan')
    
    return bands

def convert_to_array(bands):
    """
    CONVERT_TO_ARRAY transforms list of swaths with similar shape into a single 3D array. Used when bands are not combined when 
    no data is available in left part of DI.

    usage: b_array = convert_to_array(bands)

    inputs: bands       (:,(:,:71)ndarray)list: SCSP from the right part of the DI. 
    outputs b_array     (:,:,71)ndarray: merged SCSP array 

    Author: Emil Haaber Tellefsen, 2023
    """
    b_array = np.zeros((len(bands),np.shape(bands[0])[0],np.shape(bands[0])[1]))
    for i in range(0,len(bands)):
        b_array[i,:,:]=bands[i]
    return b_array



#######################################################################
# Training and using classifier
#######################################################################

def find_class_attributes(img_f):
    """
    FIND_CLASS_ATTRIBUTES locates colorbar in the DI and extracts mean and standard deviation of each box
    usage: mu, std = find_class_attributes(img_f)
    inputs: img_f       (:2000,:2000)ndarray: float version of an aligned N6ESMR image
    outputs: mu         (18,)1darray: list of means for each box in colorbar
            std         (18,)1darray: list of standard deviations for each box in colorbar

    Author: Emil Haaber Tellefsen, 2023        
    """
    
    # defining box locations
    wbox = 94
    start= 183
    top= 1622
    bottom= 1757

    #calculating mean for each legend frame
    mu = np.zeros(18)
    std = np.zeros(18) 
    pos = start
    for i in range(0,18):   
        legendframe = img_f[top+10:bottom-10,  pos+10+i*wbox:pos-10+wbox+i*wbox] 
        mu[i]=np.mean(legendframe)
        std[i] = np.std(legendframe)
    return mu, std

def correct_mean(mu):
    """
    CORRECT_MEAN takes means from colorbar as input and regularizes it s.t. means of colorbar is asymptotically decreasing
    usage: mu_correct = correct_mean(mu)
    inputs: mu                  (18,)1darray: list of means for each box in colorbar
    outputs: mu_corrected       (18,)1darray: list of corrected means for each box in colorbar

    Author: Emil Haaber Tellefsen, 2023        
    """    
    mu_corrected = np.copy(mu)

    # setting ends to minimum and maximum found value
    mu_corrected[0] = np.max(mu)
    mu_corrected[-1] = np.min(mu)
    
    # ensuring mean is asymptotically decreasing with each box by iterratively meaning difference
    signs = np.sign(mu_corrected[1:] - mu_corrected[:-1]) 
    itt = 0
    while np.any(signs>=0) and itt<10:
        for i in range(1,len(mu)-1):
            if mu_corrected[i] >= mu_corrected[i-1] or mu_corrected[i] <= mu_corrected[i+1] :
                mu_corrected[i] = (mu_corrected[i-1] + mu_corrected[i+1])/2
        signs = np.sign(mu_corrected[1:] - mu_corrected[:-1])
        itt+=1
    
    return mu_corrected

def train_classifier(mu, y):
    """
    TRAIN_CLASSIFIER takes mean values of colorbar and box number as input, and returns a first order univariate spline function 
    for converting pixel values to an interpolated value of the colorbar.

    usage: spl = train_classifier(mu,y)

    inputs: mu      (18,)1darray: list of means for each box in colorbar
            y       (18,)1darray: list of colorbar numerical values
    outputs: spl    functionhandle: spline function for converting pixelvalue to colorbar value

    Author: Emil Haaber Tellefsen, 2023
    """
    x = np.flip(mu)
    y = np.flip(y)
    spl = InterpolatedUnivariateSpline(x,y,k=1)
    return spl

def apply_classifier(spline, swath, mu):
    """
    APPLY_CLASSIFIER aplies the spline function found using "train_classifier" to the swaths. It takes class means into account to cap pixels values,
    thus avoiding extrapolation.
    
    usage: c_swath = apply_classifier(spline,swath,mu)

    inputs: spline      functionhandle: spline function for converting pixelvalue to colorbar value
            swath       (:,:,:71)ndarray: merged SCSP array 
            mu          (18,)1darray: list of means for each box in colorbar
    outputs: c_swath    (:,:,:71)ndarray: merged SCSP array with pixelvalues replaced by spline values

    Author: Emil Haaber Tellefsen, 2023
    """

    swath[swath<mu[-1]]=mu[-1]
    swath[swath>mu[0]]=mu[0]
    return spline(swath)

def findBound(type, bandsMat, spl, mu, std, sig):
    """
    FINDBOUNDS finds the confidence interval of the spline transformed swaths

    usage CI_bound = findBound(type,bandsMat,spl,mu,std,sig)
    inputs:  type           "lower"/"upper": string telling whether lower or upper bound CI is to be found
             bandsMat       (:,:,:71)ndarray: merged SCSP array
             spl            functionhandle: spline function for converting pixelvalue to colorbar value
             mu             (18,)1darray: list of means for each box in colorbar
             std            (18,)1darray: list of standard deviations for each box in colorbar
             sig            significance of confidence interval
    Outputs: CI_bound       (:,:,:71)ndarray: Confidence interval bound in colorbar value for all bandsMat inputs
    
    Author: Emil Haaber Tellefsen, 2023    
    """
    
    if type=='lower':
        training_data = apply_classifier(spl,mu+sig*std,mu)
    elif type=='upper':
        training_data = apply_classifier(spl,mu-sig*std,mu)
    
    noncapped=np.logical_and(training_data!=1, training_data!=18)
    for i in range(0,17):
        if noncapped[i+1]==True:
            noncapped[i]=True
    for i in range(1,18):
        if noncapped[i-1]==True:
            noncapped[i]=True
    
    spl_Bound = train_classifier(mu[noncapped],training_data[noncapped])
    return apply_classifier(spl_Bound,bandsMat,mu[noncapped])



#######################################################################
# Transforming to TB
#######################################################################

def makeTBSplines(lut_version): 
    """
    MAKETBSPLINES generates splines that translates from colorbarvalues to brightness temperature using tables from NIMBUS 6 data catalogue vol 1
    as reference.

    usage: splines = makeTBSplines(lut_version)

    inputs: lut_version         binary: 0: 14 July 1975 - 12 August 1975 is used, 1: 13 August 1975 - 31 March 1976 format is used
    outputs: splines            dictionary: contains splines for all 10 SCS

    Author: Emil Haaber Tellefsen, 2023     
    """

    # Class TB values: 14 July 1975 - 12 August 1975
    if lut_version==0:
        lut = {
            "H01": np.array([201.5,198,194.5,191.5,188.5,185.5,182.5,179.5,176.5,173,169.5,166.5,163.5,160.5,157.5,154.5,151.5,148.5]),
            "V02": np.array([201.5,198,194.5,191.5,188.5,185.5,182.5,179.5,176.5,173,169.5,166.5,163.5,160.5,157.5,154.5,151.5,148.5]),
            "S03": np.array([201.5,198,194.5,191.5,188.5,185.5,182.5,179.5,176.5,173,169.5,166.5,163.5,160.5,157.5,154.5,151.5,148.5]),
            "H04": np.array([251.5,248,244.5,241.5,238.5,235.5,232.5,229.5,226.5,223,219.5,216.5,213.5,210.5,207.5,204.5,201.5,198.5]),  
            "V05": np.array([251.5,248,244.5,241.5,238.5,235.5,232.5,229.5,226.5,223,219.5,216.5,213.5,210.5,207.5,204.5,201.5,198.5]),
            "S06": np.array([251.5,248,244.5,241.5,238.5,235.5,232.5,229.5,226.5,223,219.5,216.5,213.5,210.5,207.5,204.5,201.5,198.5]),
            "H07": np.array([301.5,298,294.5,291.5,288.5,285.5,282.5,279.5,276.5,273,269.5,266.5,263.5,260.5,257.5,254.5,251.5,248.5]),
            "V08": np.array([301.5,298,294.5,291.5,288.5,285.5,282.5,279.5,276.5,273,269.5,266.5,263.5,260.5,257.5,254.5,251.5,248.5]),
            "S09": np.array([301.5,298,294.5,291.5,288.5,285.5,282.5,279.5,276.5,273,269.5,266.5,263.5,260.5,257.5,254.5,251.5,248.5]),
            "D10": np.array([ 51.5, 48, 44.5, 41.5, 38.5, 35.5, 32.5, 29.5, 26.5, 23, 19.5, 16.5, 13.5, 10.5,  7.5,  4.5,  1.5, -1.5]),
        }   
    if lut_version==1:
        # Class TB values: 13 August 1975 - 31 March 1976
        lut = {
            "H01": np.array([202,198,193.5,189,185,180.5,176,171.5,167,163,158.5,154,150,145.5,141,136.5,132,128]),
            "V02": np.array([232,228,224.5,221,217,213,209.5,206,202,198,194.5,191,187,183,179.5,176,172,168]),
            "S03": np.array([212,208,204.5,201,197,193,189.5,186,182,178,174.5,171,167,163,159.5,156,152,148]),
            "H04": np.array([252,248,244.5,241,237,233,229.5,226,222,218,214.5,211,207,203,199.5,196,192,188]),  
            "V05": np.array([271.5,268.5,265.5,262.5,259.5,256,252.5,249.5,246.5,243.5,240.5,237.5,234.5,231,227.5,224.5,221.5,218.5]),
            "S06": np.array([251.5,248.5,245.5,242.5,239.5,236,232.5,229.5,226.5,223.5,220.5,217.5,214.5,211,207.5,204.5,201.5,198.5]),
            "H07": np.array([291.5,288.5,285.5,282.5,279.5,276,272.5,269.5,266.5,263.5,260.5,257.5,254.5,251,247.5,244.5,241.5,238.5]),
            "V08": np.array([301.5,299,296.5,294,291.5,289,286.5,284,281.5,279,276.5,274,271.5,269,266.5,264,261.5,259]),
            "S09": np.array([281.5,279,276.5,274,271.5,269,266.5,264,261.5,259,256.5,254,251.5,249,246.5,244,241.5,239]),
            "D10": np.array([142,138,134.5,131,127,123,119.5,116,112,108,104.5,101,97,93,89.5,86,82,78]),
         }

    splines = [InterpolatedUnivariateSpline(np.arange(1,19),lut["H01"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["V02"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["S03"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["H04"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["V05"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["S06"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["H07"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["V08"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["S09"],k=1),
            InterpolatedUnivariateSpline(np.arange(1,19),lut["D10"],k=1)]
    return splines

def convert2TB(ClBandsMat, splines):
    """
    CONVERT2TB converts colobar transformed bands to brightness tempeature by applying splines to all SCS and merging SCS intervals from the same channel
    to form the CS TH, TV, TS and TD

    usage: TB = convert2TB(ClBandsMat,splines)

    inputs:  ClBandsMat     (:10,:,:71)ndarray: merged SCSP array with pixelvalues replaced by spline values
             splines         dictionary: contains splines for all 10 SCS
    outputs: TB             (:4,:,:71)ndarray: Final brightness temperature arrays corresponding to TH, TV, TS and TD

    Author: Emil Haaber Tellefsen, 2023
    """

    #defining overlap section between arrays
    Uppercapped=(ClBandsMat>=17.5)
    Lowercapped=(ClBandsMat<=1.5)

    TB_full = np.zeros(ClBandsMat.shape)

    #applying splines
    for i in range(0,10):
        TB_full[i,:,:] = splines[i](ClBandsMat[i,:,:])

    #replacing capped values with NaN
    TB_full[0:3,:,:][Lowercapped[0:3,:,:]]=float('nan')
    TB_full[3:6,:,:][Uppercapped[3:6,:,:]]=float('nan')
    TB_full[3:6,:,:][Lowercapped[3:6,:,:]]=float('nan')
    TB_full[6:9,:,:][Uppercapped[6:9,:,:]]=float('nan')

    #meaning values
    TB = np.zeros((4,ClBandsMat.shape[1],ClBandsMat.shape[2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        TB[0,:,:]=np.nanmean(TB_full[[0,3,6],:,:],axis=0)
        TB[1,:,:]=np.nanmean(TB_full[[1,4,7],:,:],axis=0)
        TB[2,:,:]=np.nanmean(TB_full[[2,5,8],:,:],axis=0)
        TB[3,:,:]=TB_full[9,:,:]

    return TB



#######################################################################
# finding and reprojecting nodes from axis lines
#######################################################################

def find_axis_nodes(img_f):
    """
    FIND_AXIS_NODES finds MAT along axis lines

    usage: nodesL, nodesR = find_axis_nodes(img_f)
    
    inputs:   img_f           (:2000,:2000)ndarray: float version of an aligned N6ESMR image
    outputs:  nodesL          1Darray: column location of all ticks found at left axis line
              nodesR          1Darray: column location of all ticks found at right axis line   

    Author: Emil Haaber Tellefsen, 2023
    """
    #finding line with MAT
    nodeLineL = -img_f[300:1600,230]
    nodeLineR = -img_f[300:1600,1115]

    #detecting peaks
    nodesL = find_peaks(nodeLineL,height=np.mean(nodeLineL)+3*np.std(nodeLineL),distance=100)[0]+300
    nodesR = find_peaks(nodeLineR,height=np.mean(nodeLineR)+3*np.std(nodeLineR),distance=100)[0]+300
    return nodesL, nodesR

def nodes_to_swath(nodesL, nodesR, edgesL, edgesR):
    """ 
    NODES_TO_SWATH converts the found MATS to same relative reference frame as the resized and merged swaths used in the final dataset

    usage: newNodes = nodes_to_swath(nodesL, nodesR, edgesL, edgesR)

    inputs:  nodesL     1Darray: column location of all ticks found at left axis line
             nodesR     1Darray: column location of all ticks found at right axis line   
             edgesL     (2,)ndarray: top and bottom pixel location for left SCPSs
             edgesR     (2,)ndarray: top and bottom pixel location for right SCPSs
    outputs: newNodes   1Darray: axis ticks converted to new format

    Author: Emil Haaber Tellefsen, 2023    
    """

    #defining resampling constant
    to_resampled = 759/(1620-339)
    newNodes = np.zeros(len(nodesL)+len(nodesR))

    # if no data is available at right side of DI
    if np.isscalar(edgesR)==False:
        for i in range(0,len(nodesR)):
            node = nodesR[i]*to_resampled
            node -= edgesR[0]*to_resampled
            newNodes[i]=int(node)
    
        for j in range(0,len(nodesL)):
            node = nodesL[j]*to_resampled
            node = node - edgesL[0]*to_resampled + edgesR[1]*to_resampled - edgesR[0]*to_resampled
            newNodes[i+j+1]=int(node)
    
    # if data is available at both sides of DI
    else:
        for i in range(0,len(nodesR)):
            node = nodesR[i]*to_resampled
            node = node - (1603*to_resampled + (edgesL[0]*to_resampled - 387*to_resampled))
            newNodes[i]=int(node)
    
        for j in range(0,len(nodesL)):
            node = nodesL[j]*to_resampled
            node -= edgesL[0]*to_resampled
            newNodes[i+j+1]=int(node)        
    return newNodes



#######################################################################
# reading image text
#######################################################################

def histogram_stretch(img):
    """
    HISTOGRAM_STRETCH Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255'

    Author: Emil Haaber Tellefsen, 2022  
    """

    # img_as_float will divide all pixel values with 255.0
    img_f = img_as_float(img)

    min_val = img_f.min()
    max_val = img_f.max()
    min_desired = 0.0
    max_desired = 1.0
	
    img_out=(max_desired-min_desired)/(max_val-min_val)*(img_f-min_val)+min_desired

    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)

def modify_gamma(img, gamma):
    """
    MODIFY_GAMMA modifies the contrast of an image

    usage: im_out = modify_gamma(img,gamma)
    input:   img      ndarray: image that is to be modifed - works for both float and integer
             gamma    float: gamma value to modify image with
    outputs: im_out   ndarray: ubyte image with its gamma modified.

    Author: Emil Haaber Tellefsen, 2023      
    """
    img_f = img_as_float(img)
    img_out = np.power(img_f,gamma)
    
    return img_as_ubyte(img_out)

def iter_Gamma(img, outputlength, gamma_limit=3, whitelist='0123456789', checkFirstLetter=0):
    """
    ITER_GAMMA iteratively increases gamma of an image and tries to read the text of said image using Tesseract. It stops once a string output
    of the specified length is found

    usage: output = iter_Gamma(img, outputlength, gamma_limit=3, whitelist='0123456789', checkFirstLetter=0)
    inputs:  img                2darray: text image that is to be read
             outputlength       int:     length of desired output string
             gamma_limit        float:   upper contrast limit that are to be tested
             whitelist          str:     whitelist of letters/numbers Tesseract is to look for
             checkFirstLetter   0/1:     Check if first symbol is letter - is the case for coordinates
    outputs: output             str:     Found word, or "X" if not succesful

    Author: Emil Haaber Tellefsen, 2023    
    """

    # check if a string can be converted to number
    def is_number(n):
        try:
            float(n)   # Type-casting the string to `float`.
                    # If string is not a valid `float`, 
                    # it'll raise `ValueError` exception
        except ValueError:
            return False
        return True

    gamma = 1

    #starting gamma iteration
    while gamma <= gamma_limit:
            gamma+=0.1 
            mod_image = modify_gamma(img,gamma)

            #reading text
            output = pytesseract.image_to_string(mod_image,config='--psm 7  -c  load_system_dawg=0  -c load_freq_dawg=0  -c tessedit_char_whitelist='+whitelist )
            
            #checking if found word is valid
            if len(output)==outputlength:
                if checkFirstLetter==1:
                    if output[0].isdigit()==False:
                        if is_number(output[1:]):
                            return output
                else:
                     return output
            
            #returning "X" if gamma_limit is reached
            if gamma > gamma_limit:
                return 'X\n'

def read_title_info(img):
    """
    READ_TITLE_INFO reads the relevant title information of an image, and validates it

    usage: basetime, orbit = read_title_info(img)

    inputs:  img        ndarray: alligned DI in uint format from which title info is to be read
    outputs: basetime   datetime64: Date of orbit as read from title info
             orbit      int: orbit number for orbit as read from title
    
    Author: Emil Haaber Tellefsen, 2023       
    """

    # Function for validating format of date
    def valid_date(date):
        if len(date)!=9:
            return False
        
        if date[:2].isdigit()==False or date[3:5].isdigit()==False or date[6:8].isdigit()==False:
            return False

        month = int(date[0:2])
        day = int(date[3:5])
        year = int(date[6:8])

        if (month <= 0) or (month > 12):
            return False
        if (day <= 0) or (day > 31):
            return False
        if (year<75) or (year>77):
            return False
        
        return True
    
    # function for validating format of orbit number
    def valid_orbit(orbit):
        if len(orbit)!=7:
            return False
        
        if orbit[:6].isdigit()==False:
            return False
        
        if orbit[0]!='0':
            return False
        if int(orbit[1]) > 1:
            return False
        
        return True    

    # Histogram stretching image of date and orbit
    date_img = histogram_stretch(img[100:200,640:940])
    orbit_img = histogram_stretch(img[100:200,1650:1880])

    # Reading date and orbit
    date  = pytesseract.image_to_string(date_img ,config='--psm 7  -c  tessedit_char_whitelist=-0123456789 -c load_system_dawg=0 -c load_freq_dawg=0')
    orbit = pytesseract.image_to_string(orbit_img,config='--psm 7  -c  tessedit_char_whitelist=0123456789 -c load_system_dawg=0 -c load_freq_dawg=0')

    # If date or orbit is invalid, it is send through iterative gamma function until valid date and/or orbit is reached
    if valid_date(date)==False:
        date = iter_Gamma(date_img,outputlength=9,whitelist='-0123456789')
        if valid_date(date)==False:
            date = 'X\n'
    if valid_orbit(orbit)==False:
        orbit = iter_Gamma(orbit_img,outputlength=7,whitelist='0123456789')
        if valid_orbit(orbit)==False:
            orbit = 'X\n'
    
    # return date in datetime format or return 1900-01-01 if tesseract is unsuccesful
    if date[:-1] != 'X':
        basetime = datetime.datetime(year=int('19'+date[6:-1]),month=int(date[:2]),day=int(date[3:5]))
    else:
        basetime = datetime.datetime(year=1900,month=1,day=1)

    #return integer of orbit or 0 if orbit is not read succesfully
    if orbit[:-1] != 'X':
        orbit = int(orbit)
    else:
        orbit = 0
    return basetime, orbit

def read_coordinates(img, nodesL, nodesR):
    """
    READ_COORDIANTES reads the time, latitude and longitude coordinates using tesseract and validates they are of correct format before passing

    usage: time_out, lat_out, lon_out = read_coordinates(img, nodesL, nodesR)

    inputs:  img        ndarray: alligned DI in uint format from which title info is to be read
             nodesL     1Darray: column location of all ticks found at left axis line
             nodesR     1Darray: column location of all ticks found at right axis line
    outputs: time_out   list: list of times in string format
             lat_out    list: list of latitudes in string format
             lon_out    list: list of longitudes in string format

    Author: Emil Haaber Tellefsen, 2023               
    """

    time_out=[]
    lat_out=[]
    lon_out=[]
    
    #checking all right axis coordinates
    for peak in nodesR:
        text_images = [zoom(img[peak-50:peak+50,995:1080],2),zoom(img[peak-25:peak+25,1130:1210],2),zoom(img[peak+25:peak+75,1130:1210],2)]
        
        text_images_enh=[]
        for i in range(0,3):
            text_images_enh.append(histogram_stretch(text_images[i]))
        
        #reading coordinates
        time=pytesseract.image_to_string(text_images_enh[0],config='--psm 7  -c  tessedit_char_whitelist=0123456789 -c load_system_dawg=0 -c load_freq_dawg=0')
        lat=pytesseract.image_to_string(text_images_enh[1],config='--psm 7  -c  tessedit_char_whitelist=0123456789NS -c load_system_dawg=0 -c load_freq_dawg=0')
        lon=pytesseract.image_to_string(text_images_enh[2],config='--psm 7  -c  tessedit_char_whitelist=0123456789WE -c load_system_dawg=0 -c load_freq_dawg=0')

        #Iterating if formats are incorrect
        if len(time)!=5:
            time = iter_Gamma(text_images_enh[0],5)
            
        if len(lat)!=4:
            lat = iter_Gamma(text_images_enh[1],4,whitelist='0123456789NS',checkFirstLetter=1)
        elif lat[0].isdigit() or lat[1].isalpha() or lat[2].isalpha():
            lat = iter_Gamma(text_images_enh[1],4,whitelist='0123456789NS',checkFirstLetter=1)

        if len(lon)!=5:
            lon = iter_Gamma(text_images_enh[2],5,whitelist='0123456789EW',checkFirstLetter=1)
        elif lon[0].isdigit() or lon[1].isalpha() or lon[2].isalpha() or lon[3].isalpha():
            lon = iter_Gamma(text_images_enh[2],5,whitelist='0123456789EW',checkFirstLetter=1)
        elif int(lon[1])>1:
            lon = iter_Gamma(text_images_enh[2],5,whitelist='0123456789EW',checkFirstLetter=1)

        time_out.append(time[:-1])
        lat_out.append(lat[:-1])
        lon_out.append(lon[:-1])      

    # checking all left axis coordinates
    for peak in nodesL:
        text_images = [zoom(img[peak-50:peak+50,115:200],2),zoom(img[peak-25:peak+25,240:315],2),zoom(img[peak+25:peak+75,240:315],2)]
        
        text_images_enh=[]
        for i in range(0,3):
            text_images_enh.append(histogram_stretch(text_images[i]))
        
        #reading coordinates
        time=pytesseract.image_to_string(text_images_enh[0],config='--psm 7  -c  tessedit_char_whitelist=0123456789 -c load_system_dawg=0 -c load_freq_dawg=0')
        lat=pytesseract.image_to_string(text_images_enh[1],config='--psm 7  -c  tessedit_char_whitelist=0123456789NS load_system_dawg=0 -c load_freq_dawg=0')
        lon=pytesseract.image_to_string(text_images_enh[2],config='--psm 7  -c  tessedit_char_whitelist=0123456789WE load_system_dawg=0 -c load_freq_dawg=0')

        #Iterating if formats are incorrect
        if len(time)!=5:
            time = iter_Gamma(text_images_enh[0],5)
            
        if len(lat)!=4:
            lat = iter_Gamma(text_images_enh[1],4,whitelist='0123456789NS',checkFirstLetter=1)
        elif lat[0].isdigit() or lat[1].isalpha() or lat[2].isalpha():
            lat = iter_Gamma(text_images_enh[1],4,whitelist='0123456789NS',checkFirstLetter=1)

        if len(lon)!=5:
            lon = iter_Gamma(text_images_enh[2],5,whitelist='0123456789EW',checkFirstLetter=1)
        elif lon[0].isdigit() or lon[1].isalpha() or lon[2].isalpha() or lon[3].isalpha():
            lon = iter_Gamma(text_images_enh[2],5,whitelist='0123456789EW',checkFirstLetter=1)
        elif int(lon[1])>1:
            lon = iter_Gamma(text_images_enh[2],5,whitelist='0123456789EW',checkFirstLetter=1)

        # returning found coordinates
        time_out.append(time[:-1])
        lat_out.append(lat[:-1])
        lon_out.append(lon[:-1])            

    return time_out, lat_out, lon_out



#######################################################################
# fitting image text data to geometry
#######################################################################

def time_string2sec(time):
    """
    TIME_STRING2SEC converts string times to float format

    usage: time_s = time_string2sec(time)
    inputs:  time        list: time coordinates in string format
    outputs: time_s      1darray: array of times in float format (past midnight)

    Author: Emil Haaber Tellefsen, 2023     
    """
    
    N=len(time)
    time_s=np.zeros(N)
    i=0
    for t in time:
        if t.isdigit():
            time_s[i]=float(t[:2])*60**2+float(t[2:])*60
        else:
            time_s[i]=float('nan')
        i+=1

    return time_s

def corrected_time(time_s):
    """
    CORRECTED_TIME takes a time array and ensures there are exactly 15 minutes between each time stamp

    usage: time_model, succes = corrected_time(time_s)
    inputs:  time_s         1darray: array of times in float format (past midnight)
    outputs: time_model     1darray: array of validated times in float format (past midnight). If no consensus could be reached, first coordinate is set to midnight
             succes         bool: determines whether a consensus on times could be reached
    
    Author: Emil Haaber Tellefsen, 2023    
    """
    time_cont = np.copy(time_s)

    #calculating continueous time array of 15 minutes
    N=len(time_cont)
    for i in range(0,N):
        if time_cont[i]==0:
            time_cont[i:]+=60*60*24
    
    #checks through coordinates and check that they are correct, and otherwise what their expected value should be
    for i in range(0,N):
        time_model=np.arange(0,N)*900+time_cont[i]-i*900
        error = (time_model!=time_cont)
        if np.sum(error)<N/2:
            time_model[time_model >= 60*60*24] -= 60*60*24
            return time_model, True
    
    #returning past midnight if not succesfull
    return np.arange(0,N)*900, False

def fit_time(time_s, N, Nodes):
    """
    FIT_TIME fits times to swath taking curvature into account

    usage: timeMat = fit_time(time_s, N, Nodes)
    inputs: time_s    1Darray: corrected time stamps in float format
            N         int: number of rows in swath
            Nodes     1Darray: MATS found in format matching out swaths
    outputs: timeMat  (:,:71)ndarray: array containing each observations timestamp in timedelta format

    Author: Emil Haaber Tellefsen, 2023        
    """

    #defining starting times
    startTime = time_s[0] - 5.3 * Nodes[0]
    M = 71
    timeMat=np.ones((N,M))*startTime

    #making curve and meshgrid
    curve = np.array([0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,0])
    X,Y=np.meshgrid(curve,np.arange(0,N))

    # calculating time
    timeMat+=(Y-X)*5.3

    def map_timedelta(x):
        return datetime.timedelta(seconds=x)
    map_timedelta = np.vectorize(map_timedelta)

    # convert to timedelta format
    timeMat =  map_timedelta(timeMat)
    
    return timeMat

def coord2float(coords, type):
    """
    COORD2FLOAT converts cordinates to float format

    usage: coords_out = coord2float(coords,type)

    inputs:  coords     list: list of coordinates in string format that are to be converted
             type       "lat" or "lon": specifies which convertion that are to be used
    outputs: coords_out "coordinates out in float format

    Author: Emil Haaber Tellefsen, 2023            
    """
    
    # assigning type
    type = type.lower()
    if type=='latitude' or type == 'lat':
        negsignLetter = 'S'
    elif type=='longitude' or type == 'lon':
        negsignLetter = 'W'
    else:
        return 'ERROR'
    
    # making array
    coords_out = np.zeros(len(coords))
    i=0
    
    #converting
    for c in coords:
        if c[0]=='X':
            coords_out[i]=float('nan')
        else:
            if c[0]==negsignLetter:
                sign = -1
            else:
                sign = 1
        
            coords_out[i]=sign*float(c[1:])
        i+=1
    return coords_out

def haversine(p1, p2):
        """
        HAVERSINE calculates the shortest earth arc circle distance between 2 sets of coordinates, p1 and p2. Earth is assumed spherical
        with radius of 6371 km

        usage: d = haversine(p1,p2)
        inputs:  p1     (2,)array: (lat,lon) coordinates for first point
                 p2     (2,)array: (lat,lon) coordinates for second point
        outputs: d      float: arccircle distance between points in unit meter

        Author: Emil Haaber Tellefsen, 2023
        """

        # convert coordinates to radians
        to_rad = np.pi/180
        lat1_rad = p1[0,:]*to_rad
        lon1_rad = p1[1,:]*to_rad
        lat2_rad = p2[0,:]*to_rad
        lon2_rad = p2[1,:]*to_rad

        #define radius
        R = 6371000

        #defining coordinate difference
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        #returning arc circle distance
        return 2*R*np.arcsin(np.sqrt(np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2))

def residualError(lat_f, lon_f, newNodes, t0, lambda_e):
    """
    RESIDUALERROR finds the mean residual error between a set of lat/lon points, and the corresponding points found from the fitted
    NIMBUS 6 orbit model, defined by "newNodes" which is the converted MATs from the DI swats, and t0 and lambda_e which is the orbit
    parameters.

    usage: meanDistErr = residualError(lat_f, lon_f, newNodes, t0, lambda_e)

    inputs:  lat_f          1Darray: set of latitude coordinates in float format
             lon_f          1Darray: set of longitude coordinates in float format
             newNodes       1Darray: list of MAT vertical location in transformed SC
             t0             float: t0 parameter in orbit model defining start time of orbit
             lambda_e       float: lambda_e parameter for orbit model defining location of last crossing of equator
    outputs: meanDistErr    float: average residual error between model and coordinates in meters

    Author: Emil Haaber Tellefsen, 2023
    """

    # converting node locations to time
    t_fit=newNodes*5.3

    # Defining orbit parameters
    omega = - 2*np.pi / (107.25*60)
    omega_E = 2*np.pi / (23*3600 + 56*60 + 4)
    to_rad = np.pi/180
    to_degs = 180/np.pi
    inc = 81*to_rad
    tau = omega*(t_fit-t0)
    v = lambda_e - omega_E*(t_fit-t0)

    # calculating orbit in 3D cartesian coordinates
    x = np.cos(tau)*np.cos(v) - np.cos(inc)*np.sin(tau)*np.sin(v)
    y = np.cos(tau)*np.sin(v) + np.cos(inc)*np.sin(tau)*np.cos(v)
    z = np.sin(inc)*np.sin(tau)

    # converting to spherical reference frame
    lat_est = np.arctan2(z,np.sqrt(x**2+y**2))*to_degs
    lon_est = np.arctan2(y,x)*to_degs

    #calculating expected difference between model coordinates and read coordinates
    ErrDist = haversine(np.array([lat_f,lon_f]),np.array([lat_est,lon_est]))
    
    #taking the mean difference
    return np.nanmean(ErrDist)

def find_geometry_parameters(lat_f, lon_f, newNodes):
    """
    FIND_GEOMETRY_PARAMETERS fits the orbital model of the NIMBUS 6 satellite to a set of latitude and longitude coordinates

    usage: t0, lambda_e, min_err = find_geometry_parameters(lat_f, lon_f, newNodes)

    inputs:  lat_f      1Darray: set of latitude coordinates in float format
             lon_f      1Darray: set of longitude coordinates in float format
             newNodes   1Darray: list of MAT vertical location in transformed SC
    outputs: t0         float: t0 parameter in orbit model defining start time of orbit
             lambda_e   float: lambda_e parameter for orbit model defining location of last crossing of equator
             min_err    float: average residual error between model and coordinates in meters
    
    Author: Emil Haaber Tellefsen, 2023    
    """

    #converting nodes to time format
    t_fit=newNodes*5.3

    # defining function for fitting latitudes - used for finding t0
    def latfun(t, t0):
        """
        function for fitting t0 using latitude coordinates
        """
        #defining orbit parameters
        lambda_e = 0
        omega = - 2*np.pi / (107.25*60)
        omega_E = 2*np.pi / (23*3600 + 56*60 + 4)
        to_rad = np.pi/180
        to_degs = 180/np.pi
        inc = 80*to_rad

        # defining 3d cartesian orbit
        tau = omega*(t-t0)
        v = lambda_e - omega_E*(t-t0)
        x = np.cos(tau)*np.cos(v) - np.cos(inc)*np.sin(tau)*np.sin(v)
        y = np.cos(tau)*np.sin(v) + np.cos(inc)*np.sin(tau)*np.cos(v)
        z = np.sin(inc)*np.sin(tau)

        #calculating latitude
        lat = np.arctan2(z,np.sqrt(x**2+y**2))*to_degs
        return lat
    
    # finding t0 using latitude coordinates
    lat_fit = lat_f[np.isnan(lat_f)==False]
    t_fit_lat = t_fit[np.isnan(lat_f)==False]
    t0 = curve_fit(latfun,xdata=t_fit_lat,ydata=lat_fit,p0 = 0,bounds=(-150*60,150*60))[0][0]

    # defining function for fitting longitudes - used for finding lambda_e
    def lonfun(t, lambda_e):
        """
        function for fitting lambda_e using longitude coordinates and found t0
        """
        #defining orbit parameters
        omega = - 2*np.pi / (107.25*60)
        omega_E = 2*np.pi / (23*3600 + 56*60 + 4)
        to_rad = np.pi/180
        to_degs = 180/np.pi
        inc = 81*to_rad
        
        # defining 3d cartesian orbit
        tau = omega*(t-t0)
        v = lambda_e - omega_E*(t-t0)
        x = np.cos(tau)*np.cos(v) - np.cos(inc)*np.sin(tau)*np.sin(v)
        y = np.cos(tau)*np.sin(v) + np.cos(inc)*np.sin(tau)*np.cos(v)

        # finding longitude
        lon = np.arctan2(y,x)*to_degs
        return lon

    # fitting longitudes - function is periodic, so 100 guesses for initial lambda_e is chosen and best fit is returned - brute force method
    lon_fit = lon_f[np.isnan(lon_f)==False]
    t_fit_lon = t_fit[np.isnan(lon_f)==False]
    guesses = 100
    lambda_e_prel = np.zeros(guesses)
    res_err = np.zeros(guesses)
    init_guess = np.linspace(-np.pi,np.pi,guesses)
    for i in range(0,guesses):
        lambda_e_prel[i] = curve_fit(lonfun,xdata=t_fit_lon,ydata=lon_fit, p0 = init_guess[i],bounds=(-np.pi,np.pi))[0]
        res_err[i] = residualError(lat_f, lon_f, newNodes, t0, lambda_e_prel[i])
        
    # choosing fit with minimum fitting error
    min_err = np.min(res_err)
    lambda_e = lambda_e_prel[res_err==min_err][0]

    return t0, lambda_e, min_err

def fit_geometry(N, t0, lambda_e):
    """
    FIT_GEOMETRY fits the NIMBUS 6 orbital model to the swaths for the final data product using the found orbital parameters

    usage: lat_out, lon_out = fit_geometry(N, t0, lambda_e)

    inputs:  N          int: number of rows in swath
             t0         float: t0 parameter in orbit model defining start time of orbit
             lambda_e   float: lambda_e parameter for orbit model defining location of last crossing of equator
    outputs: lat_out    (:,:71)ndarray: array of latitude coordinates in same format as TB swaths
             lon_out    (:,:71)ndarray: array of longitude coordinates in same format as TB swaths
    
    Author: Emil Haaber Tellefsen, 2023    
    """

    #making time and width meshgrid
    t = np.arange(0,N)*5.3
    w = np.linspace(-636/6371,636/6371,71)
    W,T = np.meshgrid(w,t)

    #difining orbit parameters
    omega = - 2*np.pi / (107.25*60)
    omega_E = 2*np.pi / (23*3600 + 56*60 + 4)
    to_rad = np.pi/180
    to_degs = 180/np.pi
    inc = 80*to_rad
    
    # fitting 3D model
    tau = omega*(T-t0)
    v = lambda_e - omega_E*(T-t0)
    x = np.cos(tau)*np.cos(v) - np.cos(inc)*np.sin(tau)*np.sin(v) - np.sin(v)*W
    y = np.cos(tau)*np.sin(v) + np.cos(inc)*np.sin(tau)*np.cos(v) + np.sin(inc)*np.cos(v)*W
    z = np.sin(inc)*np.sin(tau) - np.cos(inc)*W

    #converting to lat/lon coordinates
    lon_out = np.arctan2(y,x)*to_degs
    lat_out = np.arctan2(z,np.sqrt(x**2+y**2))*to_degs

    return lat_out, lon_out


