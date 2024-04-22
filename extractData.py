"""
Takes all alligned N6ESMR data images (DI) in source directory, extract header, axis and
swath information and uses this to create a NetCDF file that stores the extracted data in
a georeferenced format.
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
import xarray as xr
from tqdm import tqdm

# -- Proprietary modules -- #
import projectFunctions as pf



#######################################################################
#User Inputs
#######################################################################

positive = False  #Choose whether negative of positive TIFs are to be processed 


#######################################################################
#Constants
#######################################################################
ad = """
- image_number : number granted to image durring current processing. Numbers mostly match cronological order. \n
- orbit: orbit number. If not recoverable, number is set to 0. Be aware of reading errors. \n
- image_type: negative or positive depending on source image format. \n
- original_image_title: Name of source TIFF image as given by GES DISC.  \n
- frame_number: Which full frame in source image data coresponds to. There are 1-3 frames available in each TIFF. \n
- total_axis_ticks: Number of axis ticks used for data fitting. \n
- latitude_Points_Found: Number of latitude coordinates that have been reliably read. Is at best equal to total_axis_ticks. \n
- longitude_Points_Found: Number of longitude coordinates that have been reliably read. Is at best equal to total_axis_ticks.\n
- fitting_error_km: spatial error of coordinates in frame, defined as MSE of predicted orbit vs coordinates read. Very poor fitting error (>250km) mainly stem from faulty reading of coordinates. \n
- true_times_available: True if true time of day was recoverable. Alternatively, times starts at midnight, and this is set to False.
"""

minValues = [np.array([150,150,150,0]), np.array([300,300,300,50])]
maxValues = [np.array([130,170,150,80]), np.array([290,300,280,140])]



#######################################################################
#Code
#######################################################################

# defining directory for separated DI and save directory alligned DI
if positive:
    source_dir = "./outputs/positives/alligned"
    save_dir   = "./outputs/positives/final"   
    err_report_name = "extractData_positives.err"
    imType = 'Epos'
    lastFormat1Image = "N6_Box18_140_3287_Epos_076_2.png"
    lastFormat2Image = "N6_Box18_3288_5160_Epos_136_2.png"
    
else:
    source_dir = "./outputs/negatives/alligned"
    save_dir   = "./outputs/negatives/final"
    imType = 'Eneg'
    err_report_name = "extractData_negatives.err"
    lastFormat1Image = "N6_Box18_140_3287_Eneg_067_2.png"
    lastFormat2Image = "N6_Box18_3288_5160_Eneg_139_1.png"

# making list of images and define error report in case of processing error
files = glob(source_dir + '/*.png')
err_report = ''


i = 0
for file in tqdm(files):
    filename = os.path.basename(file)

    try:
        # Determining if image is from before or after 13. Aug. 1975, And before or after 1. apr. 1976
        # 13. Aug. 1975 marks change in BT LUT
        # 1. apr. 1976 marks last day of current image format - images after cannot be processed with this algorithm
        if filename < lastFormat1Image:
            lut_version = 0
        elif filename < lastFormat2Image:
            lut_version = 1
        else:
            continue
        
        # loading image
        img = img_as_float(io.imread(file))

        # locating MAT
        nodesL, nodesR = pf.find_axis_nodes(img)

        #Improving background for detrending
        backgroundMask = pf.BGMask()
        BGMaskImp = pf.ImproveBGMask(backgroundMask, nodesL ,nodesR)

        #Removing light trend from DI
        img_detr = pf.polynomial_detrend(img, BGMaskImp)

        #finding SCSP
        bands, lpeaks, rpeaks = pf.extract_bands(img_detr)
        
        # Checking if SCSP is available in both sections of image and merges based on this
        if any(np.isscalar(band) for band in bands)==False and all(np.shape(band)[0]>50 for band in bands)==True:
            bandsMat = pf.combine_bands(bands[10:20], bands[0:10])
        else:
            bandsMat = pf.convert_to_array(bands[0:10])
            bandsMat = pf.trim_ends(bandsMat)
        N = np.shape(bandsMat)[1]

        # Calculating new location of nodes
        newNodes = pf.nodes_to_swath(nodesL, nodesR, lpeaks, rpeaks)

        # classifying bands using colorbar
        mu, std = pf.find_class_attributes(img_detr)
        mu_corrected = pf.correct_mean(mu)
        spl = pf.train_classifier(mu_corrected, np.arange(1,19))
        ClBandsMat = pf.apply_classifier(spl, bandsMat, mu_corrected)
        ClLower = pf.findBound('lower',bandsMat, spl, mu_corrected, std, 1.97)
        ClUpper = pf.findBound('upper',bandsMat, spl, mu_corrected ,std, 1.97)

        #Calculating BT using BT LUT
        splines=pf.makeTBSplines(lut_version)
        TB = pf.convert2TB(ClBandsMat, splines)
        TBUpper = pf.convert2TB(ClLower, splines)
        TBLower = pf.convert2TB(ClUpper, splines)

        # reading image text
        img_ubyte = img_as_ubyte(img_detr)
        date, orbit = pf.read_title_info(img_ubyte)
        time, lat, lon = pf.read_coordinates(img_ubyte, nodesL, nodesR)

        # Validating time measurements and fitting time
        time_s = pf.time_string2sec(time)
        time_s_cor, TrueTimesAvalailabe = pf.corrected_time(time_s)
        time_out = pf.fit_time(time_s_cor, N, newNodes)

        #converting coordinates to floats
        lat_f = pf.coord2float(lat, 'latitude')
        lon_f = pf.coord2float(lon, 'longitude')

        #finding parameters for parametric of satellite swath
        t0, lambda_e, resErr = pf.find_geometry_parameters(lat_f, lon_f, newNodes)
        lat_out, lon_out = pf.fit_geometry(N, t0, lambda_e)


        #Making NetCDF
        #######################################################################
        # making dataframe
        ds = xr.Dataset(
            data_vars = dict(
                Date = ([], date),
                Time = (["i", "beam"], time_out),
                Brightness_temperature = (["band", "i" ,"beam"], TB),
                Latitude  = (["i", "beam"], lat_out),
                Longitude = (["i", "beam"], lon_out),                
                CI_upper = (["band", "i", "beam"], TBUpper),
                CI_lower = (["band", "i", "beam"], TBLower),
            ),
            coords = dict(
                band = ("band",['TH','TV','TS','TD']),
                i = (["i"], np.arange(N)),
                beam = (["beam"], np.arange(71)),
            ),
            attrs=dict(
                description = "Raw data recovered from Nimbus-6 ESMR using image analysis techniques",
                title = "The NIMBUS 6 Electrically Scanning Microwave Radiometer: data rescue",
                summary = "In July 2022, NASA made TIFFs of NIMBUS-6 Electric Scanning Microwave Radiomenter (ESMR) image archived as 70mm photofacsimile film strip available, enabeling the recovery of the Brightness Temperature for the mission. Data is the result and error of recovery using prototype image analysis algorithm.",
                sensor = "ESMR",
                platform = 'NIMBUS-6',
                data_source = 'https://disc.gsfc.nasa.gov/datasets/ESMRN6IM_001/summary',
                image_number = i,
                orbit = orbit,
                image_type = 'positive', #'negative',
                original_image_title = filename[:-6] + '.tif',
                frame_number = int(filename[-5:-4]),
                total_axis_ticks = len(newNodes),
                latitude_Points_Found = len(newNodes) - np.sum(np.isnan(lat_f)),
                longitude_Points_Found = len(newNodes) - np.sum(np.isnan(lon_f)),
                fitting_error_km = resErr/1000,
                true_times_available = TrueTimesAvalailabe,
                creator_name = 'Emil Haaber Tellefsen',
                co_creators = 'Rasmus Tonboe (DTU Space), Wiebke Margitta Kolbe (DTU Space)',
                institution = 'Technical University of Denmark (DTU)',
                date_created = '2024-11-02',
                product_version = '1.0',
                datum = 'spherical (6371km radius)',
                attribute_description =  ad
            )
        )

        ds["band"].attrs = {
            "long_name": "Type of brigtness temperature band",
            "description": "Brightness temperature recordings in polaroids comes in 4 types; Horizontal, Vertical, Mean and difference. Be aware TS and TD were calculated differently between 12 June and 13 August 1975 compared to the remaining period.",
        }

        ds["Time"].attrs = {
            "long_name": "interpolated time of unique recording",
            "description": "UTC time calculated from image data for each recorded point. Recordings are in timedelta64 format relative to the given date. If true times were not recoverable, starting time is set to midnight, and true_times_available==False",
        }

        ds["Latitude"].attrs = {
            "long_name": "Latitude",
            "units": "degrees north"
        }

        ds["Longitude"].attrs = {
            "long_name": "Longitude",
            "units": "degrees east"   
        }

        ds["Date"].attrs = {
            "long_name": "start date of orbit",
            "description": "UTC Date at which the recording of the given orbit frames has started. If date could not be extracted, date is set to 1900-01-01. Be aware of reading errors.",
        }
        
        ds["i"].attrs = {
            "long_name": "index",
            "description": "index of rows in data array. As rows are curved this is not strictly analogous with time, though they correlate",
        }

        ds["beam"].attrs = {
            "long_name": "beam number",
            "description": "beam number defining each scan line of the sensor",
        }

        ds["Brightness_temperature"].attrs = {
            "long_name": "Brightness Temperature",
            "units": "K",
            "min_value": minValues[lut_version],
            "max_value": maxValues[lut_version],   
            "description": "Brightness Temperature at different bands. Be aware, min_value and max_value are different before and after 13 August 1975.",
        }

        ds["CI_upper"].attrs = {
            "long_name": "Upper Bound of Confidence Interval",
            "units": "K", 
            "min_value": minValues[lut_version],
            "max_value": maxValues[lut_version],   
            "description": "Upper bound of the 95% Confidence Interval for Brightness Temperature. Be aware of ceiling effects. Be aware, min_value and max_value are different before and after 13 August 1975.",
        }

        ds["CI_lower"].attrs = {
            "long_name": "Lower Bound Confidence Interval",
            "units": "K",
            "min_value": minValues[lut_version],
            "max_value": maxValues[lut_version],    
            "description": "Lower bound of the 95% Confidence Interval for Brightness Temperature. Be aware of ceiling effects. Be aware, min_value and max_value are different before and after 13 August 1975.",
        }
        
        # Exporting dataset as NetCDF
        outFilename = "N6ESMR_" + imType + "_recovered_v1.0_" + "{:04}".format(i) + ".nc"
        ds.to_netcdf(path = save_dir + '/' + outFilename ,mode = 'w', format = 'NETCDF4_CLASSIC')
        i+=1

    except:
        #Recording errors
        err_report = err_report + filename + "\n"
        pass

#writing error report
text_file = open('./reports/' + err_report_name, "w")
text_file.write(err_report)
text_file.close()