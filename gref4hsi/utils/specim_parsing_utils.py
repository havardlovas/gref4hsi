"""This script reads Specim data from default captures and reformats data to compatibility with pipeline"""
# Python builtins
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path

# Third party libraries
import numpy as np
from pathlib import Path
from spectral import envi
import pandas as pd
import h5py
import pymap3d as pm
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as RotLib

# Library dependencies
from gref4hsi.utils.geometry_utils import CalibHSI
from gref4hsi.utils.config_utils import prepend_data_dir_to_relative_paths

ACTIVE_SENSOR_SPATIAL_PIXELS = 1024 # Constant for AFX10
ACTIVE_SENSOR_SPECTRAL_PIXELS = 448 # Constant for AFX10

class Specim():
    def __init__(self, mission_path, config):
        self.mission_path = mission_path # With slash end
        self.config = config #
    
    @staticmethod
    def fov_2_param(fov):
        """
        param fov: view angle of each pixel in degrees
        returns: A dictionary of camera parameters
        """
        # Script for reading the FOV file and converting it to a camera model
        def optimize_func(param, x_true, n_pix):

            f = param[0]  # The focal length of the lens
            k1 = param[1]  # The 1st Radial distortion
            k2 = param[2]  # The 2nd Radial distortion
            k3 = param[3]
            c_x = param[4]
            n_pix = x_true.size
            u = np.arange(0, n_pix).reshape((-1,1)) + 0.5

            x_norm_lin = (u - c_x) / f
            # Distortion term if to be used
            x_norm_nonlin = -(k1 * (u - c_x) ** 5 + \
                            k2 * (u - c_x) ** 3 + \
                            k3 * (u - c_x) ** 2) / f

            
            x_norm = x_norm_lin + x_norm_nonlin
            diff_x = x_norm-x_true
            return diff_x.reshape(-1)
        
        
        theta_true = fov*np.pi/180

        x_true = np.array(np.tan(theta_true)).reshape((-1, 1))

        n_pix = x_true.size

        # Total angular FOV is:
        AFOV = np.abs(theta_true[0]) + np.abs(theta_true[-1])
        f_0 = (n_pix/2)/(np.tan(AFOV/2))

        param_0 = np.array([f_0, 0, 0, 0, 1 + n_pix/2])

        res = least_squares(optimize_func, param_0, args = (x_true, n_pix), x_scale = 'jac')

        param = res.x
        f = param[0]  # The focal length of the lens
        k1 = param[1]  # The 1st Radial distortion
        k2 = param[2]  # The 2nd Radial distortion
        k3 = param[3]  # The tangential distortion
        c_x = param[4] # The 

        # initialized to 0
        rotation_x = 0
        rotation_y = 0
        rotation_z = 0
        translation_x = 0
        translation_y = 0
        translation_z = 0

        param_dict = {'rx':rotation_x,
                      'ry':rotation_y,
                      'rz':rotation_z,
                      'tx':translation_x,
                      'ty':translation_y,
                      'tz':translation_z,
                      'f': f,
                      'cx': c_x,
                      'k1': k1,
                      'k2': k2,
                      'k3': k3,
                      'width': n_pix}
        return param_dict

    def read_fov_file(self, fov_file_path):
        

        

        df_fov = pd.read_csv(fov_file_path, sep = ',', header = None)

        fov_arr = df_fov.values[:, 1]

        param_dict = Specim.fov_2_param(fov = fov_arr)
        
        file_name_calib = self.config['Absolute Paths']['hsi_calib_path']

        CalibHSI(file_name_cal_xml=file_name_calib,
                 mode = 'w', 
                 param_dict = param_dict)

    def read_nav_file(self, nav_file_path, date):
        # Convert date string to datetime object
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        # Open file for reading.
        with open(nav_file_path, 'r') as nav_file:
            count = 0
            imu_data = []
            gnss_data = []
            sync_data = []
            # Set to -1 to an if sentence to initialize to true
            hsi_frame_nr_prev = -1
            # Variable introduced to convert from cyclic HS frame counting to cumulative counts
            restart_counts = 0

            while True:
                count += 1

                # Get next line from file
                line = nav_file.readline()

                line_arr = line.split(sep = ',')

                if line_arr[0] == '$PASHR':
                    Timestamp_str = line_arr[1]
                    sec = float(Timestamp_str[4:])
                    minut = int(Timestamp_str[2:4])
                    hour = int(Timestamp_str[0:2])

                    # Create a timedelta object for the time
                    time_delta = timedelta(hours=hour, minutes=minut, seconds=sec)

                    # Calculate the total timedelta
                    total_timedelta = date_obj + time_delta

                    # Convert to UNIX timestamp (decimal UNIX time)
                    unix_timestamp = (total_timedelta - datetime(1970, 1, 1)) / timedelta(seconds=1)

                    TimestampAbs = unix_timestamp

                    TimestampClock = date + '_' + Timestamp_str

                    Yaw = float(line_arr[2]) # Heading wrt true north
                    Roll = float(line_arr[4])
                    Pitch = float(line_arr[5])# According to https://docs.novatel.com/OEM7/Content/SPAN_Logs/PASHR.htm


                    sigmaRoll = float(line_arr[7])
                    sigmaPitch = float(line_arr[8])  # According to https://docs.novatel.com/OEM7/Content/SPAN_Logs/PASHR.htm
                    sigmaYaw = float(line_arr[9])

                    #dlogr_imu.append_data(np.array([TimestampClock, TimestampAbs, Roll, Pitch, Yaw, sigmaRoll, sigmaPitch, sigmaYaw]))

                    imu_data_row = {'TimestampClock':TimestampClock,
                                     'TimestampAbs': TimestampAbs,
                                     'Roll': Roll,
                                     'Pitch': Pitch,
                                     'Yaw': Yaw,
                                     'sigmaRoll': sigmaRoll,
                                     'sigmaPitch': sigmaPitch,
                                     'sigmaYaw': sigmaYaw}
                    imu_data.append(imu_data_row)

                elif line_arr[0] == '$GNGGA' or line_arr[0] == '$GPGGA':
                    Timestamp_str = line_arr[1]
                    sec = float(Timestamp_str[4:])
                    minut = float(Timestamp_str[2:4])
                    hour = float(Timestamp_str[0:2])

                    # Create a timedelta object for the time
                    time_delta = timedelta(hours=hour, minutes=minut, seconds=sec)

                    # Calculate the total timedelta
                    total_timedelta = date_obj + time_delta

                    # Convert to UNIX timestamp (decimal UNIX time)
                    unix_timestamp = (total_timedelta - datetime(1970, 1, 1)) / timedelta(seconds=1)

                    TimestampAbs = unix_timestamp

                    TimestampClock = date + '_' + Timestamp_str

                    LatDDMM_MM = float(line_arr[2]) # DDMM.mm
                    LatInt = int(LatDDMM_MM/100)
                    Lat = LatInt + (LatDDMM_MM - LatInt*100)/60


                    LonDDDMM_MM = float(line_arr[4]) # DDDMM.mm
                    LonInt = int(LonDDDMM_MM / 100)
                    Lon = LonInt + (LonDDDMM_MM - LonInt * 100) / 60

                    AltMSL = float(line_arr[9])
                    AltGeoid = float(line_arr[11])  # According to https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm

                    #dlogr_gnss.append_data(np.array([TimestampClock, TimestampAbs, Lon, Lat, AltMSL, AltGeoid]))
                    gnss_data_row = {'TimestampClock':TimestampClock,
                                     'TimestampAbs': TimestampAbs,
                                     'Lon': Lon,
                                     'Lat': Lat,
                                     'AltMSL': AltMSL,
                                     'AltGeoid': AltGeoid}
                    gnss_data.append(gnss_data_row)
                elif line_arr[0] == '$SPTSMP':
                    """Recordings of time sync messages, tied to the hyperspectral frames"""
                    hsi_frame_number_new = int(line_arr[2]) - 1 # Seems they start with a 1-base (like MATLAB)

                    if hsi_frame_number_new < hsi_frame_nr_prev: # When frame counter resets
                        restart_counts += (hsi_frame_nr_prev - hsi_frame_number_new) + self.metadata_obj.fps # 9950 + 50


                    hsi_frame_number = hsi_frame_number_new
                    hsi_frame_nr_prev = hsi_frame_number_new
                    prev_time = TimestampAbs

                    # In the file, fram counts go 1, 51, 101,... 9951, 1 ... in a cyclic manner. So to get cumulative counts we add the restart_counts
                    sync_data_row = {'HsiFrameNum': hsi_frame_number + restart_counts,
                                     'TimestampAbs': prev_time,
                                     'TimestampClock': TimestampClock}
                    sync_data.append(sync_data_row)




                # if line is empty
                # end of file is reached
                if not line:
                    break
        
        self.imu_data = imu_data
        self.gnss_data = gnss_data
        self.sync_data = sync_data

"""Writer for the h5 file format using a dictionary. The user provides h5 hierarchy paths as values and keys are the names given to the attributes of the specim object.
A similar write process could be applied to metadata."""
def specim_object_2_h5_file(h5_filename, h5_tree_dict, specim_object):
    with h5py.File(h5_filename, 'w', libver='latest') as f:
        for attribute_name, h5_hierarchy_item_path in h5_tree_dict.items():
            #print(attribute_name)
            dset = f.create_dataset(name=h5_hierarchy_item_path, 
                                            data = getattr(specim_object, attribute_name))            

def add_byte_order_to_envi_header(header_file_path, byte_order_value):
    """Function added to remedy lacking byte-order entry in header files of radiometric calibration data

    :param header_file_path: _description_
    :type header_file_path: _type_
    :param byte_order_value: _description_
    :type byte_order_value: _type_
    """
    # Read the existing ENVI header
    with open(header_file_path, 'r') as f:
        header_lines = f.readlines()

    # Look for the line where you want to add "byte order"
    for i, line in enumerate(header_lines):
        if line.startswith('byte order'):
            # If it already exists, update the value
            header_lines[i] = f'byte order = {byte_order_value}\n'
            break
    else:
        # If it doesn't exist, add it to the end of the header
        header_lines.append(f'byte order = {byte_order_value}\n')

    # Save the updated header
    with open(header_file_path, 'w') as f:
        f.writelines(header_lines)



def main(config, config_specim):

     # Defining the folder in which to put the data
    h5_folder = config['Absolute Paths']['h5_folder']
    
    is_already_processed = 0 != len([f for f in os.listdir(h5_folder) if not f.startswith('.')])
    
    # Exit
    if is_already_processed:
        print('Data has already been parsed')
        return
    
    # Function that calibrates data and reforms to h5 file format.
    dtype = config_specim.dtype_datacube
    transect_chunk_size = config_specim.lines_per_chunk
    cal_dir = config_specim.cal_dir
   
    mission_dir = config_specim.specim_raw_mission_dir

    mission_name = Path(mission_dir).name
    out_dir = config_specim.reformatted_missions_dir
    config_file_path = os.path.join(out_dir, config_specim.config_file_name)

    specim_object = Specim(mission_path=mission_dir, config=config)

    # Patterns for searching data cube files
    PATTERN_ENVI = '*.hdr'

    # Expects the captured data to reside in capture subfolder
    capture_dir = os.path.join(mission_dir, 'capture')

    # Path for searching (avoid explicit )
    search_path_envi = os.path.normpath(os.path.join(capture_dir, PATTERN_ENVI))
    envi_hdr_file_path = glob.glob(search_path_envi)[0]

    # Read out data
    spectral_image_obj = envi.open(envi_hdr_file_path)

    """import spectral as sp
    rgb_im = spectral_image_obj[:,:,]
    sp.imshow(spectral_image_obj, (73, 50, 24))"""

    # Read all meta data from header file (currently hard coded, but could be avoided I guess)
    class Metadata:
        pass

    metadata_obj = Metadata()
    # Could do this in iteration, but it would need to specify type (I think)
    metadata_obj.autodarkstartline = int(spectral_image_obj.metadata['autodarkstartline'])
    metadata_obj.n_lines = int(spectral_image_obj.metadata['lines'])
    metadata_obj.n_bands = int(spectral_image_obj.metadata['bands'])
    metadata_obj.n_pix = int(spectral_image_obj.metadata['samples'])
    metadata_obj.t_exp_ms = float(spectral_image_obj.metadata['tint'])
    metadata_obj.fps = float(spectral_image_obj.metadata['fps'])
    metadata_obj.description = spectral_image_obj.metadata['description']
    metadata_obj.file_type = spectral_image_obj.metadata['file type']
    metadata_obj.sensor_type = spectral_image_obj.metadata['sensor type']
    metadata_obj.acquisition_date = spectral_image_obj.metadata['acquisition date']
    metadata_obj.sensorid = spectral_image_obj.metadata['sensorid']
    metadata_obj.interleave = spectral_image_obj.metadata['interleave']
    metadata_obj.data_type = spectral_image_obj.metadata['data type']
    
    # Derived from knowledge of the sensor (sensor constants that could differ for other sensors)
    metadata_obj.binning_spatial = int(ACTIVE_SENSOR_SPATIAL_PIXELS/metadata_obj.n_pix)
    metadata_obj.binning_spectral = int(ACTIVE_SENSOR_SPECTRAL_PIXELS/metadata_obj.n_bands)

    # Allow Specim Methods to access metadata
    specim_object.metadata_obj = metadata_obj 
    # -

    # Based on the binning info, we can locate relevant calibration files, including 1) spectral, 2) geometric, 3) radiometric, and dark frame (from capture).
    #
    # We start with the spectral calibration:

    # Comprehensive band info (center, fwhm) is found in "cal_dir/wlcal<spectral binning>b_fwhm.wls"
    # Pattern follows structure
    PATTERN_BAND_INFO = '*'+ str(metadata_obj.binning_spectral) + 'b_fwhm.wls'
    # Linux-CLI search for file.
    search_path_bands = os.path.normpath(os.path.join(cal_dir, PATTERN_BAND_INFO))
    band_file_path = glob.glob(search_path_bands)[0]

    # Read the file as csv (although it has no columns)
    df_bands = pd.read_csv(band_file_path, header=None, sep = '\s+')
    df_bands.columns = ['Wavelength_nm', 'FWHM_nm']

    # Extract band data
    specim_object.wavelengths = np.array(df_bands['Wavelength_nm'])
    specim_object.fwhm = np.array(df_bands['FWHM_nm'])

    # -

    # We will here go through the loading of Specim geometric camera calibration. 
    # The first step is to load the angular Field-of-View file (AFOV) from the manufacturer. 
    # Then boresight angles and lever arms can be set, if relevant.

    # +
    # Pixel-directions is found in "cal_dir/FOV_****_<spatial binning>b.txt" 
    PATTERN_FOV = 'FOV*' + '_' +  str(metadata_obj.binning_spatial) + 'b.txt'

    # Search for fov file.
    search_path_fov = os.path.normpath(os.path.join(cal_dir, PATTERN_FOV))
    fov_file_path = glob.glob(search_path_fov)[0]

    # Calculate a camera model (zero boresight) based on FOV file.
    # Writes parameters to an *xml file specified by config

    # Read fov data (one angle for each pixel)
    df_fov = pd.read_csv(fov_file_path, header=None, sep = ',')

    df_fov.columns = ['pixel_nr', 'view_angle_deg', 'Unknown']

    specim_object.view_angles = np.array(df_fov['view_angle_deg'])
    
    # Simply converts view angles to camera parameters for a 1D line camera model
    param_dict = Specim.fov_2_param(fov = specim_object.view_angles)
    
    # param_dict are the full camera model parameters, but defaults rotation matrix and translation

    # User set rotation matrix
    R_hsi_body = config_specim.rotation_matrix_hsi_to_body

    r_zyx = RotLib.from_matrix(R_hsi_body).as_euler('ZYX', degrees=False)

    # Euler angle representation (any other would do too)
    param_dict['rz'] = r_zyx[0]
    param_dict['ry'] = r_zyx[1]
    param_dict['rx'] = r_zyx[2]

    # Vector from origin of HSI to body origin, expressed in body
    # User set
    t_hsi_body = config_specim.translation_body_to_hsi
    param_dict['tx'] = t_hsi_body[0]
    param_dict['ty'] = t_hsi_body[1]
    param_dict['tz'] = t_hsi_body[2]

    camera_calib_xml_dir = config['Absolute Paths']['calib_folder']

    file_name_xml = 'HSI_' + str(metadata_obj.binning_spatial) + 'b.xml'
    xml_cal_write_path = camera_calib_xml_dir + file_name_xml

    CalibHSI(file_name_cal_xml= xml_cal_write_path, 
                    mode = 'w', 
                    param_dict = param_dict)


    # Set value in config file:
    config.set('Relative Paths', 'hsi_calib_path', value = xml_cal_write_path)

    

    # Write the config object 
    with open(config_file_path, 'w') as configfile:
            config.write(configfile)
    # -

    


    # The next step is to read in radiometric frame.

    # +
    """Extract radiometric frame from dedicated file"""

    PATTERN_ENVI_CAL = '*_' +str(metadata_obj.binning_spectral) + 'x' +  str(metadata_obj.binning_spatial) + '.hdr'

    search_path_envi_cal = os.path.normpath(os.path.join(cal_dir, PATTERN_ENVI_CAL))
    ENVI_CAL_HDR_FILE_PATH = glob.glob(search_path_envi_cal)[0]

    # For allowing spectral module to read
    RAD_CAL_BYTE_ORDER = 0

    add_byte_order_to_envi_header(header_file_path=ENVI_CAL_HDR_FILE_PATH, byte_order_value=RAD_CAL_BYTE_ORDER)

    ENVI_CAL_IMAGE_FILE_PATH = ENVI_CAL_HDR_FILE_PATH.split('.')[0] + '.cal' # SPECTRAL does not expect this suffix by default

    # For some reason, the byte order is needed
    radiometric_image_obj = envi.open(ENVI_CAL_HDR_FILE_PATH, image = ENVI_CAL_IMAGE_FILE_PATH)

    cal_n_lines = int(radiometric_image_obj.metadata['lines'])
    cal_n_bands = int(radiometric_image_obj.metadata['bands'])
    cal_n_pix = int(radiometric_image_obj.metadata['samples'])

    # For calibration
    radiometric_frame = radiometric_image_obj[:,:,:].reshape((cal_n_pix, cal_n_bands))

    specim_object.radiometric_frame = radiometric_frame
    # -

    # The next step is to load the darkframes

    # +
    """1) Crop the hyperspectral data according to the start-stop lines. 2) Write datacube to appropriate directory"""
    # To ensure that the plots actually do appear in this notebook:
    # %matplotlib qt

    # Establish dark frame data (at end of recording)
    data_dark = spectral_image_obj[metadata_obj.autodarkstartline:metadata_obj.n_lines, :, :]
    dark_frame = np.median(data_dark, axis = 0)

    specim_object.dark_frame = dark_frame

    # -

    # The navigation data is given as messages is a *.nav file. Locate the file and parse it into a suitable format.

    # +
    # Extract the starting/stopping lines
    START_STOP_DIR = os.path.join(mission_dir, 'start_stop_lines')
    
    # Allow software to function if start_stop_lines not specified
    if not os.path.exists(START_STOP_DIR):
        os.mkdir(START_STOP_DIR)
        # Chunk according to chunk size
        start_lines = np.arange(start=0,
                                stop=metadata_obj.autodarkstartline, 
                                step=transect_chunk_size)

        stop_lines = np.arange(start=transect_chunk_size, 
                               stop=metadata_obj.autodarkstartline + transect_chunk_size, 
                               step=transect_chunk_size)
        
        stop_lines[-1] = metadata_obj.autodarkstartline # No need to georeference dark data

        # Make dataframe
        start_stop_data = {'line_start' : start_lines,
                           'line_stop' : stop_lines}
        
        df_start_stop = pd.DataFrame(start_stop_data)

        df_start_stop.to_csv(path_or_buf = os.path.join(START_STOP_DIR, 'start_stop_lines.txt'), sep='\s+')



    PATTERN_START_STOP = '*.txt'
    

    search_path_lines_start_stop = os.path.normpath(os.path.join(START_STOP_DIR, PATTERN_START_STOP))
    LINES_START_STOP_FILE_PATH = glob.glob(search_path_lines_start_stop)[0]

    df_start_stop = pd.read_csv(filepath_or_buffer=LINES_START_STOP_FILE_PATH, header=0, sep='\s+')



    # +
    # Now read the *.nav file
    NAV_PATTERN = '*.nav'

    search_path_nav = os.path.normpath(os.path.join(capture_dir, NAV_PATTERN))
    nav_file_path = glob.glob(search_path_nav)[0]

    date_string = metadata_obj.acquisition_date.split(': ')[1] # yyyy-mm-dd
    # Parse the position/orientation messages
    specim_object.read_nav_file(nav_file_path=nav_file_path, date = date_string)



    # -

    # Calculate the frame timestamps from sync data

    # +
    


    df_imu = pd.DataFrame(specim_object.imu_data)
    df_gnss = pd.DataFrame(specim_object.gnss_data)
    df_sync_hsi = pd.DataFrame(specim_object.sync_data)
    
    # Define the time stamps of HSI frames (special fix for Specim)
    sync_frames = df_sync_hsi['HsiFrameNum']
    sync_times = df_sync_hsi['TimestampAbs']
    hsi_frames = np.arange(metadata_obj.autodarkstartline)
    hsi_timestamps_total = interp1d(x = sync_frames, y = sync_times, fill_value = 'extrapolate')(x = hsi_frames)


    # For ease, let us interpolate position data to imu time (allows one timestamp for nav data)
    imu_time = df_imu['TimestampAbs']

    # Drop the specified regular clock time (as it is not needed)
    df_gnss = df_gnss.drop(columns=['TimestampClock'])

    # Interpolate each column in GNSS data based on 'imu_time'
    interpolated_values = {
        column: np.interp(imu_time, df_gnss['TimestampAbs'], df_gnss[column])
        for column in df_gnss.columns if column != 'TimestampAbs'
    }

    # Create a new DataFrame with the interpolated values
    df_gnss_interpolated = pd.DataFrame({'time': imu_time, **interpolated_values})

    # The position defined in geodetic coordinates
    lat = np.array(df_gnss_interpolated['Lat']).reshape((-1,1))
    lon = np.array(df_gnss_interpolated['Lon']).reshape((-1,1))
    ellipsoid_height = np.array(df_gnss_interpolated['AltMSL'] + df_gnss_interpolated['AltGeoid']).reshape((-1,1))

    # Assumes WGS-84 (default GNSS frame)
    x, y, z = pm.geodetic2ecef(lat = lat, lon = lon, alt = ellipsoid_height, deg=True)

    # Lastly, calculate the roll, pitch, yaw
    roll = np.array(df_imu['Roll']).reshape((-1,1))
    pitch = np.array(df_imu['Pitch']).reshape((-1,1))
    yaw = np.array(df_imu['Yaw']).reshape((-1,1))

    # Roll pitch yaw are stacked in attribute eul_zyx. The euler angles with rotation order ZYX are Yaw Pitch Roll
    specim_object.eul_zyx = np.concatenate((roll, pitch, yaw), axis = 1)

    # Position is stored as ECEF cartesian coordinates (mutually orthogonal axis) instead of spherioid-like lon, lat, alt
    specim_object.position_ecef = np.concatenate((x,y,z), axis = 1)
    specim_object.nav_timestamp = imu_time
    specim_object.t_exp_ms = metadata_obj.t_exp_ms

    # -

    # Last preprocessing step is writing to h5 files. 
    

    h5_dict_write = {'eul_zyx' : config['HDF.raw_nav']['eul_zyx'],
            'position_ecef' : config['HDF.raw_nav']['position'],
            'nav_timestamp' : config['HDF.raw_nav']['timestamp'],
            'radiance_cube': config['HDF.hyperspectral']['datacube'],
            't_exp_ms': config['HDF.hyperspectral']['exposuretime'],
            'hsi_timestamps': config['HDF.hyperspectral']['timestamp'],
            'view_angles': config['HDF.calibration']['view_angles'],
            'wavelengths' : config['HDF.calibration']['band2wavelength'],
            'fwhm' : config['HDF.calibration']['fwhm'],
            'dark_frame' : config['HDF.calibration']['darkframe'],
            'radiometric_frame' : config['HDF.calibration']['radiometricframe']}

    # Time to write all the data to a h5 file



    # # Chunking the recording to user defined sizes and writing it to disk

    # +
    # Define h5 file name
    h5_folder = config['Absolute Paths']['h5_folder']

    # Each line in start_stop defines a transect
    n_transects = df_start_stop.shape[0]
    
    print(f'There are {n_transects} transects')
    
    for transect_number in range(n_transects):
        start_line = df_start_stop['line_start'][transect_number]
        stop_line = df_start_stop['line_stop'][transect_number]

        n_chunks = int(np.ceil((stop_line-start_line)/transect_chunk_size))

        
        for chunk_number in range(n_chunks):
            chunk_start_idx = start_line + transect_chunk_size*chunk_number

            if chunk_number == n_chunks-1:
                chunk_stop_idx = stop_line
            else:
                chunk_stop_idx = chunk_start_idx + transect_chunk_size

            #print(f'Chunk from line {chunk_start_idx} to line {chunk_stop_idx} is being written')
            data_cube = spectral_image_obj[chunk_start_idx:chunk_stop_idx, :, :]
            # Calibration equation
            specim_object.radiance_cube = ( (data_cube - dark_frame)*radiometric_frame/(metadata_obj.t_exp_ms/1000) ).astype(dtype = dtype) # 4 Byte
            specim_object.hsi_timestamps = hsi_timestamps_total[chunk_start_idx:chunk_stop_idx]

            # Possible to name files with <PREFIX>_<time_start>_<Transect#>_<Chunk#>.h5
            h5_filename = h5_folder + mission_name + '_transectnr_' + str(int(transect_number)) + '_chunknr_' + str(int(chunk_number)) + '.h5'

            specim_object_2_h5_file(h5_filename=h5_filename, h5_tree_dict=h5_dict_write, specim_object=specim_object)









