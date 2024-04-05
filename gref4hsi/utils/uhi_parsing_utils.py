# Python Built-ins
import os
import glob
from datetime import datetime, timedelta, timezone
from pyproj import CRS, Transformer

# Third party libraries
from scipy.interpolate import griddata
import scipy.io as spio
from scipy.interpolate import interp1d
import h5py
import numpy as np
import pandas as pd
import pymap3d as pm
import rasterio
from rasterio.transform import Affine
from scipy.spatial.transform import Rotation as RotLib

# Lib specific utilites
from gref4hsi.utils.specim_parsing_utils import Specim
from gref4hsi.utils.geometry_utils import CalibHSI


"""Reader for the h5 file format in UHI context. The user provides h5 hierarchy paths as values and keys are the names given to the attributes """
class HyperspectralLite:
    def __init__(self, h5_filename, h5_tree_dict):
        with h5py.File(h5_filename, 'r', libver='latest') as self.f:
            for attribute_name, h5_hierarchy_item_path in h5_tree_dict.items():
                print(attribute_name)
                # Allow there to not be any attribute
                try:
                    h5_item = self.f[h5_hierarchy_item_path][()]
                except KeyError:
                    pass
                self.__setattr__(attribute_name, h5_item)




class TimeData:
    """Simple way of working with time referenced data. Supports unix and date_num at the moment"""
    def __init__(self, time = None, value = None, time_format = 'date_num'):
        if time is not None and any(time):
            if time_format == 'date_num':
                # Convert MATLAB-style datenum to Unix epoch
                converted_times = np.zeros((time.shape))

                for i, t in enumerate(time):
                    # Convert to Unix timestamp
                    unix_timestamp = datenum_to_unix_time(t)
                    converted_times[i] = unix_timestamp
                # Convert to Unix timestamp
                self.time = converted_times

            elif time_format == 'unix':
                self.time = time
            else:
                AttributeError
        else:
            self.time = time
            
        self.value = value
    def interpolate(self, time_interp):
        self.time_interp = time_interp
        self.value_interp = interp1d(x = self.time, y = self.value, kind='nearest', fill_value='extrapolate')(x=self.time_interp)
            

class NAV:
    """Very simple object allowing for setting time-referenced nav data. Entities end up having four attributes:
    time: Original time stamps
    value: Values
    time_interp: Time stamps interpolated (if interpolate has been run)
    value_interp: Values interpolated (if interpolate has been run)
    """
    def __init__(self):
        # Initialized to NONE to evoke exceptions
        self.roll = TimeData()
        self.pitch = TimeData()
        self.yaw = TimeData()
        self.pos_x = TimeData()
        self.pos_y = TimeData()
        self.lon = TimeData()
        self.lat = TimeData()
        self.pos_z = TimeData()
        self.altitude = TimeData()
    def interpolate(self, time_interp):
        # Iterate through all TimeData objects in NAV and call interpolate method
        for attr_name, time_data_obj in self.__dict__.items():
            if isinstance(time_data_obj, TimeData):
                time_data_obj.interpolate(time_interp)

# +
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def datenum_to_unix_time(datenum):
    """
    Convert Matlab datenum into unix time.
    :param datenum: Date in datenum format
    :return:        unixtime corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60

    dt = datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=float(seconds)) \
           - timedelta(days=366)
    
    # Defaults to selecting local time zone
    dt_utc = dt.replace(tzinfo=timezone.utc)
    unix_timestamp = datetime.timestamp(dt_utc)

    return unix_timestamp

def unix_to_date_and_time(unix_timestamps):
    """
    """
    formatted_dates = [datetime.utcfromtimestamp(ts).strftime('%d.%m.%Y') for ts in unix_timestamps]
    formatted_times = [datetime.utcfromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3] for ts in unix_timestamps]

    return formatted_dates, formatted_times

def write_immersion_compatible_format_from_nav(nav, unix_times_rov, time_offset, nav_filename):
        """
        The format of immersion is as follows:"""
        # Assuming that nav data is interpolated:
        nav.interpolate(time_interp=unix_times_rov)
        
        # Account for the offset in time
        hsi_times = unix_times_rov + time_offset

        # Format of data to be fed into Immersion
        # Date (dd.mm.yyyy), time (hh.mm.ss.ssss), Easting (m), Northing (m), Pitch (dd.ddd), roll (dd.ddd), Heading (dd.ddd), Depth (m.mm), Altitude (m.mm)
        formatted_dates, formatted_times = unix_to_date_and_time(hsi_times)
        data = {'Date': formatted_dates,
                'Time': formatted_times,
                'Easting': nav.pos_x.value_interp,
                'Northing': nav.pos_y.value_interp,
                'Pitch': nav.pitch.value_interp,
                'Roll': nav.roll.value_interp,
                'Heading': nav.yaw.value_interp,
                'Depth': nav.pos_z.value_interp,
                'Altitude': nav.altitude.value_interp}
        
        df = pd.DataFrame(data)
        df.to_csv(nav_filename, index=False, header=False)

def print_dict_tree_keys(dictionary, indent=0):
    for key, value in dictionary.items():
        print('  ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_tree_keys(value, indent + 1)

"""Writer (in append mode) for the h5 file format in UHI context. The user provides h5 hierarchy paths as values and keys are the names given to the attributes """
def write_data_to_h5_file(h5_filename, h5_dict_write, h5_dict_data):
    with h5py.File(h5_filename, 'a', libver='latest') as f:
        for key, h5_folder in h5_dict_write.items():
            # Check if the dataset exists
            if h5_folder in f:
                del f[h5_folder]

            dset = f.create_dataset(name=h5_folder, 
                                            data = h5_dict_data[key])
            
def immersion_filename_to_unix_time(immersion_file_name):

        list_names = immersion_file_name.split('_') # The year format should start with 202* or 201*

        result = [(entry,idx)  for (idx, entry) in enumerate(list_names) if entry.startswith(('202', '201'))]

        entry = result[0][0]
        idx = result[0][1]
        year_month_date = entry
        hour_min_sec = list_names[idx+1]

        # Combine the date and time strings
        combined_datetime_str = year_month_date + hour_min_sec

        # Parse the combined string into a datetime object
        dt = datetime.strptime(combined_datetime_str, '%Y%m%d%H%M%S')

        dt_utc = dt.replace(tzinfo=timezone.utc)
        unix_timestamp = datetime.timestamp(dt_utc)
        return unix_timestamp



def read_fov_file_to_param(LAB_CAL_DIR, binning_spatial):
    ## The absolute last step of the journey is to setup calibration the calibration file. We can read the one specified by Ecotone;
    
    PATTERN_FOV = 'FOV*' +  str(binning_spatial) + 'b.txt'
    # Search for fov file.
    search_path_fov = os.path.normpath(os.path.join(LAB_CAL_DIR, PATTERN_FOV))
    print(search_path_fov)
    FOV_FILE_PATH = glob.glob(search_path_fov)[0]

    fov_arr = np.array(pd.read_csv(FOV_FILE_PATH)['View_Angle_Deg'])
    param_dict = Specim.fov_2_param(fov = fov_arr)

    return param_dict

def set_camera_model(config, config_file_path, config_uhi, model_type, binning_spatial):
    LAB_CAL_DIR = config['General']['lab_cal_dir'] # Where the fov file is

    if model_type == 'cal_dir':
        param_dict = read_fov_file_to_param(LAB_CAL_DIR=LAB_CAL_DIR, binning_spatial=binning_spatial)
        # User set rotation matrix
        R_hsi_body = config_uhi.rotation_matrix_hsi_to_body

        r_zyx = RotLib.from_matrix(R_hsi_body).as_euler('ZYX', degrees=False)

        # Euler angle representation (any other would do too)
        param_dict['rz'] = r_zyx[0]
        param_dict['ry'] = r_zyx[1]
        param_dict['rx'] = r_zyx[2]

        # Vector from origin of HSI to body origin, expressed in body
        # User set
        t_hsi_body = config_uhi.translation_hsi_to_body
        param_dict['tz'] = t_hsi_body[0]
        param_dict['ty'] = t_hsi_body[1]
        param_dict['tz'] = t_hsi_body[2]
        
        file_name_xml = 'HSI_' + str(binning_spatial) + 'b.xml'
        CAMERA_CALIB_XML_DIR = config['Absolute Paths']['calib_folder']
        xml_cal_write_path = CAMERA_CALIB_XML_DIR + file_name_xml

        CalibHSI(file_name_cal_xml= xml_cal_write_path, 
                        config = config, 
                        mode = 'w', 
                        param_dict = param_dict)


        # Set value in config file and update:
        config.set('Relative Paths', 'hsi_calib_path', value = 'Input/Calib/' + file_name_xml)

        with open(config_file_path, 'w') as configfile:
                config.write(configfile)


    elif model_type == 'embedded':
        print('Should make support for this')
        pass
    else:
        pass


    

def read_nav_from_mat(mat_filename):
    """Function for reading mat data from the beast format into a NAV object"""
    mat_contents = {}
    mat_contents = loadmat(filename=mat_filename)

    m_att = mat_contents['ATTENTION']
    #m_att['SpotOnTime'][m_att['Note'] == 'UHI s1 start']

    

    
    # +
    """Cell defining all nav data of relevance"""

    # For ease, read position orientation data into structs
    nav = NAV()

    nav.roll = TimeData(time = mat_contents['TELEMETRY']['SpotOnTime'], 
                        value = mat_contents['TELEMETRY']['Roll'])

    nav.pitch = TimeData(time = mat_contents['TELEMETRY']['SpotOnTime'], 
                        value = mat_contents['TELEMETRY']['Pitch'])

    nav.yaw = TimeData(time = mat_contents['POSITION_RENAV']['SpotOnTime'], 
                        value = mat_contents['POSITION_RENAV']['USBLCourse']) # The USBL course

    nav.pos_x = TimeData(time = mat_contents['POSITION_RENAV']['SpotOnTime'], 
                        value = mat_contents['POSITION_RENAV']['x'])

    nav.pos_y = TimeData(time = mat_contents['POSITION_RENAV']['SpotOnTime'], 
                        value = mat_contents['POSITION_RENAV']['y'])

    nav.lat = TimeData(time = mat_contents['POSITION_RENAV']['SpotOnTime'], 
                        value = mat_contents['POSITION_RENAV']['Latitude'])

    nav.lon = TimeData(time = mat_contents['POSITION_RENAV']['SpotOnTime'], 
                        value = mat_contents['POSITION_RENAV']['Longitude'])

    nav.pos_z = TimeData(time = mat_contents['TELEMETRY']['SpotOnTime'], 
                        value = mat_contents['TELEMETRY']['Depth'])

    nav.altitude = TimeData(time = mat_contents['ALTIMETER']['SpotOnTime'], 
                        value = mat_contents['ALTIMETER']['Altitude']) # Altimeter readings
    
    return nav

def write_nav_data_to_h5(nav, time_offset, config, H5_FILE_PATH):
    
    # The time stamp used for writing
    nav_timestamp_rov = nav.yaw.time

    # Interpolate nav data to those times
    nav.interpolate(time_interp=nav_timestamp_rov)

    lon = nav.lon.value_interp
    lat = nav.lat.value_interp
    h = -nav.pos_z.value_interp # Depth is opposite of height

    # Calculate the entire data using the specified lon, lat alt
    x, y, z = pm.geodetic2ecef(lat = lat, lon = lon, alt = h, deg = True)

    # Roll pitch yaw
    roll = nav.roll.value_interp.reshape((-1,1))
    pitch = nav.pitch.value_interp.reshape((-1,1))
    yaw = nav.yaw.value_interp.reshape((-1,1))

    # Euler triplet
    eul_zyx = np.concatenate((roll, pitch, yaw), axis = 1)

    # Position ECEF
    position_ecef = np.concatenate((x.reshape((-1,1)),y.reshape((-1,1)),z.reshape((-1,1))), axis = 1)

    ##

    # Append the time offset to nav data so that all h5 data is synced to UHI clock
    hsi_synced_nav_timestamp_rov = nav_timestamp_rov + time_offset

    nav_dict_h5_folders = {'eul_ZYX' : config['HDF.raw_nav']['eul_ZYX'],
            'position_ecef' : config['HDF.raw_nav']['position'],
            'nav_timestamp' : config['HDF.raw_nav']['timestamp']}

    # Filling with the following data (all using the same time stamps)
    invalid_rows = np.isnan(eul_zyx[:,2])
    nav_dict_h5_data = {'eul_ZYX' : eul_zyx[~invalid_rows],
            'position_ecef' : position_ecef[~invalid_rows],
            'nav_timestamp' : hsi_synced_nav_timestamp_rov[~invalid_rows]}

    # Update config object with this info
    for key, value in nav_dict_h5_folders.items():
        config.set('HDF.raw_nav', key, value = value)
    # +
    write_data_to_h5_file(H5_FILE_PATH, h5_dict_write=nav_dict_h5_folders, h5_dict_data=nav_dict_h5_data)

def altimeter_data_to_point_cloud(nav, config_uhi, lat0, lon0, h0, true_time_hsi):
    """Converts the pose + altimeter data to a point cloud using the geometry of the range sensor

    :param nav: A Nav object containing UNIX time-stamped position, orientation and range measurements
    :type nav: Nav object
    :param config_uhi: A configuration object containing the altimeter's transform (rotation matrix and lever arm)
    :type config_uhi: Named tuple
    :param lat0: Latitude for north-east-down linearization
    :type lat0: float
    :param lon0: Longitude for north-east-down linearization
    :type lon0: float
    :param h0: Latitude for north-east-down linearization
    :type h0: float
    :param true_time_hsi: The time stamps of the hyperspectral imager
    :type true_time_hsi: float
    :return: point cloud of 
    :rtype: _type_
    """

    # Number of altimeter samples
    n = nav.altitude.value.size

    # Interpolate all navigation data to the time of altimeter data
    # TODO: avoid common interpolation of yaw, pitch, roll
    nav.interpolate(time_interp=nav.altitude.time)

    # Create augmented vector allowing 4x4 transformation matrices
    alt_vec = np.zeros((4, n))
    alt_vec[2] = nav.altitude.value # Ranges are captured along z-axis (by definition)
    alt_vec[3] = 1

    # Preallocate transform from altimeter to body frame
    T_alt_body = np.zeros((4,4))
    T_alt_body[3,3] = 1

    # User specified altimeter position, orientation on the body
    R_alt_body = config_uhi.rotation_matrix_alt_to_body
    t_alt_body = config_uhi.translation_alt_to_body

    # Insert in transormation matrix
    T_alt_body[:3,:3] = R_alt_body
    T_alt_body[:3, 3] = t_alt_body

    # Transform ranges to body frame
    alt_vec_body = np.matmul(T_alt_body.reshape((4,4)), alt_vec)

    # -------------------------Define transformation from body to NED-----------------------------
    roll = nav.roll.value_interp.reshape((-1,1))
    pitch = nav.pitch.value_interp.reshape((-1,1))
    yaw = nav.yaw.value_interp.reshape((-1,1))

    eul_zyx_alt = np.concatenate((roll, pitch, yaw), axis = 1)


    lat_alt = nav.lat.value_interp
    lon_alt = nav.lon.value_interp
    h_alt = -nav.pos_z.value_interp

    # User specified values
    lon0 = config_uhi.lon_lat_alt_origin[0]
    lat0 = config_uhi.lon_lat_alt_origin[1]
    h0 = config_uhi.lon_lat_alt_origin[2]

    # Transform to local NED for working with vectors
    north, east, down = pm.geodetic2ned(lat=lat_alt, lon = lon_alt, h = h_alt, lat0 = lat0, lon0=lon0, h0 = h0, deg = True)

    # Account for the motion
    R_mats_body_ned = RotLib.from_euler("ZYX", np.flip(eul_zyx_alt, axis=1), degrees=True).as_matrix()
    t_body_ned = np.vstack((north, east, down)).T

    # Set transformation matrices (one for each timestamp)
    T_body_ned = np.zeros((n, 4,4))
    T_body_ned[:, 3,3] = 1
    T_body_ned[:, :3,:3] = R_mats_body_ned
    T_body_ned[:, :3, 3] = t_body_ned

    # ----------------------------------------------------------------------------------------------

    # Transform ranges to NED
    alt_vec_ned = np.einsum('ijk, ki->ji', T_body_ned, alt_vec_body)

    # Transpose vectors (just out of preference)
    altimeter_point_cloud = alt_vec_ned[0:3,:].T

    # Select the points from an appropriate time interval
    crit_1 = nav.altitude.time < true_time_hsi.max()
    crit_2 = nav.altitude.time > true_time_hsi.min()

    points_altimeter_transect = altimeter_point_cloud[(crit_1) & (crit_2)]

    # Still the points are given in a form of NED

    # NED is no frame for GIS, but an OK frame for certain vector operations.
    # The above linearization is only accurate for small surveys such as those in UHI

    return points_altimeter_transect

def write_point_cloud_to_dem(points, config, resolution_dem, lon0, lat0, h0, method='nearest', smooth_DEM=True):
    
    # The name of the digital elevation model
    output_dem_path = config['Absolute Paths']['dem_path']

    # The padding (ensuring that all rays hit the model)
    pad_xy = float(config['General']['max_ray_length'])

    x_coords = points[:,0]
    y_coords = points[:,1]
    z_values = points[:,2]

    #-----------------------------------------------------Transform Extent and determing resolution, geotransform and grid--------------------------------------------------------
    
    # Determine the min/max coordinates with padding in the NED plane, and warp to projected grid.
    x_min, x_max = x_coords.min() - pad_xy, x_coords.max() + pad_xy

    y_min, y_max = y_coords.min() - pad_xy, y_coords.max() + pad_xy

    extent_ned = np.array([[x_min, y_min],
                          [x_min, y_max],
                          [x_max, y_min],
                          [x_max, y_max]])

    # Transform to wgs 84, aka epsg 4326
    lat_ext, lon_ext, h_ext = pm.ned2geodetic(n = extent_ned[:,0], e = extent_ned[:,1], d = extent_ned[:,1]*0 + z_values.mean(), lat0 = lat0, lon0=lon0, h0 = h0, deg = True)
    
    # Transform to desired CRS
    epsg_wgs84 = 4326
    dem_epsg = int(config['Coordinate Reference Systems']['dem_epsg'])

    # Define transformer
    crs_84 = CRS.from_epsg(epsg_wgs84)
    crs_dem = CRS.from_epsg(dem_epsg)
    transformer = Transformer.from_crs(crs_84, crs_dem)

    # Transform extent
    (x, y, z) = transformer.transform(xx=lat_ext, yy=lon_ext, zz=h_ext)

    if crs_dem.is_geocentric:

        res_x = resolution_dem*(x.max() - x.min())/(x_max - x_min)
        res_y = resolution_dem*(y.max() - y.min())/(y_max - y_min)
        transform = Affine(a = res_y,
                        b = 0,
                        c = y.min(),
                        d = 0,
                        e = -res_x,
                        f = x.max())
        # Setup mesh grid
        grid_y, grid_x = np.meshgrid(np.arange(y.min(), y.max(), res_y),
                                np.arange(x.min(), x.max(), res_x))
    elif crs_dem.is_projected:
        # North is then Y
        res_x = resolution_dem*(x.max() - x.min())/(y_max - y_min)
        res_y = resolution_dem*(y.max() - y.min())/(x_max - x_min)
        transform = Affine(a = res_x,
                        b = 0,
                        c = x.min(),
                        d = 0,
                        e = -res_y,
                        f = y.max())
        # Setup mesh grid
        grid_x, grid_y = np.meshgrid(np.arange(x.min(), x.max(), res_x),
                                np.arange(y.min(), y.max(), res_y))
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------Transform point cloud------------------------------------------------------------------------------------------
    
    # Transform to wgs 84, aka epsg 4326
    lat_pc, lon_pc, h_pc = pm.ned2geodetic(n = points[:,0], e = points[:,1], d = points[:,2], lat0 = lat0, lon0=lon0, h0 = h0, deg = True)
    # Transform to target system
    (x_pc, y_pc, z_pc) = transformer.transform(xx=lat_pc, yy=lon_pc, zz=h_pc)

    points_pc_dem_crs = np.concatenate((x_pc.reshape((-1,1)),
                                        y_pc.reshape((-1,1)),
                                        z_pc.reshape((-1,1))), axis = 1)
    

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Interpolate z-data to grid

    if method == 'nearest':
        grid_z = griddata(points_pc_dem_crs[:, :2], points_pc_dem_crs[:,2], (grid_x, grid_y), method='nearest')
    elif method == 'IDW':
        import pyinterp
        mesh = pyinterp.RTree()

        mesh.packing(points_pc_dem_crs[:, :2], points_pc_dem_crs[:,2])

        idw, neighbors = mesh.inverse_distance_weighting(
        np.vstack((grid_x.ravel(), grid_y.ravel())).T,
        within=False,  # Extrapolation is forbidden
        k=11,  # We are looking for at most 11 neighbors
        num_threads=0)

        grid_z = idw.reshape(grid_x.shape)


    if smooth_DEM == True:
        from scipy import ndimage
        sigma_x = 4
        sigma_y = 4
        grid_z = ndimage.gaussian_filter(grid_z, sigma=(sigma_y, sigma_x))

    

    with rasterio.open(output_dem_path, 'w', driver='GTiff', height=grid_z.shape[0],
                    width=grid_z.shape[1], count=1, dtype='float32', crs='EPSG:' + str(dem_epsg),
                    transform=transform) as dst:
        dst.write(grid_z, 1)




def uhi_beast(config, config_uhi):
    MISSION_PATH = config['General']['mission_dir'] # Where h5 data is 
    

    # For writing
    config_file_path = os.path.join(MISSION_PATH, 'configuration.ini')

    SPATIAL_PIXELS = 1936 # Same for almost all UHI
    

    INPUT_DIR = MISSION_PATH + 'Input/'

    h5_folder = config['Absolute Paths']['h5_folder']
    H5_PATTERN = '*.h5'

    MAT_DIR = INPUT_DIR
    MAT_PATTERN = '*.mat'

    # Locate mat file with nav data 
    search_path_mat = os.path.normpath(os.path.join(MAT_DIR, MAT_PATTERN))
    MAT_FILE_PATHS = glob.glob(search_path_mat)
    MAT_PATH = MAT_FILE_PATHS[0] # Assuming there is only one

    nav = read_nav_from_mat(mat_filename=MAT_PATH)

    

    # Search the h5 folder (these are the files to iterate)
    search_path_h5 = os.path.normpath(os.path.join(h5_folder, H5_PATTERN))
    H5_FILE_PATHS = glob.glob(search_path_h5)

    number_of_h5_files = len(H5_FILE_PATHS)

    h5_dict_read = {'radiance_cube': config['HDF.hyperspectral']['datacube'],
            'hsi_frames_timestamp': config['HDF.hyperspectral']['timestamp'],
            'fov': config['HDF.calibration']['fov'],
            'wavelengths' : config['HDF.calibration']['band2wavelength'],
            #'rgb_frames' : config['HDF.rgb']['rgb_frames'],
            #'rgb_timestamp' : config['HDF.rgb']['rgb_timestamp']
            }
    
    time_offset = config_uhi.time_offset_sec
    lon0, lat0, alt0 = config_uhi.lon_lat_alt_origin

    # from gref4hsi.utils.photogrammetry_utils import Photogrammetry
    # agisoft_object = Photogrammetry(project_folder = MISSION_PATH, software_type='agisoft')
    # TODO:: Add support for concurrent camera given that images are contained in a h5 folder. 
    # Should interface towards ODM as well

    for h5_index in range(number_of_h5_files):
        H5_FILE_PATH = H5_FILE_PATHS[h5_index]

        # Read the specified entries
        hyp = HyperspectralLite(h5_filename=H5_FILE_PATH, h5_tree_dict=h5_dict_read)

        if h5_index == 0:
            # Camera model is set once, assuming same spatial binning throughout
            binning_spatial = int(np.round(SPATIAL_PIXELS/hyp.radiance_cube.shape[1]))
            # Sets camera model with user specified boresight ...
            set_camera_model(config=config, 
                             config_file_path = config_file_path, 
                             config_uhi=config_uhi, 
                             model_type = 'cal_dir', 
                             binning_spatial = binning_spatial)

        true_time_hsi = hyp.hsi_frames_timestamp - time_offset

        # Interpolate nav to roll time, IMU time (considered nav timestamp)

        ## write nav data to h5 file
        write_nav_data_to_h5(nav, time_offset, config, H5_FILE_PATH)
        
        # Build a point cloud
        point_cloud_altimeter = altimeter_data_to_point_cloud(nav=nav, 
                                                              config_uhi=config_uhi, 
                                                              true_time_hsi = true_time_hsi, 
                                                              lon0=lon0, 
                                                              lat0=lat0, 
                                                              h0=alt0)
        if h5_index == 0:
            point_cloud_altimeter_total = point_cloud_altimeter
        else:
            point_cloud_altimeter_total = np.append(point_cloud_altimeter_total, point_cloud_altimeter, axis = 0)
        


        """# If desirable to write to Agisoft type format
        if config_uhi.agisoft_process:
            nav_rgb = nav
            nav_rgb.interpolate(time_interp=hyp.rgb_timestamp)

            # Prepare for SfM/photogrammetry processing
            try:
                agisoft_object.export_rgb_from_h5(h5_folder=H5_FILE_PATH, 
                                            rgb_image_cube = hyp.rgb_frames, 
                                            nav_rgb = nav_rgb,
                                            rgb_write_dir=config['Absolute Paths']['rgbimagedir'],
                                            pos_acc = np.array([1, 1, 0.05]), 
                                            rot_acc = np.array([1, 1, 5]))
            except:
                pass # If no RGB data, move forward"""
        
        
        
        print(H5_FILE_PATH)
    
    # Use the total point cloud to make a DEM
    

    if config_uhi.agisoft_process:
        agisoft_object.load_photos_and_reference()





    write_point_cloud_to_dem(point_cloud_altimeter_total, 
                                 config, 
                                 resolution_dem = config_uhi.resolution_dem, 
                                 lon0=lon0, 
                                 lat0=lat0, 
                                 h0=alt0)


        
        

        