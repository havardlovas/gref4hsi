from spectral import envi
import pymap3d as pm
import pyproj
import rasterio
from pathlib import Path
import json
import h5py
import numpy as np

import os
import glob
import shutil

from massipipe.pipeline import PipelineProcessor
from scipy.spatial.transform import Rotation as RotLib
from gref4hsi.utils.geometry_utils import CalibHSI


# Helper function
def _get_geoid_undulation(src, latitude, longitude):
    """Extracts geoid undulation from a GeoTIFF at a given point.

    Args:
        src: Rasterio dataset object.
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.

    Returns:
        Geoid undulation in meters.
    """

    # Transform coordinates to raster pixel coordinates
    if src.crs.is_projected:
        geodetic_epsg = 4326
        transformer = pyproj.Transformer.from_crs(geodetic_epsg, src.crs, always_xy=True)
        x, y, _ = transformer.transform(longitude, latitude, 0*np.ones(longitude.shape))
        row, col = src.index(x, y)

    else:
        row, col = src.index(longitude, latitude)

    # Extract pixel value (geoid undulation)
    geoid_undulation = src.read(1)[row, col]

    return geoid_undulation

# Defining a writer for the relevant attributes
def _img_object_2_h5_file(h5_filename, h5_tree_dict, img_object):
    with h5py.File(h5_filename, 'w', libver='latest') as f:
        for attribute_name, h5_hierarchy_item_path in h5_tree_dict.items():
            dset = f.create_dataset(name=h5_hierarchy_item_path, 
                                            data = getattr(img_object, attribute_name))
# Define metadata
# Read all meta data from header file (currently hard coded, but could be avoided I guess)
# An instance of ResononImage will be created for each image
class ResononImage:
    def __init__(self, config, config_file, config_specim_preprocess, hsi_mission_folder, processing_lvl, afov_deg=None):
        self.config = config
        self.config_file = config_file
        self.config_specim_preprocess = config_specim_preprocess
        self.hsi_mission_folder = hsi_mission_folder
        self.afov = np.deg2rad(afov_deg) # in radians
        self.fov_file = ''
        self.geoid_path = config['Absolute Paths']['geoid_path']
        self.processing_lvl = processing_lvl
        
        self.pre_process_mission_folder()
        
        preprocess_lvl_dict = {"0": "0_raw",
                       "1a": "1a_radiance",
                       "2a": "2a_reflectance",
                       "2b": "2b_reflectance_gc"} # key word, subfolder name
        
        # Locate the paths to the ENVI data cubes
        self.capture_dir = os.path.join(hsi_mission_folder,
                           "pre-processed",
                           preprocess_lvl_dict[processing_lvl])
        
        # Locate the nav data
        self.nav_dir =  os.path.join(hsi_mission_folder,
                           "pre-processed",
                           'imudata')
        
        self.n_imgs = len([f for f in os.listdir(self.nav_dir) if not f.startswith('.')]) # Fixes hidden file problem
        
        
        

        
        
        
        
        
    
    def pre_process_mission_folder(self):
        # First copy the 0_raw and calibration folder into a pre-processed folder to avoid too many folders at highest level
        # Folder structure is hardcoded
        try:
            shutil.copytree(os.path.join(self.hsi_mission_folder, "0_raw"), 
                            os.path.join(self.hsi_mission_folder, "pre-processed", "0_raw"))
            shutil.copytree(os.path.join(self.hsi_mission_folder, "calibration"), 
                            os.path.join(self.hsi_mission_folder, "pre-processed", "calibration"))
            shutil.copy(os.path.join(self.hsi_mission_folder, "config.seabee.yaml"), 
                            os.path.join(self.hsi_mission_folder, "pre-processed", "config.seabee.yaml"))
        except:
            return # Presumably it already has been processed
        
        # Perform the pre-processing from pre-processed folder
        processor = PipelineProcessor(os.path.join(self.hsi_mission_folder, "pre-processed"))
        processor.run(mosaic_geotiffs = False)
        # Remove the copied config file to avoid any problems
        os.remove(os.path.join(self.hsi_mission_folder, "pre-processed", "config.seabee.yaml"))
    
    def format_2_gref4hsi(self):
        """Run through and process the hyperspectral data to a format for """
        # Iterate
        # For legacy reasons the software expects data to come in the h5 file format. 
        
        # Defining the mapping from the ResononImage object to the h5 file
        self.h5_dict_write = {'eul_zyx' : self.config['HDF.raw_nav']['eul_zyx'],
                    'position_ecef' : self.config['HDF.raw_nav']['position'],
                    'nav_timestamp' : self.config['HDF.raw_nav']['timestamp'],
                    'datacube': self.config['HDF.hyperspectral']['datacube'],
                    't_exp_ms': self.config['HDF.hyperspectral']['exposuretime'],
                    'hsi_timestamps': self.config['HDF.hyperspectral']['timestamp'],
                    'wavelengths' : self.config['HDF.calibration']['band2wavelength']}
        # Defining the folder in which to put the data
        h5_folder = self.config['Absolute Paths']['h5_folder']
        
        is_already_processed = 0 != len([f for f in os.listdir(h5_folder) if not f.startswith('.')])
        
        # Exit
        if is_already_processed:
            return

        for i in range(self.n_imgs):
            print(self.n_imgs)
            img_id = f"*{i:03d}*"
            transect_nr = f"{i:03d}"

            search_path_nav = os.path.normpath(os.path.join(self.nav_dir, img_id + '.json'))
            search_path_envi_hdr = os.path.normpath(os.path.join(self.capture_dir, img_id + '.bip.hdr'))

            if self.processing_lvl != '0':
                envi_hdr_file = glob.glob(search_path_envi_hdr)[0]

            else:
                # This will occur if user chooses processing level 0
                envi_hdr_file = [str(item) for item in Path(self.capture_dir).rglob(f'*_{i+1}.bil.hdr')][0]

            nav_file = glob.glob(search_path_nav)[0]

            self.process_envi_hdr(envi_hdr_file)
            self.generate_camera_model(fov_file = self.fov_file, afov = self.afov)
            self.process_nav_json(nav_file, self.geoid_path)
            
            # Same name as ENVI file, just different extention
            transect_name = os.path.basename(envi_hdr_file).split(sep = '.')[0] # To remove suffix
            h5_filename = h5_folder + transect_name + '.h5'
            
            # Write to h5 file
            _img_object_2_h5_file(h5_filename=h5_filename, 
                                     h5_tree_dict=self.h5_dict_write, 
                                     img_object=self)

            print(f"Image nr {i:03d}")
        
        
    def process_envi_hdr(self, envi_hdr_file_path):
        """parse with an envi HDR"""
        
        self.spectral_image_obj = envi.open(envi_hdr_file_path)
        # Verbosely written out so that user can see available meta
        """self.n_lines = int(self.spectral_image_obj.metadata['lines'])
        self.n_bands = int(self.spectral_image_obj.metadata['bands'])
        self.n_pix = int(self.spectral_image_obj.metadata['samples'])
        self.binning_spatial = int(self.spectral_image_obj.metadata['sample binning'])
        self.binning_spectral = int(self.spectral_image_obj.metadata['spectral binning'])
        
        self.file_type = self.spectral_image_obj.metadata['file type']
        self.hdr_offset = int(self.spectral_image_obj.metadata['header offset'])
        self.interleave = self.spectral_image_obj.metadata['interleave']
        self.byte_order = self.spectral_image_obj.metadata['byte order']
        self.t_exp_ms = float(self.spectral_image_obj.metadata['shutter'])
        self.t_exp_unit = self.spectral_image_obj.metadata['shutter units']
        self.fps = float(self.spectral_image_obj.metadata['framerate'])
        self.wlen_unit = self.spectral_image_obj.metadata['wavelength units']
        self.gain = float(self.spectral_image_obj.metadata['gain'])

        self.direction = self.spectral_image_obj.metadata['direction']
        self.flip_radiometric_calibration = self.spectral_image_obj.metadata['flip radiometric calibration']
        self.timestamp = self.spectral_image_obj.metadata['timestamp'] # A single timestamp 

        self.target = self.spectral_image_obj.metadata['target']
        self.wavelengths = np.array(self.spectral_image_obj.metadata['wavelength']).astype(np.float32)
        """
        self.datacube = self.spectral_image_obj[:,:,:] # Load the datacube
        # Load metadata
        for key, value in self.spectral_image_obj.metadata.items():
            # Replace spaces with underscores
            key = key.replace(" ", "_")
            setattr(self, key, value)
        
        self.t_exp_ms = self.shutter
        # For some reason interpreted as strings
        self.wavelengths = np.array(self.wavelength).astype(np.float64)

        
    def generate_camera_model(self, fov_file = '', afov = None):
        """Generate camera model for the particular mission. Assuming that spatial binning is constant"""
        
        # Naming the file. If there is a desire to override the autogenerated calib-file, the user can create a 
        #"HSI_1b.xml", "HSI_2b.xml" file etc depending on the binning level and placing it on the top level
        
        
        file_name_xml = 'HSI_' + str(self.sample_binning) + 'b.xml'
        
        # First check if it exists within tree:
        f_list = [f.parent for f in Path(self.hsi_mission_folder).rglob(file_name_xml)]
        
        if len(f_list) == 1:
            # Set value in config file:
            self.config.set('Absolute Paths', 'hsi_calib_path', value = 
                            os.path.join( str(f_list[0]), file_name_xml))
            # Write the config object 
            with open(self.config_file, 'w') as configfile:
                    self.config.write(configfile)
                
            return
        
        elif len(f_list) == 0:
            camera_calib_xml_dir = self.config['Absolute Paths']['calib_folder'] # Where we put geometric calib files
            xml_cal_write_path = os.path.join(camera_calib_xml_dir, file_name_xml)
            
            self.config.set('Absolute Paths', 'hsi_calib_path', value = xml_cal_write_path)
            # Write the config object 
            with open(self.config_file, 'w') as configfile:
                    self.config.write(configfile)
        
        
        
        
        if fov_file == '':
            # This means that there is no manufacturer precise calibration for the FOV. Assume a pinhole model:
            # See https://github.com/havardlovas/gref4hsi for info
            width = int(self.samples)
            cx = width/2
            f = width / (2*np.tan(afov/2))
            k1, k2, k3 = 0, 0, 0

        else:
            raise NameError

        # We define the rotations/translations from the user input:

        # User set rotation matrix
        R_hsi_body = self.config_specim_preprocess.rotation_matrix_hsi_to_body

        r_zyx = RotLib.from_matrix(R_hsi_body).as_euler('ZYX', degrees=False)

        # Euler angle representation (any other would do too)
        rotation_z = r_zyx[0]
        rotation_y = r_zyx[1]
        rotation_x = r_zyx[2]

        # Vector from origin of HSI to body origin, expressed in body
        # User set
        t_hsi_body = self.config_specim_preprocess.translation_body_to_hsi
        translation_x = t_hsi_body[0]
        translation_y = t_hsi_body[1]
        translation_z = t_hsi_body[2]



        param_dict = {'rx':rotation_x,
                      'ry':rotation_y,
                      'rz':rotation_z,
                      'tx':translation_x,
                      'ty':translation_y,
                      'tz':translation_z,
                      'f': f,
                      'cx': cx,
                      'k1': k1,
                      'k2': k2,
                      'k3': k3,
                      'width': width}

        



        
        # Write to file
        CalibHSI(file_name_cal_xml= xml_cal_write_path, 
                            mode = 'w', 
                            param_dict = param_dict)

        
        
    
# The processing in massipipe nicely renders the necessary navigation data in a json format
    def process_nav_json(self, json_file, geoid_path):
        """Reads"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Unravel data into variables
        lat = np.array(data['latitude'])
        lon = np.array(data['longitude'])
        
        # Allow sampling of geoid height
        with rasterio.open(geoid_path, 'r') as src:
            geoid_height = _get_geoid_undulation(src, lat, lon)
        
        alt_msl = np.array(data['altitude']) # Is above geoid
        
        alt_ell = alt_msl + geoid_height.reshape(alt_msl.shape) # above ellipsoid
        
        roll = np.array(data['roll'])
        pitch = np.array(data['pitch'])
        yaw = np.array(data['yaw'])
        
        timestamp = np.array(data['time']).reshape((-1, 1)) # Only relative time
        
        # If your nav system gave you geodetic positions, convert them to earth centered earth fixed (ECEF). Make sure to use ellipsoid height (not height above mean sea level (MSL) aka geoid)
        x, y, z = pm.geodetic2ecef(lat = lat, lon = lon, alt = alt_ell, deg=True)

        # Roll pitch yaw are ordered with in an unintuitive attribute name eul_zyx. The euler angles with rotation order ZYX are Yaw Pitch Roll
        self.eul_zyx = np.vstack((roll, pitch, yaw)).T

        # The ECEF positions
        self.position_ecef = np.vstack((x,y,z)).T

        self.nav_timestamp = timestamp
        self.hsi_timestamps = timestamp


        
        
        
        


