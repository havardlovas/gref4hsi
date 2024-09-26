# gref4hsi - a toolchain for the georeferencing and orthorectification of hyperspectral pushbroom data. 
This software was made with a special emphasis on georeferencing and orthorectification of hyperspectral imagery from drones and underwater robots. However, it is equally simple to use for airborne data, and probably even for satellite imagery (although modified approaches including analytical ellipsoid intersection may be better). There is also a coregistration module (at beta stage) which, given an accurate RGB orthomosaic, can optimize geometric parameters (static or time-varying) to align the data.
The functionality in bullet form is:
* georeference.py: The software currently supports direct georeferencing through CPU-accelerated ray tracing of push broom measurements onto terrain files including 3D triangular meshes (\*.ply), 2.5D raster DEM/DSM (e.g. \*.tif) and geoid models.
* orthorectification.py: The software performs orthorectification (a form of image resampling) to any user specified projection (by EPSG code) and resolution. The default resampling strategy is north-east oriented rasters, but the software does support memory-optimally oriented rasters (smallest bounding rectangle). When you resample, you essentially figure out where to put measurements in a geographic grid. In the software, we resample the data cube, an RGB composite of the cube, but also ancillary data like intersection/sun geometries, pixel coordinates (in spatial dimension of imager) and timestamps.
* coregistration.py: Given a reference RGB orthomosaic (e.g. from photo-matching) covering the area of the hyperspectral image, the coregistration module does SIFT-matching with the RGB composite (handling any differences in projection and raster transforms). The module also has an optimization component for using the matches to minimize the reprojection error. The user can select/toggle which parameters to optimize, including boresight angles, lever arms, camera model parameters or time-varying errors in position/orientation. Notably the module requires that the pixel-coordinates and timestamps are resampled, as done automatically in the orthorectification step.

This README is a bit verbose, but is meant to act as a "tutorial" as well, and I recommend trying to understand as much as possible.

## Installation instructions for Linux:
The easiest approach is to use conda/mamba to get gdal and pip to install the package with the following commands (has been tested for python 3.8-3.10):
```
conda create -n my_env python=3.10 gdal rasterio
conda activate my_env
pip install gref4hsi 
```

## Installation instructions for Windows:
Seamless installation for windows is, to my experience a bit more challenging because of gdal and rasterio. For python 3.10 you could run
```
conda create -n my_env python=3.10
conda activate my_env
pip install GDAL‑3.4.3‑cp310‑cp310‑win_amd64.whl
pip install rasterio‑1.2.10‑cp310‑cp310‑win_amd64.whl
pip install gref4hsi 
```
The wheel files are provided at [Christoph Gohlke's page](https://www.lfd.uci.edu/~gohlke/pythonlibs/). Examples for python 3.8 are given in the dependencies folder of the Github repo.

## Getting started:
To run the code you will most likely need to re-format your raw acquired hyperspectral data and navigation data. To start with I'd recommend creating a top folder "<mission_dir>" (e.g. a folder "/processed" under your sensor-proprietary raw data). An example is shown below.

```
<mission_dir>/
├── configuration.ini
├── Input
│   ├── Calib
│   │   └── HSI_2b.xml
│   └── H5
│       ├── 2022-08-31_0800_HSI_transectnr_0_chunknr_0.h5
│       ├── 2022-08-31_0800_HSI_transectnr_0_chunknr_1.h5
│       ├── 2022-08-31_0800_HSI_transectnr_1_chunknr_0.h5
```


You minimally need the following to successfully run georeferencing: a configuration file ("\*.ini"), a terrain model, a camera model ("\*.xml") and h5/hdf file (datacube + navigation data). To appropriately format/create these, I recommend creating a parser. An example parser for a specim hyperspectral imager + navigation system is added under gref4hsi/utils/specim_parsing_utils.py

### Configuration file ("\*.ini"). 
It is recommended to just go with the template and maybe change a few minor things, depending on need. A template is given in the Github repo under data/config_examples/configuration_template.ini. The file contains comments describing the entries. Take a careful look at all comments containing "Change" as these are relevant to adjust depending on your setup. If your imager records h5 data, adjusting the h5 paths (sections 'HDF.xxxx') is relevant too. Morover, we interface with the configuration in a dictionary-style manner:

```
# Make an object and read the config file for the mission
config = configparser.ConfigParser()
config.read(config_file_mission)

# Access a specific configuration, e.g. the calibrated camera model of the imager
hsi_calib_path = config['Absolute Paths']['hsi_calib_path'] # config[<section>][<config-entry>]
```


### Calibration file ("\*.xml")
The *.xml file is our chosen camera model file and looks like this:
```
<?xml version="1.0" encoding="utf-8"?>
<calibration>
    <rx>0.0</rx>
    <ry>0.0</ry>
    <rz>-1.5707963267948966</rz>
    <tx>0</tx>
    <ty>0</ty>
    <tz>0</tz>
    <f>754.9669285377008</f>
    <cx>255.00991757530994</cx>
    <k1>-72.31892156971124</k1>
    <k2>-389.57799574674084</k2>
    <k3>4.075384496065511</k3>
    <width>512</width>
</calibration>
```
rx, ry and rz are the boresight angles in radians, while tx, ty and tz are lever arms in the vehicle body frame. They are used to transform vectors from the HSI camera frame to the vehicle body frame through:
```
rot_hsi_ref_eul = np.array([rz, ry, rx])

R = RotLib.from_euler(seq = 'ZYX', angles = rot_hsi_ref_eul, degrees=False).as_matrix() # Rotation matrix from HSI to body

t = np.array([tx, ty, tz])

X_hsi = np.array([x, y, z]) # some vector

X_body =  R*X_hsi + t# the vector expressed in the body frame 
```
In our computation we use a frame convention akin to the camera frames in computer vision. This means that x-right, y-backward, and z-downward for a well aligned camera. In contrast the vehicle body frame follows the roll-pitch-yaw convention with x-forward, y-starboard, z-downward. This is why the rz is by default $\pm \pi/2$. The easiest if you are unsure of whether your scanner is flipped is to swap the sign rz.



Moreover, the f represent the camera's focal length in pixels, width is the number of spatial pixels, while cx is the principal pixel. Often cx=(width+1)/2, e.g. if you have 5 pixels u = 0.5,1.5,..,4.5 the middle pixel is cx=2.5. In other words, our convention is to assign the cell centres of pixels as whole integers (starting at 1), in contrast to OpenCV which starts at 0.5. The k1, k2 are radial distortion coefficients, while k3 is a tangential coefficient. The camera model is adapted from [Sun et al. (2016)](https://opg.optica.org/ao/fulltext.cfm?uri=ao-55-25-6836&id=348983). Expressing a pixel on an undistorted image plane in the HSI frame is done through
```
u = np.random.randint(0, width) + 0.5 # Pixel numbers left to right (value from 1 to width)

# Pinhole part
x_norm_pinhole = (u - cx) / f

# Distortion part
x_norm_nonlin = -(k1 * (u - cx) ** 5 + \
                  k2 * (u - cx) ** 3 + \
                  k3 * (u - cx) ** 2) / f

x_norm_hsi = x_norm_pinhole + x_norm_nonlin

X_norm_hsi = np.array([x_norm_hsi, 0, 1]) # The pixel ray in the hsi frame
```

Of course most of these parameters are hard to guess for a HSI and there are two simple ways of finding them. The first way is to ignore distortions and assume you know the angular field of view (AFOV). Then you can calculate:

$$f = \frac{width}{2tan(AFOV/2)}$$


Besides that, you set cx=width/2, and remaining k's to zero. 

The second approach to get the camera model is if you have an array describing the FOV (often from the manufacturer). 
In our example above that would amount to a 512-length array, e.g. in degrees 
```
from gref4hsi.specim_parsing_utils import Specim

# FOV array from manufacturer describing the view angle of each pixel
fov = np.array([-19.1, -19.0 .... 19.0, 19.1]) # Length as number of pixels

# Use static method to convert fov to parameters dictionary equivalent to xml file (with boresights and lever arms to zero)
param_dict = Specim.fov_2_param(fov = specim_object.view_angles)

# Write to an *.xml file
param_dict['rz'] = rx
param_dict['ry'] = ry
param_dict['rx'] = rz

# Vector from origin of HSI to body origin, expressed in body
# User set
t_hsi_body = config_specim.translation_body_to_hsi
param_dict['tz'] = tx
param_dict['ty'] = ty
param_dict['tz'] = tz

# Write dictionary to *.xml file
CalibHSI(file_name_cal_xml= confic['Absolute Paths']['hsi_calib_path'], 
                    mode = 'w', 
                    param_dict = param_dict)
```



### Terrain files
There are three allowable types ("model_export_type" in the configuration file), namely "ply_file" (a.k.a. mesh files), "dem_file" and "geoid". Example geoids are added under data/world/geoids/ including the global egm08 and a norwegian geoid. Your local geoid can probably be found from [geoids](https://www.agisoft.com/downloads/geoids/) or similar. Just ensure you add this path to the 'Absolute Paths' section in the configuration file. Similarly, if you choose "model_export_type = dem_file", you can use a terrain model from from your area as long as you add the path to the file in the 'Absolute Paths'. Remember that if the local terrain (dem file) is given wrt the geoid, you can plus them together in e.g. QGIS or simply add an approximate offset in python with rasterio. This is because the software expects DEMs giving the ellipsoid height of the terrain. Lastly, if the "ply_file", or 3D triangular mesh is the desired option, a path to this file must be added to the config file. It should be fairly easy to use any triangular mesh format even though the option suggests "\*.ply". 

### The h5 files
This format must comply with the h5 paths in the configuration file. All sections starting with "HDF.xxxx" are related to these paths. I realize that it is inconvenient to transform the format if your recorded data is in NETCDF, ENVI etc, but making this software I chose H5/HDF because of legacy and because of the flexibility of the mini-file system. The input structure (only mandatory entries) of the H5 files under mission_dir/Input/H5 is as follows (printed using h5tree in Linux):  
```
2022-08-31_0800_HSI_transectnr_0_chunknr_0.h5
├── processed
│   └── radiance
│       ├── calibration
│       │   ├── geometric # Optional
│       │   │   └── view_angles
│       │   ├── radiometric # Only if is_radiance = False
│       │   │   ├── darkFrame
│       │   │   └── radiometricFrame
│       │   └── spectral 
│       │       ├── band2Wavelength
│       │       └── fwhm # Optional
│       ├── dataCube 
│       ├── exposureTime
│       └── timestamp
└── raw
    └── nav
        ├── euler_angles
        ├── position_ecef
        └── timestamp
```
Note that all these paths are parameters in the config file and you will now we will explain how to simply fill in this file tree using the config file. An example of format conversion is given under gref4hsi/utils/specim_parsing_utils.py where the raw format of the specim data and a navigation system is converted and written to the h5 format. The following code is taken from that script and shows how you can write all the necessary data to the appropriate h5 format. Assume that "specim_object" represents an object whose attributes are the datasets, meaning that e.g. specim_object.radiance_cube is the 3D datacube, and the method 'specim_object_2_h5_file' will write the data cube to the h5-path specified in configuration section 'HDF.hyperspectral' by entry 'datacube'

```
def specim_object_2_h5_file(h5_filename, h5_tree_dict, specim_object):
    with h5py.File(h5_filename, 'w', libver='latest') as f:
        for attribute_name, h5_hierarchy_item_path in h5_tree_dict.items():
            #print(attribute_name)
            dset = f.create_dataset(name=h5_hierarchy_item_path, 
                                            data = getattr(specim_object, attribute_name))

h5_dict_write = {'eul_zyx' : config['HDF.raw_nav']['eul_zyx'],
            'position_ecef' : config['HDF.raw_nav']['position'],
            'nav_timestamp' : config['HDF.raw_nav']['timestamp'],
            'radiance_cube': config['HDF.hyperspectral']['datacube'],
            't_exp_ms': config['HDF.hyperspectral']['exposuretime'],
            'hsi_timestamps': config['HDF.hyperspectral']['timestamp'],
            'view_angles': config['HDF.calibration']['view_angles'], # Optional
            'wavelengths' : config['HDF.calibration']['band2wavelength'],
            'fwhm' : config['HDF.calibration']['fwhm'], # Optional
            'dark_frame' : config['HDF.calibration']['darkframe'], # Optional unless 'is_radiance' is False
            'radiometric_frame' : config['HDF.calibration']['radiometricframe']} # Optional unless 'is_radiance' is False

# The dictionary is equivalent to the tree view above (assuming configuration_template.ini):
h5_dict_write = {
  'eul_zyx':            'raw/nav/euler_angles',
  'position_ecef':     'raw/nav/position_ecef',
  'nav_timestamp':     'raw/nav/timestamp',
  'radiance_cube':    'processed/radiance/dataCube',
  't_exp_ms':          'processed/radiance/exposureTime',
  'hsi_timestamps':   'processed/radiance/timestamp',
  'view_angles':       'processed/radiance/calibration/geometric/view_angles',
  'wavelengths':       'processed/radiance/calibration/spectral/band2Wavelength',
  'fwhm':              'processed/radiance/calibration/spectral/fwhm',
  'dark_frame':        'processed/radiance/calibration/radiometric/darkFrame',
  'radiometric_frame': 'processed/radiance/calibration/radiometric/radiometricFrame'
}

# At last, write the data to a h5 file
# In this case 'specim_object' is an object whoose attributes correspond to the above keys
# e.g. specim_object.radiance_cube is the 3D datacube, and the method 'specim_object_2_h5_file' will write  the data cube to the h5-path specified in configuration section 'HDF.hyperspectral' by entry 'datacube'
specim_object_2_h5_file(h5_filename=h5_filename, h5_tree_dict=h5_dict_write, specim_object=specim_object)
```
My recommended approach would be that you create your own modified xxx_parsing_utils.py (feel free to send a pull request) which parses and writes your specific navigation data and hyperspectral data into the format.

The last thing worth mentioning is the navigation data. The aforementioned specim_parsing_utils.py parses navigation data from messages, e.g.

Positions from 
"$GPGGA,125301.00,6335.47830829,N,00932.34206055,E,2,07,1.7,18.434,M,41.216,M,4.0,0123\*79"

Orientations/attitude from messages like
"$PASHR,125648.280,322.905,T,0.621,-0.114,,0.034,0.034,0.450,2,3\*1B"

The above data log is read with the method (in the case you have a similar navigation format)

```
specim_object.read_nav_file(nav_file_path=nav_file_path, date = date_string)
```

Lastly, the data should be formatted as follows for writing:

```
import pymap3d as pm

# If your nav system gave you geodetic positions, convert them to earth centered earth fixed (ECEF). Make sure to use ellipsoid height (not height above mean sea level (MSL) aka geoid)
x, y, z = pm.geodetic2ecef(lat = lat, lon = lon, alt = ellipsoid_height, deg=True)

# Roll pitch yaw are ordered with in an unintuitive attribute name eul_zyx. The euler angles with rotation order ZYX are Yaw Pitch Roll
specim_object.eul_zyx = np.concatenate((roll, pitch, yaw), axis = 1)

# The ECEF positions
specim_object.position_ecef = np.concatenate((x,y,z), axis = 1)
```

Then the "specim_object_2_h5_file" puts the data into the right place.

## Running the processing
Once the above has been defined the processing can be run with a few lines of code (taken from gref4hsi/tests/test_main_specim.py):
```
from gref4hsi.scripts import georeference
from gref4hsi.scripts import orthorectification
from gref4hsi.utils import parsing_utils, specim_parsing_utils
from gref4hsi.scripts import visualize


# This function parses raw specim data including (spectral, radiometric, geometric) calibrations and nav data
# into an h5 file. The nav data is written to "raw/nav/" subfolders, whereas hyperspectral data and calibration data 
# written to "processed/hyperspectral/" and "processed/calibration/" subfolders. The camera model \*.xml file is also generated.
# The script must be adapted to other 
specim_parsing_utils.main(config=config,
                          config_specim=config_specim_preprocess)

# Interpolates and reformats the pose (of the vehicle body) to "processed/nav/" folder.
config = parsing_utils.export_pose(config_file_mission)

# Exports model
parsing_utils.export_model(config_file_mission)

# Georeference the line scans of the hyperspectral imager. Utilizes parsed data
georeference.main(config_file_mission)

# Orthorectify/resample datacube, RGB-composite and ancillary data
orthorectification.main(config_file_mission)

# Optional: coregistration
# Match RGB composite to reference, find features and following data, ground control point (gcp) list, for each feature pair:
# reference point 3D (from reference), position/orientation of vehicle (using resampled time) and pixel coordinate (using resampled pixel coordinate)
coregistration.main(config_file_mission, mode='compare')

# The gcp list allows reprojecting reference points and evaluate the reprojection error,
# which is used to optimize static geometric parameters (e.g. boresight...) or dynamic geometric parameters (time-varying nav errors).
coregistration.main(config_file_mission, mode='calibrate')
```

