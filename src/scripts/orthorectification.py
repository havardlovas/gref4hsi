import configparser
import os
import pickle
import sys
from scipy.spatial.transform import Rotation as RotLib
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import interp1d

# Local resources:
from scripts.geometry import CameraGeometry, FeatureCalibrationObject, CalibHSI
from scripts.gis_tools import GeoSpatialAbstractionHSI
from lib.parsing_utils import Hyperspectral



def main(iniPath):
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Paths to 3D mesh ply file 
    path_mesh = config['Absolute Paths']['modelPath']

    # Directory of H5 files
    dir_r = config['Absolute Paths']['h5Dir']

    # The path to the XML file
    hsi_cal_xml = config['Absolute Paths']['HSICalibFile']


    print('Orthorectifying Images')

    for filename in sorted(os.listdir(dir_r)):
        if filename.endswith('h5') or filename.endswith('hdf'):
            # Path to hierarchical file
            path_hdf = dir_r + filename

            # Read h5 file
            hyp = Hyperspectral(path_hdf, config)

            point_cloud_ecef = hyp.points_global

            # Generates an object for dealing with GIS operations
            gisHSI = GeoSpatialAbstractionHSI(point_cloud=point_cloud_ecef, 
                                              transect_string=filename.split('.')[0],
                                              config=config)

            gisHSI.transform_geocentric_to_projected()

            gisHSI.footprint_to_shape_file()

            gisHSI.resample_datacube(hyp, rgb_composite_only=True)



if __name__ == '__main__':
    args = sys.argv[1:]
    iniPath = args[0]
    main(iniPath)