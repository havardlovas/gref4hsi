import Metashape as MS

from Metashape import Vector as vec
import pickle
import time
import sys
import numpy as np
from os import path
import configparser
from scipy.spatial.transform import Rotation
import json
from osgeo import gdal
import pyvista as pv
import osr
from pyproj import CRS, Transformer
import pandas as pd


class DataLogger:
    def __init__(self, filename, header):
        self.filename = filename
        # Generate file with header
        with open(filename, 'w') as fh:
            fh.write(header + '\n')

    def append_data(self, data):
        # Append data line to file
        with open(self.filename, 'a') as fh:
            # Generate data line to file
            if data[0] != None:
                line = ','.join([str(el) for el in data])

                # Append line to file
                fh.write(line + '\n')




def extract_model(config, iniPath):
    # Input and output file paths
    if  config['General']['modelType'] == 'DEM':
        input_dem = config['General']['modelpathDEM']
        output_xyz = config['General']['modelpathXYZ']
        model_no_suffix = config['General']['modelpath']
        # No-data value
        no_data_value = int(config['General']['nodataDEM'])  # Replace with your actual no-data value
        # Open the input raster dataset
        ds = gdal.Open(input_dem)
        if ds is None:
            print(f"Failed to open {input_dem}")
        else:
            # Read the first band (band index is 1)
            band = ds.GetRasterBand(1)
            if band is None:
                print(f"Failed to open band 1 of {input_dem}")
            else:
                # Get the geotransform information to calculate coordinates
                geotransform = ds.GetGeoTransform()
                x_origin = geotransform[0]
                y_origin = geotransform[3]
                x_resolution = geotransform[1]
                y_resolution = geotransform[5]
                # Get the CRS information
                spatial_reference = osr.SpatialReference(ds.GetProjection())

                # Get the EPSG code
                epsg_code = None
                if spatial_reference.IsProjected():
                    epsg_code = spatial_reference.GetAttrValue("AUTHORITY", 1)
                elif spatial_reference.IsGeographic():
                    epsg_code = spatial_reference.GetAttrValue("AUTHORITY", 0)

                print(f"EPSG Code: {epsg_code}")

                config.set('General', 'demepsg', str(epsg_code))
                epsg_proj = epsg_code
                # Get the band's data as a NumPy array
                band_data = band.ReadAsArray()
                # Create a mask to identify no-data values
                mask = band_data != no_data_value
                # Create and open the output XYZ file for writing
                with open(output_xyz, 'w') as xyz_file:
                    # Write data to the XYZ file using the mask and calculated coordinates
                    for y in range(ds.RasterYSize):
                        for x in range(ds.RasterXSize):
                            if mask[y, x]:
                                x_coord = x_origin + x * x_resolution
                                y_coord = y_origin + y * y_resolution
                                xyz_file.write(f"{x_coord} {y_coord} {band_data[y, x]}\n")
                # Clean up
                ds = None
                band = None
        print("Conversion completed.")
        points = np.loadtxt(output_xyz)
        # Create a pyvista point cloud object
        cloud = pv.PolyData(points)
        # Generate the mesh
        mesh = cloud.delaunay_2d()

        epsg_geocsc = config['General']['modelepsg']
        # Transform the mesh points to ECEF.
        geocsc = CRS.from_epsg(epsg_geocsc)
        proj = CRS.from_epsg(epsg_proj)
        transformer = Transformer.from_crs(proj, geocsc)

        points_proj = mesh.points

        eastUTM = points_proj[:, 0].reshape((-1, 1))
        northUTM = points_proj[:, 1].reshape((-1, 1))
        heiUTM = points_proj[:, 2].reshape((-1, 1))

        (xECEF, yECEF, zECEF) = transformer.transform(xx=eastUTM, yy=northUTM, zz=heiUTM)

        mesh.points[:, 0] = xECEF.reshape(-1)
        mesh.points[:, 1] = yECEF.reshape(-1)
        mesh.points[:, 2] = zECEF.reshape(-1)

        mean_vec = np.mean(mesh.points, axis = 0)

        mesh.points -= mean_vec

        config.set('General', 'offsetX', str(mean_vec[0]))
        config.set('General', 'offsetY', str(mean_vec[1]))
        config.set('General', 'offsetZ', str(mean_vec[2]))

        # Save mesh
        mesh.save(model_no_suffix + '.vtk')
        mesh.save(model_no_suffix + '.stl')
        mesh.save(model_no_suffix + '.ply')

        with open(iniPath, 'w') as configfile:
            config.write(configfile)
    else:
        print('This model type has not been defined')



def main():
    show_mesh_camera()




if __name__ == '__main__':
    main()
