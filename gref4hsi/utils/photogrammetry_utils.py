import Metashape as MS
import ntpath
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
from PIL import Image, ImageOps, ExifTags
from datetime import datetime
import os

import os


def add_gps_metadata_to_array(image_array, output_path):
    # Convert the NumPy array to an image
    image = Image.fromarray(np.uint8(image_array))

    if os.path.exists(output_path):
        pass
    else:
    # Save the image with Exif metadata
        image.save(output_path)







class Photogrammetry:
    def __init__(self, project_folder, software_type = 'agisoft'):
        """Initialize an object for interfacing with with photogrammetry softwares. Agisoft Metashape Professional is considered 
        (by the author) to be easiest to use, although opendronemap and Colmap are available to anyone

        :param project_folder: _description_
        :type project_folder: _type_
        :param software_type: _description_, defaults to 'agisoft'
        :type software_type: str, optional
        """
        self.software_type = software_type
        # We initialize the thing using
        if software_type == 'agisoft':
            reference_name = 'agisoft_reference.csv'

            self.doc = MS.Document()
            self.project_folder = project_folder
            try:
                self.doc.save(project_folder + '/project.psx')
                self.chunk = self.doc.addChunk()
            except OSError:
                pass

            self.df_nav = pd.DataFrame(columns=['Label','Timestamp',
                                       'Lon','Lat','EllipsoidHeight', 'LonAcc', 'LatAcc', 'HAcc',
                                       'Roll','Pitch', 'Yaw', 'RollAcc','PitchAcc', 'YawAcc'])

        

        
        # Define a clean csv reference file
        self.df_nav.to_csv(path_or_buf= project_folder + '/' + reference_name, mode='w')
        self.reference_path = project_folder + '/' + reference_name
        with open(self.reference_path, 'a+') as f:
            f.flush()

    
    def export_rgb_from_h5(self, h5_folder, rgb_write_dir, rgb_image_cube, nav_rgb, pos_acc, rot_acc):
        """
        Write images into png, appends to a nav file, and tags. Is potentially called many times"""
        h5_filename = ntpath.basename(h5_folder).split('.')[0]
        
        

        n_rgb_images = rgb_image_cube.shape[0]
        

        if not os.path.exists(rgb_write_dir):
            os.mkdir(rgb_write_dir)
        
        self.image_folder = rgb_write_dir
        

        # Simple counter at the end of the images to sort them.
        self.df_nav['Label'] = [h5_filename + '_' + "{:05d}".format(i) + '.png' for i in range(n_rgb_images)]
        self.df_nav['Timestamp'] = nav_rgb.lon.time_interp

        self.df_nav['Lon'] = nav_rgb.lon.value_interp
        self.df_nav['Lat'] = nav_rgb.lat.value_interp
        self.df_nav['EllipsoidHeight'] = -nav_rgb.pos_z.value_interp

        self.df_nav['LonAcc'] = pos_acc[0]*np.ones(n_rgb_images)
        self.df_nav['LatAcc'] = pos_acc[1]*np.ones(n_rgb_images)
        self.df_nav['HAcc'] = pos_acc[2]*np.ones(n_rgb_images)

        self.df_nav['Roll'] = nav_rgb.roll.value_interp
        self.df_nav['Pitch'] =  nav_rgb.pitch.value_interp
        self.df_nav['Yaw'] =  nav_rgb.yaw.value_interp

        self.df_nav['RollAcc'] = rot_acc[0]*np.ones(n_rgb_images)
        self.df_nav['PitchAcc'] =  rot_acc[1]*np.ones(n_rgb_images)
        self.df_nav['YawAcc'] =  rot_acc[2]*np.ones(n_rgb_images)

        for i, image_array_BGR in enumerate(rgb_image_cube):
            image_array_RGB = np.flip(image_array_BGR, axis = 2)
            image_write_path = rgb_write_dir + self.df_nav['Label'][i]
            add_gps_metadata_to_array(image_array=image_array_RGB,
                                      output_path=image_write_path)


        self.df_nav.to_csv(path_or_buf=self.reference_path, mode='a+', header=False, index=False)

        with open(self.reference_path, 'a+') as f:
            f.flush()
        

        self.df_nav = pd.DataFrame(columns=['Label','Timestamp',
                                       'Lon','Lat','EllipsoidHeight', 'LonAcc', 'LatAcc', 'HAcc',
                                       'Roll','Pitch', 'Yaw', 'RollAcc','PitchAcc', 'YawAcc'])
        

        
    def load_photos_and_reference(self):
        def find_files(folder, types):
            return [entry.path for entry in os.scandir(folder) if (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

        image_folder = self.image_folder

        photos = find_files(image_folder, [".jpg", ".png", ".jpeg", ".tif", ".tiff"])

        self.chunk.addPhotos(photos)

        self.doc.save()

        # See Metashape Python reference to make sense of columns string. The whitespace indicates skip column
        self.chunk.importReference(self.reference_path, delimiter=",", columns ='n xyzXYZcbaCBA', crs = MS.CoordinateSystem('EPSG::4326'))

        self.doc.save()

        try:
            sensor = self.chunk.sensors[0] #first calibration group in the active chunk
            calib = MS.Calibration()
            calib.width = sensor.width
            calib.height = sensor.height
            calib.load(self.project_folder + 'RGB_CAM_CAL.xml', format = MS.CalibrationFormatXML)
            sensor.user_calib = calib
            sensor.fixed_params = ['f', 'k1']
        except:
            calib_filename = self.project_folder + 'RGB_CAM_CAL.xml'
            print(f'No Camera model named {calib_filename}')
            pass

        self.doc.save()


    def mask_images(self):
        
        try:
            self.chunk.generateMasks(self.project_folder + 'mask.png', masking_mode=MS.MaskingMode.MaskingModeFile, cameras = self.chunk.cameras)
            self.doc.save()
        except:
            mask_filename = self.project_folder + 'mask.png'
            print(f'No Camera model named {mask_filename}')


    def align_images(self):
        self.chunk.matchPhotos(keypoint_limit = 40000, 
                               tiepoint_limit = 10000, 
                               generic_preselection = True, 
                               reference_preselection = True,
                               reference_preselection_mode = MS.ReferencePreselectionSequential)
        self.doc.save()

        self.chunk.alignCameras(reset_alignment = True)
        self.doc.save()






"""import Metashape
import os, sys, time"""

"""# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

def find_files(folder, types):
    return [entry.path for entry in os.scandir(folder) if (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

if len(sys.argv) < 3:
    print("Usage: general_workflow.py <image_folder> <output_folder>")
    raise Exception("Invalid script arguments")

image_folder = sys.argv[1]
output_folder = sys.argv[2]

photos = find_files(image_folder, [".jpg", ".jpeg", ".tif", ".tiff"])

doc = Metashape.Document()
doc.save(output_folder + '/project.psx')

chunk = doc.addChunk()

chunk.addPhotos(photos)
doc.save()

print(str(len(chunk.cameras)) + " images loaded")

chunk.matchPhotos(keypoint_limit = 40000, tiepoint_limit = 10000, generic_preselection = True, reference_preselection = True)
doc.save()

chunk.alignCameras()
doc.save()

chunk.buildDepthMaps(downscale = 2, filter_mode = Metashape.MildFiltering)
doc.save()

chunk.buildModel(source_data = Metashape.DepthMapsData)
doc.save()

chunk.buildUV(page_count = 2, texture_size = 4096)
doc.save()

chunk.buildTexture(texture_size = 4096, ghosting_filter = True)
doc.save()

has_transform = chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation

if has_transform:
    chunk.buildPointCloud()
    doc.save()

    chunk.buildDem(source_data=Metashape.PointCloudData)
    doc.save()

    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
    doc.save()

# export results
chunk.exportReport(output_folder + '/report.pdf')

if chunk.model:
    chunk.exportModel(output_folder + '/model.obj')

if chunk.point_cloud:
    chunk.exportPointCloud(output_folder + '/point_cloud.las', source_data = Metashape.PointCloudData)

if chunk.elevation:
    chunk.exportRaster(output_folder + '/dem.tif', source_data = Metashape.ElevationData)

if chunk.orthomosaic:
    chunk.exportRaster(output_folder + '/orthomosaic.tif', source_data = Metashape.OrthomosaicData)

print('Processing finished, results saved to ' + output_folder + '.')"""