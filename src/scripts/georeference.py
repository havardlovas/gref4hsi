import configparser
import os
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import interp1d

from geometry import CameraGeometry, FeatureCalibrationObject, CalibHSI
from gis_tools import GeoSpatialAbstractionHSI


def plot_figures_paper(fignum, data):
    if fignum == 1:
        r = data[data < 10] * 0.005 * 1000

        bins = np.linspace(0, 50, 50)

        hist, bins = np.histogram(r, bins=bins, density=True)
        r_hist = 0.5 * (bins[:-1] + bins[1:])

        def rayleigh_pdf(r, scale):
            return (r / scale ** 2) * np.exp(-r ** 2 / (2 * scale ** 2))

        # Compute the bin centers and widths
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = bins[1:] - bins[:-1]

        # Compute the cumulative distribution function
        cdf = np.cumsum(hist * bin_widths)

        # Compute the 95th percentile
        percentile_95 = np.interp(0.95, cdf / cdf[-1], bin_centers)

        # Compute the mean and median
        mean = np.sum(bin_centers * hist * bin_widths) / np.sum(hist * bin_widths)
        median = np.interp(0.5, np.cumsum(hist * bin_widths) / np.sum(hist * bin_widths), bin_centers)

        # Compute the mode
        mode = bin_centers[np.argmax(hist)]

        fig, ax = plt.subplots()
        ax.hist(r, bins=bins, label='Error histogram', alpha=0.5)
        ax.axvline(percentile_95, color='r', label='95th percentile = ' + "{:.1f}".format(percentile_95) + ' mm')
        ax.axvline(mean, color='g', label='mean = ' + "{:.1f}".format(mean) + ' mm')
        ax.axvline(median, color='m', label='median = ' + "{:.1f}".format(median) + ' mm')
        ax.axvline(mode, color='k', label='mode = ' + "{:.1f}".format(mode) + ' mm')
        plt.xlabel('Error [mm]')
        ax.legend()
        plt.show()
    elif fignum == 2:
        r = data[data < 10] * 0.005 * 1000

        bins = np.linspace(0, 20, 50)

        hist, bins = np.histogram(r, bins=bins, density=True)
        r_hist = 0.5 * (bins[:-1] + bins[1:])

        # Compute the bin centers and widths
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = bins[1:] - bins[:-1]

        # Compute the cumulative distribution function
        cdf = np.cumsum(hist * bin_widths)

        # Compute the 95th percentile
        percentile_95 = np.interp(0.95, cdf / cdf[-1], bin_centers)

        # Compute the mean and median
        mean = np.sum(bin_centers * hist * bin_widths) / np.sum(hist * bin_widths)
        median = np.interp(0.5, np.cumsum(hist * bin_widths) / np.sum(hist * bin_widths), bin_centers)

        # Compute the mode
        mode = bin_centers[np.argmax(hist)]

        fig, ax = plt.subplots()
        ax.hist(r, bins=bins, label='Error histogram', alpha=0.5)
        ax.axvline(percentile_95, color='r', label='95th percentile = ' + "{:.1f}".format(percentile_95) + ' mm')
        ax.axvline(mean, color='g', label='mean = ' + "{:.1f}".format(mean) + ' mm')
        ax.axvline(median, color='m', label='median = ' + "{:.1f}".format(median) + ' mm')
        ax.axvline(mode, color='k', label='mode = ' + "{:.1f}".format(mode) + ' mm')
        plt.xlabel('Error [mm]')
        ax.legend()
        plt.show()


class Hyperspectral:
    def __init__(self, filename, config):
        self.name = filename
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(filename, 'r', libver='latest') as self.f:
            # Special case
            if eval(config['HDF']['isMarmineUHISpring2017']):
                self.dataCube = self.f['uhi/pixels'][()]

                # Perform division by some value

                self.t_exp = self.f['uhi/parameters'][()][0, 0] / 1000
                self.dataCubeTimeStamps = self.f['uhi/parameters'][()][:, 2]
                self.band2Wavelength = self.f['uhi/calib'][()]

                self.RGBTimeStamps = self.f['rgb/parameters'][()][:, 2]
                self.RGBImgs = self.f['rgb/pixels'][()]

                # Check if the dataset exists
                if 'georeference' in self.f:
                    dir = 'georeference/'
                    self.normals_local = self.f[dir + 'normals_local'][()]
                    self.points_global = self.f[dir + 'points_global'][()]
                    self.points_local = self.f[dir + 'points_local'][()]
                    self.position_hsi = self.f[dir + 'position_hsi'][()]
                    self.quaternion_hsi = self.f[dir + 'quaternion_hsi'][()]

            else:
                dataCubePath = config['HDF.hyperspectral']['dataCube']
                exposureTimePath = config['HDF.hyperspectral']['exposureTime']
                timestampHyperspectralPath = config['HDF.hyperspectral']['timestamp']
                band2WavelengthPath = config['HDF.calibration']['band2Wavelength']
                radiometricFramePath = config['HDF.calibration']['radiometricFrame']
                darkFramePath = config['HDF.calibration']['darkFrame']
                RGBFramesPath = config['HDF.rgb']['rgbFrames']
                timestampRGBPath = config['HDF.rgb']['timestamp']

                self.dataCube = self.f[dataCubePath][()]

                self.t_exp = self.f[exposureTimePath][()][0] / 1000  # Recorded in milliseconds
                self.dataCubeTimeStamps = self.f[timestampHyperspectralPath][()]
                self.band2Wavelength = self.f[band2WavelengthPath][()]
                self.darkFrame = self.f[darkFramePath][()]
                self.radiometricFrame = self.f[radiometricFramePath][()]
                self.RGBTimeStamps = self.f[timestampRGBPath][()]
                self.RGBImgs = self.f[RGBFramesPath][()]

                # Check if the dataset exists
                if 'georeference' in self.f:
                    dir = 'georeference/'
                    self.normals_local = self.f[dir + 'normals_local'][()]
                    self.points_global = self.f[dir + 'points_global'][()]
                    self.points_local = self.f[dir + 'points_local'][()]
                    self.position_hsi = self.f[dir + 'position_hsi'][()]
                    self.quaternion_hsi = self.f[dir + 'quaternion_hsi'][()]





        self.n_scanlines = self.dataCube.shape[0]
        self.n_pix = self.dataCube.shape[1]
        self.n_bands = self.dataCube.shape[1]
        self.n_imgs = self.RGBTimeStamps.shape[0]

        # The maximal image size is typically 1920 by 1200. It is however cropped during recording
        # And there can be similar issues. The binning below is somewhat heuristic but general
        if self.n_pix > 1000:
            self.spatial_binning = 0
        elif self.n_pix > 500:
            self.spatial_binning = 1
        elif self.n_pix > 250:
            self.spatial_binning = 2

        if self.n_bands  > 600:
            self.spectral_binning = 0
        elif self.n_bands  > 250:
            self.spectral_binning = 1
        elif self.n_bands  > 125:
            self.spectral_binning = 2



    def DN2Radiance(self, config):

        # Special case
        if eval(config['HDF']['isMarmineUHISpring2017']):
            calibFolder = config['HDF']['calibFolder']
            if self.dataCube.shape[1] == 480:
                self.radiometricFrame = np.load(calibFolder + '/radiometricFrame480.npy')
                self.darkFrame = np.load(calibFolder + '/darkFrame480.npy')
                self.spatial_binning = 2
            elif self.dataCube.shape[1] == 960:
                self.radiometricFrame = np.load(calibFolder + '/radiometricFrame.npy')
                self.darkFrame = np.load(calibFolder + '/darkFrame.npy')
                self.spatial_binning = 1




        self.dataCubeRadiance = np.zeros(self.dataCube.shape, dtype = np.float64)
        for i in range(self.dataCube.shape[0]):
            self.dataCubeRadiance[i, :, :] = (self.dataCube[i, :, :] - self.darkFrame) / (
                    self.radiometricFrame * self.t_exp)

            #self.dataCubeRadiance[i, :, :] /= np.median(self.dataCubeRadiance[i, :, :], axis = 0)

        #for j in range(self.dataCube.shape[1]):
        #    self.dataCubeRadiance[:, j, :] /= np.median(self.dataCubeRadiance[:, j, :], axis=0)



        #import spectral as sp
        #sp.imshow(self.dataCubeRadiance, bands=(150, 97, 50))
        #import matplotlib.pyplot as plt
        #plt.pause(100)

    def addDataset(self, data, name):
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(self.name, 'a', libver='latest') as self.f:
            # Check if the dataset exists
            if name in self.f:
                self.f[name][:] = data
            else:
                dset = self.f.create_dataset(name=name, data = data)


class HyperspectralHI:
    def __init__(self, filename, config):
        self.name = filename
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(filename, 'r', libver='latest') as self.f:

                dataCubePath = config['HDF.hyperspectral']['dataCube']
                timestampHyperspectralPath = config['HDF.hyperspectral']['timestamp']

                self.dataCubeTimeStamps = self.f[timestampHyperspectralPath][()]


                self.dataCubeSet = self.f[dataCubePath]

                self.dataCube = self.dataCubeSet[0:4000,:,:]


                # Check if the dataset exists
                if 'nav' in self.f:
                    dir = 'georeference/'
                    self.points_local = self.f[dir + 'time'][()]
                    self.position_hsi = self.f[dir + 'position_hsi'][()]
                    self.quaternion_hsi = self.f[dir + 'quaternion_hsi'][()]





        self.n_scanlines = self.dataCube.shape[0]
        self.n_pix = self.dataCube.shape[1]
        self.n_bands = self.dataCube.shape[1]
        self.n_imgs = self.RGBTimeStamps.shape[0]

        # The maximal image size is typically 1920 by 1200. It is however cropped during recording
        # And there can be similar issues. The binning below is somewhat heuristic but general
        self.spatial_binning = 1



    def DN2Radiance(self, config):

        # Special case
        if eval(config['HDF']['isMarmineUHISpring2017']):
            calibFolder = config['HDF']['calibFolder']
            if self.dataCube.shape[1] == 480:
                self.radiometricFrame = np.load(calibFolder + '/radiometricFrame480.npy')
                self.darkFrame = np.load(calibFolder + '/darkFrame480.npy')
                self.spatial_binning = 2
            elif self.dataCube.shape[1] == 960:
                self.radiometricFrame = np.load(calibFolder + '/radiometricFrame.npy')
                self.darkFrame = np.load(calibFolder + '/darkFrame.npy')
                self.spatial_binning = 1




        self.dataCubeRadiance = np.zeros(self.dataCube.shape, dtype = np.float64)
        for i in range(self.dataCube.shape[0]):
            self.dataCubeRadiance[i, :, :] = (self.dataCube[i, :, :] - self.darkFrame) / (
                    self.radiometricFrame * self.t_exp)

            #self.dataCubeRadiance[i, :, :] /= np.median(self.dataCubeRadiance[i, :, :], axis = 0)

        #for j in range(self.dataCube.shape[1]):
        #    self.dataCubeRadiance[:, j, :] /= np.median(self.dataCubeRadiance[:, j, :], axis=0)



        #import spectral as sp
        #sp.imshow(self.dataCubeRadiance, bands=(150, 97, 50))
        #import matplotlib.pyplot as plt
        #plt.pause(100)

    def addDataset(self, data, name):
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(self.name, 'a', libver='latest') as self.f:
            # Check if the dataset exists
            if name in self.f:
                self.f[name][:] = data
            else:
                dset = self.f.create_dataset(name=name, data = data)




class RayCasting:
    def __init__(self, config):
        self.config = config
        # Offsets should correspond with polygon model offset
        self.offX = float(config['General']['offsetX'])
        self.offY = float(config['General']['offsetY'])
        self.offZ = float(config['General']['offsetZ'])
        self.pos0 = np.array([self.offX, self.offY, self.offZ]).reshape([-1, 1])


        self.hyp = []
        self.upsample_factor = int(config['Georeferencing']['upsamplingFactor'])
        self.step_imgs = int(config['Calibration']['stepImages'])

        self.max_ray_length = float(config['General']['maxRayLength'])

    def loadSeaBed(self, filenameSeaBed, filenameTexture = None):  # Loading Mesh in *.ply format
        
        if filenameTexture == None:
            self.mesh = pv.read(filenameSeaBed)
        else:
            print('Consult georeference_drone_imagery_multi_ray.py to see more about including a texture image')

    def loadPose(self, filenamePose):
        self.pose = pd.read_csv(filenamePose, sep=',', header=0)

    def loadCamCalibration(self, filenameCal, config):
        calHSI = CalibHSI(file_name_cal=filenameCal, config = config)  # Generates a calibration object
        self.f = calHSI.f
        self.v_c = calHSI.cx
        self.k1 = calHSI.k1
        self.k2 = calHSI.k2
        self.k3 = calHSI.k3

        self.trans_x = calHSI.tx
        self.trans_y = calHSI.ty
        self.trans_z = calHSI.tz

        self.rot_x = calHSI.rx
        self.rot_y = calHSI.ry
        self.rot_z = calHSI.rz

        #if eval(config['General']['isFlippedRGB']):
        #    self.trans_x *= -1
        #    self.trans_y *= -1
        #if eval(config['General']['isFlippedHSI']):
        #    self.rot_z += np.pi

    def init_hsi_rays(self):

        # Define camera model array for 960 pixels and 480.
        self.v_not_binned = np.arange(1, self.hyp.n_pix*self.hyp.spatial_binning + 1)

        if self.hyp.spatial_binning != 1:
            self.v = np.mean(self.v_not_binned.reshape((-1, self.hyp.spatial_binning)), axis = 1)
        else:
            self.v = self.v_not_binned

        # Express uhi ray directions in uhi frame using line-camera model
        x_norm_lin = (self.v - self.v_c) / self.f

        x_norm_nonlin = -(self.k1 * ((self.v - self.v_c) / 1000) ** 5 + \
                          self.k2 * ((self.v - self.v_c) / 1000) ** 3 + \
                          self.k3 * ((self.v - self.v_c) / 1000) ** 2) / self.f

        self.x_norm = x_norm_lin + x_norm_nonlin

        self.p_dir = np.zeros((len(self.x_norm), 3))

        # Rays are defined in the UHI frame with positive z down
        self.p_dir[:, 0] = self.x_norm
        self.p_dir[:, 2] = 1

        self.HSICamera = self.RGBCamera

        rot_hsi_rgb = np.array([self.rot_z, self.rot_y, self.rot_x]) * 180/np.pi
        translation_hsi_rgb = np.array([self.trans_x, self.trans_y, self.trans_z]) / 1000 # These are millimetres

        self.HSICamera.intrinsicTransformHSI(translation_rgb_hsi=-translation_hsi_rgb, rot_hsi_rgb= rot_hsi_rgb)

        self.HSICamera.defineRayDirections(dir_local=self.p_dir)

        #import visualize
        #visualize.show_camera_geometry(self.HSICamera, self.config)


    def update_camera_model_param(self, param):
        # For the different binning, a slightly different v_c and focal length was applied
        self.rot_x = param[0]  # Pitch relative to cam (Equivalent to cam defining NED and uhi BODY)
        self.rot_y = param[1]  # Roll relative to cam
        self.rot_z = param[2]  # Yaw relative to cam
        self.v_c = param[3]
        self.f = param[4]
        self.k1 = param[5]
        self.k2 = param[6]
        self.k3 = param[7]

        self.init_camera_rays()

    def interpolate_poses(self, transect_string, mode):
        self.transect_string = transect_string

        # Select the images wiht a label containing the string
        relevant_images = self.pose.iloc[:, 0].str.contains(transect_string)
        ind_vec = np.arange(len(relevant_images))

        # The last part of the camera label holds the index
        self.rgb_ind = [int(self.pose["CameraLabel"].str.split("_")[i][-1]) for i in ind_vec[relevant_images == True]]
        self.agisoft_ind = ind_vec[relevant_images == True] # The images containing the string
        # The relevant image indices (based on which images were matched during alignment)
        # 0-CameraLabel, 1-X, 2-Y, 3-Z, 4-Roll, 5-Pitch, 6-Yaw, 7-RotX, 8-RotY, 9-RotZ
        self.agisoft_poses = self.pose.values[self.agisoft_ind, 1:10]

        self.time_rgb = self.hyp.RGBTimeStamps[self.rgb_ind]

        self.select_timestamps(mode) # Selects timestamps for HSI

        # Interpolate for the selected time stamps. First find the time stamps within the RGB time stamps to avoid extrapolation

        minRGB = np.min(self.time_rgb)
        maxRGB = np.max(self.time_rgb)

        self.minInd = np.argmin(np.abs(minRGB - self.time_hsi))
        self.maxInd = np.argmin(np.abs(maxRGB - self.time_hsi))
        if self.time_hsi[self.minInd] < minRGB:
            self.minInd += 1
        if self.time_hsi[self.maxInd] > maxRGB:
            self.maxInd -= 1

        rotZXY = np.concatenate((self.agisoft_poses[:, 8].reshape((-1, 1)),
                                 self.agisoft_poses[:, 7].reshape((-1, 1)),
                                 self.agisoft_poses[:, 6].reshape((-1, 1))), axis=1)

        self.RGBCamera = CameraGeometry(pos0=self.pos0, pos=self.agisoft_poses[:, 0:3], rot=rotZXY, time=self.time_rgb)

        self.RGBCamera.interpolate(time_hsi=self.time_hsi, minIndRGB=self.minInd, maxIndRGB=self.maxInd, extrapolate=True)

        #print(self.transect_string)

    def select_timestamps(self, mode):
        if mode == 'georeference': ## Geometric upsampling of footprints. Only for georeferencing.
            if self.upsample_factor != 1:
                a1 = np.arange(len(self.hyp.dataCubeTimeStamps))  # Original timestep number
                f_time = interp1d(a1, self.hyp.dataCubeTimeStamps, fill_value='extrapolate')  # Interpolation function
                a2 = np.arange(len(self.hyp.dataCubeTimeStamps) * self.upsample_factor) / self.upsample_factor - (
                        self.upsample_factor - 1) / (
                             2 * self.upsample_factor)
                self.time_hsi = f_time(a2)
            else:
                self.time_hsi = self.hyp.dataCubeTimeStamps
        ##
        elif mode == 'calibrate': ## Selection of one scan per RGB or even less depending on a variety of factors.
            self.ind_hsi = np.zeros(len(range(0, self.hyp.n_imgs, self.step_imgs)), dtype='int32')
            self.n_valid_rgb_imgs = len(self.img_ind.reshape(-1))
            for i in range(0, self.n_valid_rgb_imgs, self.step_imgs):
                ind_rgb = self.rgb_ind[i]
                # We pair hsi indices with rgb indices
                self.ind_hsi[int(i / self.step_imgs)] = np.argmin(
                    np.abs(self.hyp.dataCubeTimeStamps - self.time_rgb[ind_rgb]))

            self.time_hsi = self.hyp.dataCubeTimeStamps[self.ind_hsi] # Only closest time stamps should be utilized.
        elif mode == 'geotag':
            self.time_hsi = self.hyp.dataCubeTimeStamps
        else:
            print('There is no such mode')

    def ray_trace(self):
        # This function is run per H5 file
        #print('Defining rays')
        self.init_hsi_rays()

        self.HSICamera.intersectWithMesh(mesh = self.mesh, max_ray_length=self.max_ray_length)

        print('Writing Point Cloud')
        self.HSICamera.writeRGBPointCloud(config = self.config, hyp = self.hyp, minInd=self.minInd, maxInd=self.maxInd, transect_string=self.transect_string, extrapolate =True)

        print('Point cloud written')
        import visualize
        visualize.show_projected_hsi_points(HSICameraGeometry=self.HSICamera, config=self.config, transect_string=self.transect_string)



        gisHSI = GeoSpatialAbstractionHSI(point_cloud=self.HSICamera.projection, transect_string=self.transect_string, datacube_indices=np.arange(self.minInd, self.maxInd), config=self.config)

        gisHSI.transform_geocentric_to_projected()

        gisHSI.footprint_to_shape_file()

        gisHSI.resample_datacube(self.hyp, rgb_composite=True, minInd=self.minInd, maxInd=self.maxInd, extrapolate = True)

        gisHSI.compare_hsi_composite_with_rgb_mosaic()

        self.gisHSI = gisHSI


    def map_back(self):
        w_datacube = self.hyp.n_pix
        self.gisHSI.map_pixels_back_to_datacube(w_datacube=w_datacube)
        # Write some sort of




def optimize_function(param, calObj):
    calObj.computeDirections(param) # Computes the directions in a local frame

    calObj.reprojectFeaturesHSI() # reprojects to the same frame

    errorx = calObj.x_norm - calObj.HSIToFeaturesLocal[:, 0]
    errory = -calObj.HSIToFeaturesLocal[:, 1]

    print(np.median(np.abs(errorx)))
    print(np.median(np.abs(errory)))
    #if np.median(np.abs(errorx)) < 0.01:
    #import matplotlib.pyplot as plt
##
    #plt.scatter(calObj.pixel, errorx)
    #plt.scatter(errorx, errory)




    #print(np.median(np.abs(errorx)))
    #print(np.median(np.abs(errory)))



    return np.concatenate((errorx.reshape(-1), errory.reshape(-1)))

def main(iniPath, mode, is_calibrated):
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Paths to pose csv, 3D model ply file and the directory of H5 files
    path_pose = config['General']['posePath']
    path_mesh = config['General']['modelPath']
    dir_r = config['HDF']['h5Dir']
    # The path to the XML file
    hsi_cal_xml_b1 = config['Georeferencing']['HSICalibFileB1']
    hsi_cal_xml_b2 = config['Georeferencing']['HSICalibFileB2']
    if is_calibrated != True:
        hsi_cal_xml = config['Calibration']['hsiCalibFileInit']
    rc = RayCasting(config)
    rc.loadPose(filenamePose=path_pose)
    rc.loadSeaBed(filenameSeaBed=path_mesh)


    # Only relevant for calibration part of things
    if is_calibrated != True:
        #calObj = FeatureCalibrationObject(type='camera calibration', config=config)
        #calObj.loadCamCalibration(filenameCal=hsi_cal_xml, config= config)

        calObj1 = FeatureCalibrationObject(type='camera calibration', config=config)
        calObj1.loadCamCalibration(filenameCal=hsi_cal_xml, config=config)

        calObj2 = FeatureCalibrationObject(type='camera calibration', config=config)
        calObj2.loadCamCalibration(filenameCal=hsi_cal_xml, config=config)
    else:
        calObj1 = FeatureCalibrationObject(type='camera calibration', config=config)
        calObj1.loadCamCalibration(filenameCal=hsi_cal_xml_b1, config=config)

        calObj2 = FeatureCalibrationObject(type='camera calibration', config=config)
        calObj2.loadCamCalibration(filenameCal=hsi_cal_xml_b2, config=config)



    count = 0
    if mode == 'georeference':
        print('Georeferencing Images')
        first = True
        for filename in sorted(os.listdir(dir_r)):
            if filename.endswith('h5') or filename.endswith('hdf'):
                if count > -1:
                    is_uhi = config['HDF']['is_uhi']

                    if is_uhi == 'True':
                        filename_splitted = filename.split('_')

                        transect_string = filename_splitted[2] + '_' + filename_splitted[3].split('.')[0]
                        print(transect_string)
                        path_hdf = dir_r + filename
                        # Read h5 file and asign to raycaster
                        rc.hyp = Hyperspectral(path_hdf, config)
                        rc.hyp.DN2Radiance(config)
                        print(transect_string + ' With binning ' + str(rc.hyp.spatial_binning))
                    else:
                        path_hdf = dir_r + filename
                        rc.hyp = HyperspectralHI(path_hdf, config)


                    # Load the appropriate calibration file
                    if is_calibrated == True:
                        if rc.hyp.spatial_binning == 1:
                            rc.loadCamCalibration(filenameCal=hsi_cal_xml_b1, config=config)
                        if rc.hyp.spatial_binning == 2:
                            rc.loadCamCalibration(filenameCal=hsi_cal_xml_b2, config=config)
                    else:
                        rc.loadCamCalibration(filenameCal=hsi_cal_xml, config= config)


                    # The interpolation of poses can be done prior to calibration. We want a position and orientation
                    rc.interpolate_poses(transect_string, mode)
                    rc.ray_trace()
    #
    #
    #
                    rc.map_back()

                    if is_calibrated == True:
                        # Append datasets to H5 file
                        dir = 'georeference/'

                        # Add global points
                        points_global = rc.gisHSI.points_proj # Use projected system for global description
                        points_global_name = dir + 'points_global'
                        rc.hyp.addDataset(data = points_global, name=points_global_name)

                        # Add local points
                        points_local = rc.HSICamera.camera_to_seabed_local  # Use projected system for global description
                        points_local_name = dir + 'points_local'
                        rc.hyp.addDataset(data=points_local, name=points_local_name)

                        # Add camera position
                        position_hsi = rc.HSICamera.PositionHSI  # Use projected system for global description
                        position_hsi_name = dir + 'position_hsi'
                        rc.hyp.addDataset(data=position_hsi, name=position_hsi_name)

                        # Add camera quaternion
                        quaternion_hsi = rc.HSICamera.RotationHSI.as_quat()  # Use projected system for global description
                        quaternion_hsi_name = dir + 'quaternion_hsi'
                        rc.hyp.addDataset(data=quaternion_hsi, name=quaternion_hsi_name)

                        # Add normals
                        normals_local = rc.HSICamera.normalsLocal # Use projected system for global description
                        normals_local_name = dir + 'normals_local'
                        rc.hyp.addDataset(data=normals_local, name=normals_local_name)

                    # Append datasets
    #
                    ##file_pi = open( 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/BarentsSea06052021/Pickle/RayCasting.pkl','wb')
                    ##pickle.dump(rc, file_pi)
    ##
                    ##file = open(
                    ##    'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/BarentsSea06052021/Pickle/RayCasting.pkl',
                    ##    'rb')
                    #### dump information to that file
                    ##rc = pickle.load(file)
    #

                    if rc.hyp.spatial_binning == 1:
                        calObj1.appendGeometry(hsiGis=rc.gisHSI, cameraGeometry=rc.HSICamera, binning = rc.hyp.spatial_binning)
                    if rc.hyp.spatial_binning == 2:
                        calObj2.appendGeometry(hsiGis=rc.gisHSI, cameraGeometry=rc.HSICamera, binning = rc.hyp.spatial_binning)

                    # After aquiring a bunch of calibration data
                    calObjPath1 = config['Calibration']['hsicalibObjPathB1']
                    file_cal1 = open(calObjPath1, 'wb')
                    pickle.dump(calObj1, file_cal1)

                    #calObjPath2 = config['Calibration']['hsicalibObjPathB2']
                    #file_cal2 = open(calObjPath2, 'wb')
                    #pickle.dump(calObj2, file_cal2)

                count += 1
                if count == 1:
                    break






    elif mode == 'calibrate':
        config = configparser.ConfigParser()
        config.read(iniPath)

        calObjPath1 = config['Calibration']['hsicalibObjPathB1']

        file1 = open(calObjPath1, 'rb')
        ## dump information to that file
        calObj1 = pickle.load(file1)

        #calObjPath2 = config['Calibration']['hsicalibObjPathB2']

        #file2 = open(calObjPath2, 'rb')
        #### dump information to that file
        #calObj2 = pickle.load(file2)


        if is_calibrated == False:
            print('Calibrating Boresight and Camera Model')




            # Step 1:
            param0 = np.array([calObj1.rot_x, calObj1.rot_y, calObj1.rot_z, calObj1.v_c, calObj1.f, calObj1.k1, calObj1.k2, calObj1.k3])

            from scipy.optimize import least_squares
            res = least_squares(fun = optimize_function, x0 = param0, args= (calObj1,) , x_scale='jac', method='lm')


            file_name_calib = config['Georeferencing']['hsicalibfileb1']


            param_calib = res.x

            if eval(config['General']['isFlippedHSI']):
                param_calib[2] -= np.pi

            param_calib[5] = 0
            param_calib[1] = 0
            print(param_calib)
            CalibHSI(file_name_cal=file_name_calib, config = config, mode = 'w', param = param_calib)

            # Step 1:
            param02 = np.array(
                [calObj2.rot_x, calObj2.rot_y, calObj2.rot_z, calObj2.v_c, calObj2.f, calObj2.k1, calObj2.k2, calObj2.k3])

            from scipy.optimize import least_squares
            res2 = least_squares(fun=optimize_function, x0=param02, args=(calObj2,), x_scale='jac', method='lm')

            file_name_calib2 = config['Georeferencing']['hsicalibfileb2']

            param_calib2 = res2.x

            if eval(config['General']['isFlippedHSI']):
                param_calib2[2] -= np.pi

            param_calib2[5] = 0
            param_calib2[1] = 0
            print(param_calib2)
            CalibHSI(file_name_cal=file_name_calib2, config=config, mode='w', param=param_calib2)
        else:
            import matplotlib.pyplot as plt
            resolution = float(config['Georeferencing']['resolutionhyperspectralmosaic'])
            print(np.median(calObj1.diff[calObj1.diff < 5])*resolution*1000)

            try:
                diff_tot = np.concatenate((calObj1.diff, calObj2.diff))
            except AttributeError:
                diff_tot = calObj1.diff



            print(np.median(diff_tot[diff_tot < 10]) * resolution * 1000)

            print(calObj1.diff)
            print(len(diff_tot))
            from scipy.optimize import curve_fit

            plot_figures_paper(fignum=2, data = diff_tot)
            #params, cov = curve_fit(f = rayleigh_pdf, xdata = r_hist, ydata=hist, p0=[1])

            #print(*params)
            #plt.hist(r, bins=bins, label='Error distribution')
            #plt.xlabel('Error [mm]')
            #plt.plot(r_hist, rayleigh_pdf(r_hist, 5), 'r-', label='Rayleigh fit')
            plt.legend()
            plt.show()

            resolution = float(config['Georeferencing']['resolutionhyperspectralmosaic'])
            print(np.median(calObj2.diff[calObj2.diff < 5]) * resolution * 1000)
            print(len(calObj2.diff[calObj2.diff < 5]))
            plt.hist(calObj2.diff[calObj2.diff < 10] * resolution * 1000, 25)
            plt.show()
















    elif mode == 'geotag':
        # We may geotag images all images within H5 file
        print('Geotagging RGB poses for hyperspectral and RGB time stamps')
        # We only need one ray caster object, but probably we can append hyperspectral
        files = os.listdir(dir_r)
        for filename in sorted(files):
            if filename.endswith('h5') or filename.endswith('hdf'):

                path_hdf = dir_r + filename
                # Read h5 file and asign to raycaster
                rc.hyp = Hyperspectral(path_hdf, config)

                #rc.hyp.DN2Radiance(config)
                filename_splitted = filename.split('_')
                transect_string = filename_splitted[2] + '_' + filename_splitted[3].split('.')[0]
                print(transect_string)
                rc.interpolate_poses(transect_string, mode)
                print()

if __name__ == '__main__':
    args = sys.argv[1:]
    iniPath = args[0]
    main(iniPath)
