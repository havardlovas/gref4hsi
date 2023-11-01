import numpy as np
from scipy.spatial.transform import Rotation as RotLib
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import open3d as o3d
import xmltodict
from pyproj import CRS, Transformer
# A file were we define geometry and geometric transforms.

# Perhaps a class with one "__init__.py
class CalibHSI:
    def __init__(self, file_name_cal_xml, config, mode = 'r', param = None):
        """
        :param file_name_cal_xml: str
        File name of calibration file for line camera model
        :param config: config
        global configuration object.
        :param mode: str
        open file for reading (for general use) or writing (post calibration)
        :param param: ndarray:
        1D array with data type float. Is the line camera parameter vector (currently numpy array) with order
        "rotation_x, rotation_y, rotation_z, translation_x, translation_y, translation_z, c_x, focal_length,
        distortion_coeff_1, distortion_coeff_2, distortion_coeff_3"
        """
        if mode == 'r':
            with open(file_name_cal_xml, 'r', encoding='utf-8') as file:
                my_xml = file.read()
            xml_dict = xmltodict.parse(my_xml)
            self.calibrationHSI = xml_dict['calibration']

            self.w = float(self.calibrationHSI['width'])
            self.f = float(self.calibrationHSI['f'])
            self.cx = float(self.calibrationHSI['cx'])

            # Rotations
            self.rx = float(self.calibrationHSI['rx'])
            self.ry = float(self.calibrationHSI['ry'])
            self.rz = float(self.calibrationHSI['rz'])

            # Translations
            self.tx = float(self.calibrationHSI['tx'])
            self.ty = float(self.calibrationHSI['ty'])
            self.tz = float(self.calibrationHSI['tz'])


            # If encoding this into
            if eval(config['General']['isFlippedRGB']):
                self.tx *= -1
                self.ty *= -1
            if eval(config['General']['isFlippedHSI']):
                self.rz += np.pi

            # Distortions
            self.k1 = float(self.calibrationHSI['k1'])
            self.k2 = float(self.calibrationHSI['k2'])
            self.k3 = float(self.calibrationHSI['k3'])
        elif mode == 'w':
            with open(file_name_cal_xml, 'r', encoding='utf-8') as file:
                my_xml = file.read()
            xml_dict = xmltodict.parse(my_xml)

            xml_dict['calibration']['rx'] = param[0]
            xml_dict['calibration']['ry'] = param[1]
            xml_dict['calibration']['rz'] = param[2]

            #xml_dict['calibration']['tx'] = param[3]
            #xml_dict['calibration']['ty'] = param[4]
            #xml_dict['calibration']['tz'] = param[5]

            xml_dict['calibration']['cx'] = param[3]
            xml_dict['calibration']['f'] = param[4]
            xml_dict['calibration']['k1'] = param[5]
            xml_dict['calibration']['k2'] = param[6]
            xml_dict['calibration']['k3'] = param[7]
            with open(file_name_cal_xml, 'w') as fd:
                fd.write(xmltodict.unparse(xml_dict))

class CameraGeometry():
    def __init__(self, pos0, pos, rot, time, is_interpolated = False, is_offset = False):
        self.pos0 = pos0
        if is_offset:
            self.Position = pos
        else:
            self.Position = pos - np.transpose(pos0) # Camera pos



        self.Rotation = rot

        self.time = time
        self.IsLocal = False
        self.decoupled = True

        if is_interpolated:
            self.PositionInterpolated = self.Position
            self.RotationInterpolated = self.Rotation



    def interpolate(self, time_hsi, minIndRGB, maxIndRGB, extrapolate):
        """"""
        # A simple interpolation of transforms where all images should be aligned.
        # Should also implement a more sensor fusion-like Kalman filter implementation
        self.time_hsi = time_hsi
        if self.decoupled == True:

            if extrapolate == False:
                time_interpolation = time_hsi[minIndRGB:maxIndRGB]
                linearPositionInterpolator = interp1d(self.time, np.transpose(self.Position))
                self.PositionInterpolated = np.transpose(linearPositionInterpolator(time_interpolation))
                linearSphericalInterpolator = Slerp(self.time, self.Rotation)
                self.RotationInterpolated = linearSphericalInterpolator(time_interpolation)
            else:
                # Extrapolation of position:
                time_interpolation = time_hsi
                linearPositionInterpolator = interp1d(self.time, np.transpose(self.Position), fill_value='extrapolate')
                self.PositionInterpolated = np.transpose(linearPositionInterpolator(time_interpolation))

                # Extrapolation of orientation/rotation:

                # Synthetizize additional frames
                delta_rot_b1_b2 = (self.Rotation[-1].inv()) * (self.Rotation[-2])
                delta_time_last = self.time[-1] - self.time[-2]
                time_last = self.time[-1] + delta_time_last
                rot_last = self.Rotation[-1] * (delta_rot_b1_b2.inv()) # Assuming a continuation of the rotation

                # Rotation from second to first attitude
                delta_rot_b1_b2 = (self.Rotation[0].inv()) * (self.Rotation[1])
                delta_time_last = self.time[0] - self.time[1]
                time_first = self.time[0] + delta_time_last # Subtraction
                # Add the rotation from second to first attitude to the first attitude "continue" rotation
                rot_first = self.Rotation[0] * (delta_rot_b1_b2.inv())  # Assuming a continuation of the rotation
                time_concatenated = np.concatenate((np.array(time_first).reshape((1,-1)),
                                                    self.time.reshape((1,-1)),
                                                    np.array(time_last).reshape((1,-1))), axis = 1)\
                    .reshape(-1).astype(np.float64)
                rotation_list = [self.Rotation]
                rotation_list.append(rot_last)
                rotation_list.insert(0, rot_first)


                rot_vec_first = rot_first.as_rotvec().reshape((1,-1))
                rot_vec_mid = self.Rotation.as_rotvec()
                rot_vec_last = rot_last.as_rotvec().reshape((1,-1))

                rotation_vec_tot = np.concatenate((rot_vec_first, rot_vec_mid, rot_vec_last), axis = 0)

                Rotation_tot = RotLib.from_rotvec(rotation_vec_tot)



                time_concatenated = np.array(time_concatenated).astype(np.float64)

                linearSphericalInterpolator = Slerp(time_concatenated, Rotation_tot)

                self.RotationInterpolated = linearSphericalInterpolator(time_interpolation)






        else:
            print('Proper interpolation of transformation with constant velocity and rotation has not yet been implemented')
            print('See https://www.geometrictools.com/Documentation/InterpolationRigidMotions.pdf')
            #self.RotationInterpolated, self.PositionInterpolated = self.interpolateTransforms()
    def intrinsicTransformHSI(self, translation_rgb_hsi, rot_hsi_rgb, euler = True):
        # An intrinsic transform is a transformation to another reference frame on the moving body, i.e. the IMU or UHI
        self.PositionHSI = self.PositionInterpolated + self.RotationInterpolated.apply(translation_rgb_hsi)
        if euler == True:
            self.Rotation_hsi_rgb = RotLib.from_euler('ZYX', rot_hsi_rgb, degrees=True)

        self.RotationHSI = self.RotationInterpolated * self.Rotation_hsi_rgb # Composing rotations. See
        # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.spatial.transform.Rotation.__mul__.html
    def localTransform(self, frame):
        self.IsLocal = True
        self.LocalTransformFrame = frame

        if frame == 'ENU':
            self.PositionInterpolated = self.Rotation_ecef_enu*self.PositionInterpolated
            self.RotationInterpolated = self.Rotation_ecef_enu*self.RotationInterpolated
            self.Position = self.Rotation_ecef_enu * self.Position
            self.Rotation = self.Rotation_ecef_enu * self.Rotation
            self.RotationHSI = self.Rotation_ecef_enu * self.RotationHSI
            self.Position = self.Rotation_ecef_enu * self.PositionHSI
        elif frame == 'NED':
            self.localPositionInterpolated = self.Rotation_ecef_ned*self.PositionInterpolated
            self.localRotationInterpolated = self.Rotation_ecef_ned*self.RotationInterpolated
            self.localPosition = self.Rotation_ecef_ned * self.Position
            self.localRotation = self.Rotation_ecef_ned * self.Rotation
            self.RotationHSI = self.Rotation_ecef_ned * self.RotationHSI
            self.Position = self.Rotation_ecef_ned * self.PositionHSI
        else:
            print('Frame must be ENU or NED')
    def localTransformInverse(self):

        if self.IsLocal:
            self.IsLocal = False
            if self.LocalTransformFrame == 'ENU':
                self.PositionInterpolated = self.Rotation_ecef_enu.inv()*self.PositionInterpolated
                self.RotationInterpolated = self.Rotation_ecef_enu.inv()*self.RotationInterpolated
                self.Position = self.Rotation_ecef_enu.inv() * self.Position
                self.Rotation = self.Rotation_ecef_enu.inv() * self.Rotation
                self.RotationHSI = self.Rotation_ecef_enu.inv() * self.RotationHSI
                self.Position = self.Rotation_ecef_enu.inv() * self.PositionHSI
            elif self.LocalTransformFrame == 'NED':
                self.localPositionInterpolated = self.Rotation_ecef_ned.inv()* self.PositionInterpolated
                self.localRotationInterpolated = self.Rotation_ecef_ned.inv()*self.RotationInterpolated
                self.localPosition = self.Rotation_ecef_ned.inv() * self.Position
                self.localRotation = self.Rotation_ecef_ned.inv() * self.Rotation
                self.RotationHSI = self.Rotation_ecef_ned.inv() * self.RotationHSI
                self.Position = self.Rotation_ecef_ned.inv() * self.PositionHSI
            else:
                print('Frame must be ENU or NED')
        else:
            print('Poses are defined globally already')
    def defineRayDirections(self, dir_local):
        self.rayDirectionsLocal = dir_local

        n = self.PositionHSI.shape[0]
        m = dir_local.shape[0]

        self.rayDirectionsGlobal = np.zeros((n, m, 3))
        #
        #self.rayDirectionsGlobal = self.RotationHSI.apply(np.transpose(dir_local))
        for i in range(n):
            self.rayDirectionsGlobal[i, :, :] = self.RotationHSI[i].apply(dir_local)
    def intersectWithMesh(self, mesh, max_ray_length):

        n = self.rayDirectionsGlobal.shape[0]
        m = self.rayDirectionsGlobal.shape[1]

        self.projection = np.zeros((n, m, 3), dtype=np.float64)
        self.normalsGlobal = np.zeros((n, m, 3), dtype=np.float64)

        # Duplicate multiple centra
        start = np.einsum('ijk, ik -> ijk', np.ones((n, m, 3), dtype=np.float64), self.PositionHSI).reshape((-1,3))

        dir = (self.rayDirectionsGlobal * max_ray_length).reshape((-1,3))

        print('Starting Ray Tracing')
        print(start.shape)
        import time
        start_time = time.time()
        points, rays, cells = mesh.multi_ray_trace(origins=start, directions=dir, first_point=True)
        stop_time = time.time()
        print(start_time - stop_time)
        print(cells.max())

        normals = mesh.cell_normals[cells,:]

        slit_image_number = np.floor(rays / m).astype(np.int32)
        pixel_number = rays % m

        # Assign normals
        self.projection[slit_image_number, pixel_number] = points
        self.normalsGlobal[slit_image_number, pixel_number] = normals

        self.camera_to_seabed_world = self.projection - start.reshape((n, m, 3))
        self.camera_to_seabed_local = np.zeros(self.camera_to_seabed_world.shape)

        self.normalsLocal = np.zeros(self.normalsGlobal.shape)

        self.depth_map = np.zeros((n, m))
        for i in range(n):
            self.camera_to_seabed_local[i, :, :] = (self.RotationHSI[i].inv()).apply(self.camera_to_seabed_world[i, :, :])
            self.normalsLocal[i,:,:] = (self.RotationHSI[i].inv()).apply(self.normalsGlobal[i, :, :])
            self.depth_map[i, :] = self.camera_to_seabed_local[i, :, 2]/self.rayDirectionsLocal[:, 2]

        #dir = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/BarentsSea06052021/'
        #np.save(dir + 'local', self.camera_to_seabed_local)
        #np.save(dir + 'global', self.camera_to_seabed_world)
        #np.save(dir + 'hsi_pos', self.PositionHSI)


        #import matplotlib.pyplot as plt
        #self.depth_map[self.depth_map > 1] = 0
        #plt.imshow(self.depth_map)
        #plt.show()



        print('Finished ray tracing')
    def writeRGBPointCloud(self, config, hyp, transect_string, extrapolate = True, minInd = None, maxInd = None):
        wl_red = float(config['General']['RedWavelength'])
        wl_green = float(config['General']['GreenWavelength'])
        wl_blue = float(config['General']['BlueWavelength'])
        dir_point_cloud = config['Georeferencing']['rgbPointCloudPath']

        wavelength_nm = np.array([wl_red, wl_green, wl_blue])

        # Localize the appropriate band indices used for analysis
        band_ind_R = np.argmin(np.abs(wavelength_nm[0] - hyp.band2Wavelength))
        band_ind_G = np.argmin(np.abs(wavelength_nm[1] - hyp.band2Wavelength))
        band_ind_B = np.argmin(np.abs(wavelength_nm[2] - hyp.band2Wavelength))

        if extrapolate == False:
            rgb = hyp.dataCube[minInd:maxInd, :, [band_ind_R, band_ind_G, band_ind_B]]
        else:
            rgb = hyp.dataCube[:, :, [band_ind_R, band_ind_G, band_ind_B]]


        points = self.projection[self.projection != 0].reshape((-1,3))
        rgb_points = (rgb[self.projection != 0] / rgb.max()).astype(np.float64).reshape((-1,3))
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb_points)
        o3d.io.write_point_cloud(dir_point_cloud + transect_string + '.ply', pcd)


    #def transformRays(self, rays):
        # Rays are a n x m x 3 set of rays where n is each time step and coincides with the rotations and translations:

class FeatureCalibrationObject():
    def __init__(self, type, config):
        self.config = config
        self.type = type
        self.HSIPositionFeature = [] # Function of feature line
        self.HSIRotationFeature = [] #
        self.isfirst = True

    def load_cam_calibration(self, filenameCal, config):
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


    def appendGeometry(self, hsiGis, cameraGeometry, binning):
        # Step 1: Define the global projected position of features
        point_feature_gt = hsiGis.features_points
        # Define the hsi pixel number
        pixel = self.bilinearInterpolation(x1_x=hsiGis.x1_x_hsi, y1_y=hsiGis.y1_y_hsi, f_Q=hsiGis.v_datacube_hsi)
        # Define the translation:
        # First define the time stamp for the four lines:
        line_nr = hsiGis.u_datacube_hsi.astype(np.int64)

        trans_Q_00 = cameraGeometry.PositionHSI[line_nr[:, 0]]
        trans_Q_01 = cameraGeometry.PositionHSI[line_nr[:, 1]]
        trans_Q_10 = cameraGeometry.PositionHSI[line_nr[:, 2]]
        trans_Q_11 = cameraGeometry.PositionHSI[line_nr[:, 3]]

        translationHSI = self.bilinearInterpolationPosition(x1_x=hsiGis.x1_x_hsi, y1_y=hsiGis.y1_y_hsi, trans_Q_00=trans_Q_00, trans_Q_01=trans_Q_01, trans_Q_10=trans_Q_10,
                                                              trans_Q_11=trans_Q_11)

        rot_Q_00 = cameraGeometry.RotationInterpolated[line_nr[:, 0]]
        rot_Q_01 = cameraGeometry.RotationInterpolated[line_nr[:, 1]]
        rot_Q_10 = cameraGeometry.RotationInterpolated[line_nr[:, 2]]
        rot_Q_11 = cameraGeometry.RotationInterpolated[line_nr[:, 3]]

        rotationRGB = self.bilinearInterpolationRotation(x1_x=hsiGis.x1_x_hsi, y1_y=hsiGis.y1_y_hsi,
                                                              rot_Q_00=rot_Q_00, rot_Q_10=rot_Q_10, rot_Q_01=rot_Q_01,
                                                              rot_Q_11=rot_Q_11)
        if self.isfirst:
            self.point_feature_gt = point_feature_gt
            self.pixel = pixel
            self.translationHSI = translationHSI
            self.rotationRGB = rotationRGB
            self.binning = np.array([binning]).reshape((1,1))
            self.diff = hsiGis.diff

            self.isfirst = False
        else:
            # Add observations
            self.diff = np.concatenate((self.diff, hsiGis.diff), axis = 0)
            self.point_feature_gt = np.concatenate((self.point_feature_gt, point_feature_gt), axis = 0)
            self.pixel = np.concatenate((self.pixel, pixel), axis = 0)
            self.translationHSI = np.concatenate((self.translationHSI, translationHSI), axis = 0)
            self.rotationRGB = np.concatenate((self.rotationRGB, rotationRGB), axis = 0)



    #def defineRayDirections
    def bilinearInterpolation(self, x1_x, y1_y, f_Q):


        f_x_y1 = (1 - x1_x)*f_Q[:, 0] + x1_x*f_Q[:, 1]
        f_x_y2 = (1 - x1_x) * f_Q[:, 2] + x1_x * f_Q[:, 3]

        f_x_y = (1 - y1_y)*f_x_y1 + y1_y*f_x_y2

        return f_x_y
    def bilinearInterpolationPosition(self, x1_x, y1_y, trans_Q_00, trans_Q_01, trans_Q_10,
                                                              trans_Q_11):

        trans_tot = np.zeros(trans_Q_00.shape)
        for i in range(3):

            f_x_y1 = (1 - x1_x)*trans_Q_00[:, i] + x1_x*trans_Q_01[:, i]
            f_x_y2 = (1 - x1_x) * trans_Q_10[:, i] + x1_x * trans_Q_11[:, i]

            f_x_y = (1 - y1_y)*f_x_y1 + y1_y*f_x_y2
            trans_tot[:, i] = f_x_y


        return trans_tot

    def bilinearInterpolationRotation(self, x1_x, y1_y, rot_Q_00, rot_Q_10, rot_Q_01, rot_Q_11):

        # Define all rotations from rot_Q_00
        delta_rot0_rot = np.zeros((4, x1_x.shape[0], 3))
        # Define all relative rotations
        delta_rot0_rot[1, :, :] = (rot_Q_01 * rot_Q_00.inv()).as_rotvec(degrees = False)
        delta_rot0_rot[2, :, :] = (rot_Q_10 * rot_Q_00.inv()).as_rotvec(degrees = False)
        delta_rot0_rot[3, :, :] = (rot_Q_11 * rot_Q_00.inv()).as_rotvec(degrees = False)

        Q_rots_permuted = np.transpose(delta_rot0_rot, axes = [2, 1, 0])

        rot_vec_final = np.zeros((3, x1_x.shape[0]))

        for i  in range(3):
            f_Q = Q_rots_permuted[i, :, :]
            f_x_y1 = (1 - x1_x) * f_Q[:, 0] + x1_x * f_Q[:, 1]
            f_x_y2 = (1 - x1_x) * f_Q[:, 2] + x1_x * f_Q[:, 3]

            f_x_y = (1 - y1_y) * f_x_y1 + y1_y * f_x_y2

            rot_vec_final[i, :] = f_x_y

        delta_rot_interp = RotLib.from_rotvec(np.transpose(rot_vec_final))

        # Compose the values to rot0
        rot_tot = delta_rot_interp * rot_Q_00


        return rot_tot

    def computeDirections(self, param):
        self.rot_x = param[0]  # Pitch relative to cam (Equivalent to cam defining NED and uhi BODY)
        self.rot_y = param[1]*0  # Roll relative to cam
        self.rot_z = param[2]  # Yaw relative to cam
        self.v_c = param[3]
        self.f = param[4]
        self.k1 = param[5]*0
        self.k2 = param[6]
        self.k3 = param[7]


        #self.v_c = 455.414296
        #self.f = 9.55160147e+02
        #self.k1 = 0
        #self.k2 = 3.22199561e+02
        #self.k3 = -7.36445822e+01
        #self.rot_x = 1.41822029e-03
        #self.rot_z = -3.77072811e-03
        # Define camera model array for 960 pixels and 480. 1 is subtracted when.
        # Pixel ranges from -0.5
        # How to compensate for binning?

        self.v = self.pixel*self.binning + 0.5*(self.binning + 1)

        # Express uhi ray directions in uhi frame using line-camera model
        x_norm_lin = (self.v - self.v_c) / self.f

        x_norm_nonlin = -(self.k1 * ((self.v - self.v_c) / 1000) ** 5 + \
                          self.k2 * ((self.v - self.v_c) / 1000) ** 3 + \
                          self.k3 * ((self.v - self.v_c) / 1000) ** 2) / self.f

        self.x_norm = x_norm_lin + x_norm_nonlin

    # At the time of updated parameters
    def reprojectFeaturesHSI(self):
        rot_hsi_rgb = np.array([self.rot_z, self.rot_y, self.rot_x]) * 180 / np.pi

        self.Rotation_hsi_rgb = RotLib.from_euler('ZYX', rot_hsi_rgb, degrees=True)

        self.RotationHSI = self.rotationRGB * self.Rotation_hsi_rgb # Composing rotations.

        self.HSIToFeaturesGlobal = self.point_feature_gt - self.translationHSI

        n = self.HSIToFeaturesGlobal.shape[0]

        self.HSIToFeaturesLocal = np.zeros(self.HSIToFeaturesGlobal.shape)
        for i in range(n):
            self.HSIToFeaturesLocal[i, :] = (self.RotationHSI[i].inv()).apply(
                self.HSIToFeaturesGlobal[i, :])

            self.HSIToFeaturesLocal[i, :] /= self.HSIToFeaturesLocal[i, 2]




def interpolate_poses(timestamp_from, pos_from, pos0, rot_from, timestamps_to, extrapolate = True):
    """

    :param timestamp_from:
    Original timestamps
    :param pos_from: numpy array (n,3)
    Original positions
    :param rot_from: Rotation-object (n rotations)
    Original orientations
    :param timestamps_to:
    Timestamps desired for position and orientations
    :return:
    """

    minRGB = np.min(timestamp_from)
    maxRGB = np.max(timestamp_from)

    minInd = np.argmin(np.abs(minRGB - timestamps_to))
    maxInd = np.argmin(np.abs(maxRGB - timestamps_to))

    if timestamps_to[minInd] < minRGB:
        minInd += 1
    if timestamps_to[maxInd] > maxRGB:
        maxInd -= 1

    # Interpolate poses
    referenceGeometry = CameraGeometry(pos0=pos0,
                               pos=pos_from,
                               rot=rot_from,
                               time=timestamp_from)

    # We borrow a method from the camera object
    referenceGeometry.interpolate(time_hsi=timestamps_to,
                          minIndRGB=minInd,
                          maxIndRGB=maxInd,
                          extrapolate=extrapolate)


    position_to = referenceGeometry.PositionInterpolated
    quaternion_to = referenceGeometry.RotationInterpolated.as_quat()

    return position_to, quaternion_to














class MeshGeometry():
    def __init__(self, config):
        mesh_path = config['General']['modelPath']
        texture_path = config['General']['texPath']
        offsetX = float(config['General']['offsetX'])
        offsetY = float(config['General']['offsetY'])
        offsetZ = float(config['General']['offsetZ'])
        self.pos0 = np.array([offsetX, offsetY, offsetZ]).reshape([-1, 1])



def rotation_matrix_ecef2ned(lon, lat):
    """
    Computes the rotation matrix R from earth-fixed-earth centered (ECEF) to north-east-down (NED).
    :param lat: float in range [-90, 90]
    The latitude in degrees
    :param lon: float in range [-180, 180]
    :return R_ned_ecef: numpy array of shape (3,3)
    rotation matrix ECEF to NED
    """
    R_ned_ecef = rot_mat_ned_2_ecef(lon=lon, lat=lat)
    return np.transpose(R_ned_ecef)

def rotation_matrix_ecef2enu(lon, lat):
    l = np.deg2rad(lon)
    mu = np.deg2rad(lat)

    R_enu_ecef = np.array([[-np.cos(l) * np.sin(mu), -np.sin(l), -np.cos(l) * np.cos(mu)],
                           [-np.sin(l) * np.sin(mu), np.cos(l), -np.sin(l) * np.cos(mu)],
                           [np.cos(mu), 0, -np.sin(mu)]])
    #
    R_enu_ecef[[0, 1]] = R_enu_ecef[[1, 0]] # Swap rows
    R_enu_ecef[2] = -R_enu_ecef[2] # Switch signs to compensate for up and down


    R_ecef_enu = np.transpose(R_enu_ecef)
    return R_ecef_enu


def convert_rotation_ned_2_ecef(rot_obj_ned, position, is_geodetic = True, epsg_pos = 4326):
    """

    :param rot_obj_:
    :param position:
    The position
    :param is_geodetic: bool
    Whether position is geodetic
    :return:
    """


class GeoPose:
    def __init__(self, timestamps, rot_obj, rot_ref, pos, pos_epsg, is_pos_geocentric):
        self.timestamps = timestamps
        if rot_ref == 'NED':
            self.rot_obj_ned = rot_obj
            self.rot_obj_ecef = None
        elif rot_ref == 'ECEF':
            self.rot_obj_ned = None
            self.rot_obj_ecef = rot_obj
        else:
            print('This rotation reference is not supported')
            TypeError

        # Define position
        self.position = pos
        self.epsg = pos_epsg


        if is_pos_geocentric:
            self.pos_geocsc = self.position
        else:
            self.pos_geocsc = None

        self.lat = None
        self.lon = None
        self.hei = None


    def compute_geodetic_position(self, epsg_geod):
        """
            Function for transforming positions to latitude longitude height
            :param epsg_geod: int
            EPSG code of the transformed geodetic coordinate system
        """
        # If geocentric position has not been defined.
        if (self.pos_geocsc == None):
            from_CRS = CRS.from_epsg(self.epsg)
            geod_CRS = CRS.from_epsg(epsg_geod)
            transformer = Transformer.from_crs(from_CRS, geod_CRS)

            x = self.position[:, 0].reshape((-1, 1))
            y = self.position[:, 1].reshape((-1, 1))
            z = self.position[:, 2].reshape((-1, 1))

            (lat, lon, hei) = transformer.transform(xx=x, yy=y, zz=z)

            self.epsg_geod = epsg_geod
            self.lat = lat.reshape((self.position.shape[0], 1))
            self.lon = lon.reshape((self.position.shape[0], 1))
            self.hei = hei.reshape((self.position.shape[0], 1))
        else:
            pass

    def compute_geocentric_position(self, epsg_geocsc):
        """
            Function for transforming positions to geocentric
            :param epsg_geod: int
            EPSG code of the transformed geodetic coordinate system
        """

        from_CRS = CRS.from_epsg(self.epsg)
        geod_CRS = CRS.from_epsg(epsg_geocsc)
        transformer = Transformer.from_crs(from_CRS, geod_CRS)

        x = self.position[:, 0].reshape((-1, 1))
        y = self.position[:, 1].reshape((-1, 1))
        z = self.position[:, 2].reshape((-1, 1))

        (x, y, z) = transformer.transform(xx=x, yy=y, zz=z)

        self.pos_geocsc = np.concatenate((x,y,z), axis = 1)


    def compute_geocentric_orientation(self):
        if self.rot_obj_ecef == None:
            # Define rotations from ned to ecef
            if self.lat == None:
                # Needed to encode rotation between NED and ECEF
                self.compute_geodetic_position()

            R_body_2_ned = self.rot_obj_ned
            R_ned_2_ecef = self.compute_rot_obj_ned_2_ecef()
            R_body_2_ecef = R_ned_2_ecef*R_body_2_ned

            self.rot_obj_ecef = R_body_2_ecef

        else:
            pass

    def compute_ned_orientation(self):
        if self.rot_obj_ned == None:
            # Define rotations from body to ecef
            R_body_2_ecef = self.rot_obj_ecef

            # Define rotation from ecef 2 ned
            R_ecef_2_ned = self.compute_rot_obj_ned_2_ecef().inv()

            # Compose
            R_body_2_ned = R_ecef_2_ned*R_body_2_ecef

            self.rot_obj_ned = R_body_2_ned


        else:
            pass



    def compute_rot_obj_ned_2_ecef(self):

        N = self.lat.shape[0]
        rot_mats_ned_to_ecef = np.zeros((N, 3, 3), dtype=np.float64)



        for i in range(N):
            rot_mats_ned_to_ecef[i,:,:] = rot_mat_ned_2_ecef(lat=self.lat[i], lon = self.lon[i])


        self.rots_obj_ned_2_ecef = RotLib.from_matrix(rot_mats_ned_to_ecef)

        return self.rots_obj_ned_2_ecef




def rot_mat_ned_2_ecef(lat, lon):
    """
    Computes the rotation matrix R from north-east-down (NED) to earth-fixed-earth centered (ECEF)
    :param lat: float in range [-90, 90]
    The latitude in degrees
    :param lon: float in range [-180, 180]
    :return R_ned_ecef: numpy array of shape (3,3)
    rotation matrix from NED to ECEF
    """

    # Convert to radians
    l = np.deg2rad(lon)
    mu = np.deg2rad(lat)

    # Compute rotation matrix
    # TODO: add source
    R_ned_ecef = np.array([[-np.cos(l) * np.sin(mu), -np.sin(l), -np.cos(l) * np.cos(mu)],
                           [-np.sin(l) * np.sin(mu), np.cos(l), -np.sin(l) * np.cos(mu)],
                           [np.cos(mu), 0, -np.sin(mu)]])
    return R_ned_ecef
