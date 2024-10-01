# Third party
import subprocess
import numpy as np
import numpy.matlib
from osgeo import gdal, osr
import psutil
import pyproj
import rasterio
from scipy.spatial.transform import Rotation as RotLib
from scipy.spatial.transform import Slerp
import open3d as o3d
import xmltodict
from pyproj import CRS, Transformer
import pymap3d as pm
import ephem
import pandas as pd
from scipy.interpolate import interp1d
import pyvista as pv
from shapely.geometry import Polygon, mapping, MultiPoint
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from rasterio.windows import Window
import trimesh

# Python standard lib
import os
import time
from datetime import datetime
from dateutil import parser
import json

# Internals:
from gref4hsi.utils.gis_tools import GeoSpatialAbstractionHSI as geohsi

# A file were we define geometry and geometric transforms
class CalibHSI:
    def __init__(self, file_name_cal_xml, mode = 'r', param_dict = None):
        """
        :param file_name_cal_xml: str
        File name of calibration file for line camera model
        :param config: config
        global configuration object.
        :param mode: str
        open file for reading (for general use) or writing (post calibration)
        :param param_dict: dictionary:
        dictionary with keys
        "'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'cx', 'f', 'k1', 'k2', 'k3', 'width'"
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

            # Distortions
            self.k1 = float(self.calibrationHSI['k1'])
            self.k2 = float(self.calibrationHSI['k2'])
            self.k3 = float(self.calibrationHSI['k3'])
        elif mode == 'w':
            # Check if the file exists
            if os.path.exists(file_name_cal_xml):
                with open(file_name_cal_xml, 'r', encoding='utf-8') as file:
                    my_xml = file.read()
                xml_dict = xmltodict.parse(my_xml)
            else:
                # Create a new xml_dict with default structure
                xml_dict = {'calibration': {}}

            # Update xml_dict['calibration'] with values from param_dict
            for key, value in param_dict.items():
                xml_dict['calibration'][key] = value
            with open(file_name_cal_xml, 'w') as fd:
                fd.write(xmltodict.unparse(xml_dict))

class CameraGeometry():
    def __init__(self, pos, rot, time, is_interpolated = False):
        
        self.position_nav = pos
        self.rotation_nav = rot

        self.time = time
        self.IsLocal = False
        self.decoupled = True

        if is_interpolated:
            self.position_nav_interpolated = self.position_nav
            self.rotation_nav_interpolated = self.rotation_nav



    def interpolate(self, time_hsi, minIndRGB, maxIndRGB, extrapolate):
        """"""
        # A simple interpolation of transforms where all images should be aligned.
        # Should also implement a more sensor fusion-like Kalman filter implementation
        self.time_hsi = time_hsi
        if self.decoupled == True:

            if extrapolate == False:
                time_interpolation = time_hsi[minIndRGB:maxIndRGB]
                linearPositionInterpolator = interp1d(self.time, np.transpose(self.position_nav))
                self.position_nav_interpolated = np.transpose(linearPositionInterpolator(time_interpolation))
                linearSphericalInterpolator = Slerp(self.time, self.rotation_nav)
                self.rotation_nav_interpolated = linearSphericalInterpolator(time_interpolation)
            else:
                # Extrapolation of position:
                time_interpolation = time_hsi
                linearPositionInterpolator = interp1d(self.time, np.transpose(self.position_nav), fill_value='extrapolate')
                self.position_nav_interpolated = np.transpose(linearPositionInterpolator(time_interpolation))

                # Extrapolation of orientation/rotation:

                # Synthetizize additional frames
                delta_rot_b1_b2 = (self.rotation_nav[-1].inv()) * (self.rotation_nav[-2])
                delta_time_last = self.time[-1] - self.time[-2]
                time_last = self.time[-1] + delta_time_last
                rot_last = self.rotation_nav[-1] * (delta_rot_b1_b2.inv()) # Assuming a continuation of the rotation

                # Rotation from second to first attitude
                delta_rot_b1_b2 = (self.rotation_nav[0].inv()) * (self.rotation_nav[1])
                delta_time_last = self.time[0] - self.time[1]
                time_first = self.time[0] + delta_time_last # Subtraction
                # Add the rotation from second to first attitude to the first attitude "continue" rotation
                rot_first = self.rotation_nav[0] * (delta_rot_b1_b2.inv())  # Assuming a continuation of the rotation
                time_concatenated = np.concatenate((np.array(time_first).reshape((1,-1)),
                                                    self.time.reshape((1,-1)),
                                                    np.array(time_last).reshape((1,-1))), axis = 1)\
                    .reshape(-1).astype(np.float64)
                rotation_list = [self.rotation_nav]
                rotation_list.append(rot_last)
                rotation_list.insert(0, rot_first)


                rot_vec_first = rot_first.as_rotvec().reshape((1,-1))
                rot_vec_mid = self.rotation_nav.as_rotvec()
                rot_vec_last = rot_last.as_rotvec().reshape((1,-1))

                rotation_vec_tot = np.concatenate((rot_vec_first, rot_vec_mid, rot_vec_last), axis = 0)

                Rotation_tot = RotLib.from_rotvec(rotation_vec_tot)



                time_concatenated = np.array(time_concatenated).astype(np.float64)

                linearSphericalInterpolator = Slerp(time_concatenated, Rotation_tot)

                self.rotation_nav_interpolated = linearSphericalInterpolator(time_interpolation)






        else:
            print('Proper interpolation of transformation with constant velocity and rotation has not yet been implemented')
            print('See https://www.geometrictools.com/Documentation/InterpolationRigidMotions.pdf')
            #self.rotation_nav_interpolated, self.PositionInterpolated = self.interpolateTransforms()
    def intrinsicTransformHSI(self, translation_ref_hsi, rot_hsi_ref_obj):

        # An intrinsic transform is a transformation to another reference frame on the moving body, i.e. the IMU or an RGB cam
        self.position_ecef = self.position_nav_interpolated + self.rotation_nav_interpolated.apply(translation_ref_hsi)

        # Composing rotations. See:
        # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.spatial.transform.Rotation.__mul__.html
        self.rotation_hsi = self.rotation_nav_interpolated * rot_hsi_ref_obj

        self.quaternion_ecef = self.rotation_hsi.as_quat()
        
    def localTransform(self, frame):
        self.IsLocal = True
        self.LocalTransformFrame = frame

        if frame == 'ENU':
            self.position_nav_interpolated = self.Rotation_ecef_enu*self.position_nav_interpolated
            self.rotation_nav_interpolated = self.Rotation_ecef_enu*self.rotation_nav_interpolated
            self.position_nav = self.Rotation_ecef_enu * self.position_nav
            self.rotation_nav = self.Rotation_ecef_enu * self.rotation_nav
            self.rotation_hsi = self.Rotation_ecef_enu * self.rotation_hsi
            self.position_nav = self.Rotation_ecef_enu * self.position_ecef
        elif frame == 'NED':
            self.localPositionInterpolated = self.Rotation_ecef_ned*self.position_nav_interpolated
            self.localRotationInterpolated = self.Rotation_ecef_ned*self.rotation_nav_interpolated
            self.localPosition = self.Rotation_ecef_ned * self.position_nav
            self.localRotation = self.Rotation_ecef_ned * self.rotation_nav
            self.rotation_hsi = self.Rotation_ecef_ned * self.rotation_hsi
            self.position_nav = self.Rotation_ecef_ned * self.position_ecef
        else:
            print('Frame must be ENU or NED')
    def localTransformInverse(self):

        if self.IsLocal:
            self.IsLocal = False
            if self.LocalTransformFrame == 'ENU':
                self.position_nav_interpolated = self.Rotation_ecef_enu.inv()*self.position_nav_interpolated
                self.rotation_nav_interpolated = self.Rotation_ecef_enu.inv()*self.rotation_nav_interpolated
                self.position_nav = self.Rotation_ecef_enu.inv() * self.position_nav
                self.rotation_nav = self.Rotation_ecef_enu.inv() * self.rotation_nav
                self.rotation_hsi = self.Rotation_ecef_enu.inv() * self.rotation_hsi
                self.position_nav = self.Rotation_ecef_enu.inv() * self.position_ecef
            elif self.LocalTransformFrame == 'NED':
                self.localPositionInterpolated = self.Rotation_ecef_ned.inv()* self.position_nav_interpolated
                self.localRotationInterpolated = self.Rotation_ecef_ned.inv()*self.rotation_nav_interpolated
                self.localPosition = self.Rotation_ecef_ned.inv() * self.position_nav
                self.localRotation = self.Rotation_ecef_ned.inv() * self.rotation_nav
                self.rotation_hsi = self.Rotation_ecef_ned.inv() * self.rotation_hsi
                self.position_nav = self.Rotation_ecef_ned.inv() * self.position_ecef
            else:
                print('Frame must be ENU or NED')
        else:
            print('Poses are defined globally already')
    def defineRayDirections(self, dir_local):
        self.rayDirectionsLocal = dir_local

        n = self.position_ecef.shape[0]
        m = dir_local.shape[0]

        self.rayDirectionsGlobal = np.zeros((n, m, 3))

        # Converts data from local frame to global
        for i in range(n):
            self.rayDirectionsGlobal[i, :, :] = self.rotation_hsi[i].apply(dir_local)
    def intersect_with_mesh(self, mesh, max_ray_length, mesh_trans):
        """Intersects the rays of the camera with the 3D triangular mesh

        :param mesh: A mesh object read via the pyvista library
        :type mesh: Pyvista mesh
        :param max_ray_length: The upper bound length of the camera rays (it is determined )
        :type max_ray_length: _type_
        :param mesh_trans: The offset of the mesh
        :type mesh_trans: _type_
        """

        n = self.rayDirectionsGlobal.shape[0]
        m = self.rayDirectionsGlobal.shape[1]

        self.points_ecef_crs = np.zeros((n, m, 3), dtype=np.float64)
        self.normals_ecef_crs = np.zeros((n, m, 3), dtype=np.float64)

        # Duplicate multiple camera centres
        start_ECEF = np.einsum('ijk, ik -> ijk', np.ones((n, m, 3), dtype=np.float64), self.position_ecef).reshape((-1,3))

        # Subtract the mesh offset to avoid rounding errors and perform intersection tests with "small coordinates"
        start = start_ECEF - mesh_trans

        dir = (self.rayDirectionsGlobal * max_ray_length).reshape((-1,3))

        
        
        start_time = time.time()


        
        try:
            # This will only work if exact Python version is rigght and you have PyEmbree
            points, rays, cells = mesh.multi_ray_trace(origins=start, directions=dir, first_point=True, retry=True)
        except:
            # If you instead use embreex, python>3.6 will do
            
            
            # If the faces are 
            faces = mesh.regular_faces

            # Convert PolyData to trimesh.Trimesh
            tri_mesh = trimesh.Trimesh(vertices=mesh.points, faces=faces)

            # Define an intersector object
            ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(geometry=tri_mesh)

            # Intersect data
            count = 0
            cells, rays, points = ray_mesh_intersector.intersects_id(ray_origins=start,  
                                                                ray_directions=dir, 
                                                                multiple_hits=False,
                                                                return_locations=True)
            
        n_points = int(np.size(points)/3)
        n_rays = int(np.size(start)/3)
        # Recurring is that 1- 10s of rays fail to detect intersections using Trimesh.
        # A handy fix is to iterate the failed intersections using individual ray tracing
        # This equivalent to the retry=True in 
        # https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PolyDataFilters.multi_ray_trace.html#pyvista.PolyDataFilters.multi_ray_trace
        if n_points != n_rays:
            n_missing = n_rays - n_points
            print(f'Trimesh failed to find intersections for {n_missing} rays')
            print(f'Ray tracing for these rays is retried in pyvista')

            # Identify the missing intersections:
            missing_rays = np.array(list( set(range(n_rays)) - set(rays) ))

            # And ray trace them individually in VTK
            for ray in missing_rays:
                # Retry failed intersections with slow pyvista version
                point, cell = mesh.ray_trace(start[ray,:], start[ray,:] + dir[ray,:], first_point=True)
                cells = np.concatenate((cells, cell), axis=0)
                points = np.concatenate((points, point.reshape((1,3))), axis=0)
                rays = np.concatenate((rays, np.array([ray])), axis=0)
            n_points = int(np.size(points)/3)
            print('All rays successfully traced in VTK')

        else:
            print(f'All rays were successfully intersected with trimesh')
            


        stop_time = time.time()

        normals = mesh.cell_normals[cells,:]

        slit_image_number = np.floor(rays / m).astype(np.int32)

        pixel_number = rays % m

        


        # Assign normals
        self.points_ecef_crs[slit_image_number, pixel_number] = points + mesh_trans

        self.normals_ecef_crs[slit_image_number, pixel_number] = normals

        self.camera_to_seabed_ECEF = self.points_ecef_crs - start_ECEF.reshape((n, m, 3))


        # Calculate
        self.points_hsi_crs = np.zeros(self.camera_to_seabed_ECEF.shape)

        self.normals_hsi_crs = np.zeros(self.normals_ecef_crs.shape)

        self.depth_map = np.zeros((n, m))

        # For local geometry (when vehicle fixed artificial light is used):
        for i in range(n):
            # Calculate vector from HSI to seabed in local coordinates (for artificial illumination)
            self.points_hsi_crs[i, :, :] = (self.rotation_hsi[i].inv()).apply(self.camera_to_seabed_ECEF[i, :, :])

            # Calculate surface normals of intersected triangles (for artificial illumination)
            self.normals_hsi_crs[i,:,:] = (self.rotation_hsi[i].inv()).apply(self.normals_ecef_crs[i, :, :])

            # Calculate a depth map (the z-component, 1D scanline)
            self.depth_map[i, :] = self.points_hsi_crs[i, :, 2]/self.rayDirectionsLocal[:, 2]

        

        self.unix_time_grid = np.einsum('ijk, ik -> ijk', np.ones((n, m, 1), dtype=np.float64), self.time.reshape((-1,1)))
        self.unix_time = self.unix_time_grid.reshape((-1, 1)) # Vector-form
        self.pixel_nr_grid = np.matlib.repmat(np.arange(m), n, 1)
        self.frame_nr_grid = np.matlib.repmat(np.arange(n).reshape(-1,1), 1, m)

        
    @staticmethod
    def intersect_ray_with_earth_ellipsoid(p0, dir_hat, B):
        """_summary_

        :param p0: Ray origin
        :type p0: (3,1) array of floats 
        :param dir_hat: describing ray direction in some ECEF
        :type dir_hat: (3,1) array of floats 
        :param B: Ellipsoid matrix of earth so that (p0 + lam*dir_hat)' * B * (p0 + lam*dir_hat) = 1
        :type B: (3, 3) matrix describing earth ellipsoid in some ECEF
        """

        # Solves equation lam^2 *(d_hat'B*d_hat) + lam * 2*(p0' * B* dir_hat) + (p0' *B* p0)= 1
        a = np.transpose(dir_hat).dot(B.dot(dir_hat))
        b = 2*np.transpose(p0).dot(B.dot(dir_hat))
        c = np.transpose(p0).dot(B.dot(p0)) - 1

        p = np.zeros(3)
        p[0] = a
        p[1] = b
        p[2] = c

        lam = np.roots(p)

        hits = p0 + dir_hat*lam


        return hits 

    @staticmethod
    def calculate_sun_directions(longitude, latitude, altitude, unix_time, degrees=True):
        # Ensure all inputs are NumPy arrays
        longitude = np.asarray(longitude).flatten()
        latitude = np.asarray(latitude).flatten()
        altitude = np.asarray(altitude).flatten()
        unix_time = np.asarray(unix_time).flatten()

        n = max(longitude.size, latitude.size, altitude.size, unix_time.size)

        # If a single value is provided for any parameter, broadcast it to match the size of the other parameters
        longitude = np.broadcast_to(longitude, (n,))
        latitude = np.broadcast_to(latitude, (n,))
        altitude = np.broadcast_to(altitude, (n,))
        unix_time = np.broadcast_to(unix_time, (n,))

        phi_s = np.zeros(n)
        theta_s = np.zeros(n)

        observer = ephem.Observer()

        for i in range(n):
            # Extract a single value from the arrays
            observer.lon = str(longitude[i])
            observer.lat = str(latitude[i])
            observer.elev = altitude[i]
            
            sun = ephem.Sun()
            observer.date = datetime.utcfromtimestamp(unix_time[i])
            sun.compute(observer)

            phi_s[i] = np.rad2deg(sun.az)
            theta_s[i] = 90 - np.rad2deg(sun.alt)

        if not degrees:
            phi_s = np.deg2rad(phi_s)
            theta_s = np.deg2rad(theta_s)

        return phi_s, theta_s

    def compute_view_directions_local_tangent_plane(self):
        """Takes the intersection points and HSI camera positions and computes the angles from seabed to HSI with respect to the local tangent plane to the ellipsoid. 
        """
        n = self.rayDirectionsGlobal.shape[0]
        m = self.rayDirectionsGlobal.shape[1]

        self.seabed_to_camera_NED = np.zeros(self.camera_to_seabed_ECEF.shape)
        self.normals_ned_crs = np.zeros(self.normals_ecef_crs.shape)
        self.theta_v = np.zeros((n, m))
        self.phi_v = np.zeros((n, m))

        x_ecef = self.points_ecef_crs[:, :, 0].reshape((-1,1))
        y_ecef = self.points_ecef_crs[:, :, 1].reshape((-1,1))
        z_ecef = self.points_ecef_crs[:, :, 2].reshape((-1,1))
        
        lats, lons, alts = pm.ecef2geodetic(x = x_ecef, y = y_ecef, z = z_ecef)

        start = np.einsum('ijk, ik -> ijk', np.ones((n, m, 3), dtype=np.float64), self.position_ecef).reshape((-1,3))

        x_hsi = start[:, 0].reshape((-1,1))
        y_hsi = start[:, 1].reshape((-1,1))
        z_hsi = start[:, 2].reshape((-1,1))

        # Compute vectors from seabed intersections to HSI in NED
        NED = pm.ecef2ned(x= x_hsi, y= y_hsi, z=z_hsi, lat0 = lats, lon0=lons, h0 = alts)

        self.seabed_to_camera_NED = np.hstack((NED[0], NED[1], NED[2])).reshape((n, m, 3))

        # For remote sensing with a sun-lit seafloor:
        for i in range(n):
            
            R_ecef_2_ned = RotLib.from_matrix(rotation_matrix_ecef2ned(lon = lons[i], lat = lats[i]))

            # Calculate vector from HSI to seabed in local tangent plane NED
            #self.camera_to_seabed_NED[i, :, :] = R_ecef_2_ned.apply(self.camera_to_seabed_ECEF[i, :, :])

            # Decompose vector to angles
            polar = cartesian_to_polar(xyz = self.seabed_to_camera_NED[i,:,:])

            self.theta_v[i, :] = polar[:,1]

            self.phi_v[i, :] = polar[:,2]

            # Calculate surface normals of intersected triangles (for artificial illumination)
            self.normals_ned_crs[i, :, :] = R_ecef_2_ned.apply(self.normals_ecef_crs[i, :, :])

            


    def compute_sun_angles_local_tangent_plane(self):
        n = self.rayDirectionsGlobal.shape[0]
        m = self.rayDirectionsGlobal.shape[1]

        x_ecef = self.points_ecef_crs[:, :, 0].reshape((-1,1))
        y_ecef = self.points_ecef_crs[:, :, 1].reshape((-1,1)) 
        z_ecef = self.points_ecef_crs[:, :, 2].reshape((-1,1))

        lats, lons, alts = pm.ecef2geodetic(x = x_ecef, y = y_ecef, z = z_ecef)

        self.lats = lats
        self.lons = lons
        self.alts = alts

        phi_s, theta_s = CameraGeometry.calculate_sun_directions(longitude = lons, latitude = lats, altitude = alts, unix_time = self.unix_time, degrees = True)

        self.phi_s = phi_s.reshape((n, m, 1))

        self.theta_s = theta_s.reshape((n, m, 1))

        

    def compute_elevation_mean_sealevel(self, source_epsg, geoid_path = 'data/world/geoids/egm08_25.gtx'):
        n = self.rayDirectionsGlobal.shape[0]
        m = self.rayDirectionsGlobal.shape[1]

        x_ecef = self.position_ecef[:, 0].reshape((-1,1))
        y_ecef = self.position_ecef[:, 1].reshape((-1,1))
        z_ecef = self.position_ecef[:, 2].reshape((-1,1))

        #lats, lons, alts = pm.ecef2geodetic(x = x_ecef, y = y_ecef, z = z_ecef)

        

        alts_msl = CameraGeometry.elevation_msl(x_ecef, y_ecef, z_ecef, source_epsg=source_epsg, geoid_path = geoid_path)
        

        self.hsi_alts_msl = np.einsum('ijk, ik -> ijk', np.ones((n, m, 1), dtype=np.float32), alts_msl.reshape((-1,1)))

        
    
    @staticmethod
    def elevation_msl(x_ecef, y_ecef, z_ecef, source_epsg, geoid_path):
        """_summary_

        :param x_ecef: _description_
        :type x_ecef: _type_
        :param y_ecef: _description_
        :type y_ecef: _type_
        :param z_ecef: _description_
        :type z_ecef: _type_
        :param source_epsg: _description_
        :type source_epsg: _type_
        :param geoid_path: _description_
        :type geoid_path: _type_
        :return: _description_
        :rtype: _type_
        """
        
        with rasterio.open(geoid_path) as src:

            source_crs = CRS.from_epsg(source_epsg)
                
            target_crs = src.crs

            transformer = Transformer.from_crs(source_crs, target_crs)

            (lat, lon, alt_ell) = transformer.transform(xx=x_ecef, yy=y_ecef, zz=z_ecef)

            undulation = np.zeros(lat.shape)
            # Compute undulation and orthometric height for each point (height above MSL)
            
            for i, x, y in zip(range(lat.size), lon, lat):
                undulation[i] = next(src.sample([(float(x), float(y))]))[0]
            

        
        alt_msl = alt_ell - undulation

        return alt_msl



    

    def compute_tide_level(self, path_tide, tide_format, constant_height = 0):

        n = self.rayDirectionsGlobal.shape[0]
        m = self.rayDirectionsGlobal.shape[1]

        if path_tide == 'Undefined':

            self.hsi_tide_gridded = constant_height*np.ones((n, m, 1))

        else:

            if tide_format == 'NMA':
                try:
                    df_tide = pd.read_csv(path_tide, sep='\s+', parse_dates=[0], index_col=0, comment='#', date_parser=parser.parse)

                    # Convert the datetime index to Unix time and add column
                    df_tide['UnixTime'] = df_tide.index.astype('int64') // 10**9  # Convert nanoseconds to seconds

                    tide_height_NN2000 = 0.01*df_tide['Observations'] # Since in cm

                    tide_timestamp = df_tide['UnixTime']
                    
                    hsi_tide_interp = interp1d(x = tide_timestamp, y= tide_height_NN2000)(x = self.time)

                    # Make into gridded form:
                    self.hsi_tide_gridded = np.einsum('ijk, ik -> ijk', np.ones((n, m, 1), dtype=np.float64), hsi_tide_interp.reshape((-1,1)))
                except:
                    #print('No tide file was found!!')
                    self.hsi_tide_gridded = constant_height*np.ones((n, m, 1))



            else: # A Good place to write parsers for other formats
                TypeError
        
    def write_rgb_point_cloud(self, config, hyp, transect_string, mesh_trans, extrapolate = True, minInd = None, maxInd = None):
        wl_red = float(config['General']['red_wave_length'])
        wl_green = float(config['General']['green_wave_length'])
        wl_blue = float(config['General']['blue_wave_length'])
        dir_point_cloud = config['Absolute Paths']['rgb_point_cloud_folder']

        wavelength_nm = np.array([wl_red, wl_green, wl_blue])

        # Localize the appropriate band indices used for analysis
        band_ind_R = np.argmin(np.abs(wavelength_nm[0] - hyp.band2Wavelength))
        band_ind_G = np.argmin(np.abs(wavelength_nm[1] - hyp.band2Wavelength))
        band_ind_B = np.argmin(np.abs(wavelength_nm[2] - hyp.band2Wavelength))

        if extrapolate == False:
            rgb = hyp.dataCubeRadiance[minInd:maxInd, :, [band_ind_R, band_ind_G, band_ind_B]]
        else:
            rgb = hyp.dataCubeRadiance[:, :, [band_ind_R, band_ind_G, band_ind_B]]


        points = self.points_ecef_crs[self.points_ecef_crs != 0].reshape((-1,3))
        rgb_points = (rgb[self.points_ecef_crs != 0] / rgb.max()).astype(np.float64).reshape((-1,3))

        # Subtract the mesh offset to avoid rounding errors
        points -= mesh_trans


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

    def load_cam_calibration(self, filename_cal, config):
        calHSI = CalibHSI(file_name_cal_xml=filename_cal)  # Generates a calibration object
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

        trans_Q_00 = cameraGeometry.position_ecef[line_nr[:, 0]]
        trans_Q_01 = cameraGeometry.position_ecef[line_nr[:, 1]]
        trans_Q_10 = cameraGeometry.position_ecef[line_nr[:, 2]]
        trans_Q_11 = cameraGeometry.position_ecef[line_nr[:, 3]]

        translationHSI = self.bilinearInterpolationPosition(x1_x=hsiGis.x1_x_hsi, y1_y=hsiGis.y1_y_hsi, trans_Q_00=trans_Q_00, trans_Q_01=trans_Q_01, trans_Q_10=trans_Q_10,
                                                              trans_Q_11=trans_Q_11)

        rot_Q_00 = cameraGeometry.rotation_nav_interpolated[line_nr[:, 0]]
        rot_Q_01 = cameraGeometry.rotation_nav_interpolated[line_nr[:, 1]]
        rot_Q_10 = cameraGeometry.rotation_nav_interpolated[line_nr[:, 2]]
        rot_Q_11 = cameraGeometry.rotation_nav_interpolated[line_nr[:, 3]]

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

    
    # At the time of updated parameters
    def reprojectFeaturesHSI(self):
        rot_hsi_rgb = np.array([self.rot_z, self.rot_y, self.rot_x]) * 180 / np.pi

        self.rotation_hsi_rgb = RotLib.from_euler('ZYX', rot_hsi_rgb, degrees=True)

        self.rotation_hsi = self.rotationRGB * self.rotation_hsi_rgb # Composing rotations.

        self.HSIToFeaturesGlobal = self.point_feature_gt - self.translationHSI

        n = self.HSIToFeaturesGlobal.shape[0]

        self.HSIToFeaturesLocal = np.zeros(self.HSIToFeaturesGlobal.shape)
        for i in range(n):
            self.HSIToFeaturesLocal[i, :] = (self.rotation_hsi[i].inv()).apply(
                self.HSIToFeaturesGlobal[i, :])

            self.HSIToFeaturesLocal[i, :] /= self.HSIToFeaturesLocal[i, 2]




def compute_camera_rays_from_parameters(pixel_nr, cx, f, k1, k2, k3):
    """_summary_

    :param pixel_nr: _description_
    :type pixel_nr: _type_
    :param rot_x: _description_
    :type rot_x: _type_
    :param rot_y: _description_
    :type rot_y: _type_
    :param rot_z: _description_
    :type rot_z: _type_
    :param cx: _description_
    :type cx: _type_
    :param f: _description_
    :type f: _type_
    :param k1: _description_
    :type k1: _type_
    :param k2: _description_
    :type k2: _type_
    :param k3: _description_
    :type k3: _type_
    :return: _description_
    :rtype: _type_
    """

    u = pixel_nr

    # Express uhi ray directions in uhi frame using line-camera model
    x_norm_lin = (u - cx) / f

    x_norm_nonlin = -(k1 * (u - cx)** 5 + \
                        k2 * (u - cx) ** 3 + \
                        k3 * (u - cx) ** 2) / f

    x_norm = x_norm_lin + x_norm_nonlin

    return x_norm


def reproject_world_points_to_hsi_plane(trans_hsi_body, rot_hsi_body, pos_body, rot_body, points_world):
    """Reprojects world points to local image plane coordinates (x_h, y_h, 1)

    :param trans_hsi_body: lever arm from body center to hsi focal point
    :type trans_hsi_body: _type_
    :param rot_hsi_body: boresight rotation 
    :type rot_hsi_body: Scipy rotation object
    :param pos_body: The earth fixed earth centered position of the vehicle body centre
    :type pos_body: _type_
    :param rot_body: The rotation of the body in ECEF
    :type rot_body: Scipy rotation object
    :param points_world: World points in ECEF
    :type points_world: _type_
    :return: The reprojection coordinates
    :rtype: ndarray(n, 3)
    """

    

    # We compose the hypothesized boresight (rot_hsi_body) an lever arm (trans_hsi_body) to get the hypothesized position/orientation
    # of the hyperspectral imager
    rotation_hsi = rot_body * rot_hsi_body
    position_hsi = pos_body + rot_body.apply(trans_hsi_body)

    # Given the positions, the hsi-point vector can be expressed in ECEF
    hsi_to_feature_global = points_world - position_hsi

    # Number of features to iterate
    n = hsi_to_feature_global.shape[0]

    # We can now express the hsi-point vector in the HSI frame and normalize by the z component to find the image plane projection
    hsi_to_feature_local = np.zeros(hsi_to_feature_global.shape)
    for i in range(n):
        # The vector expressed in HSI frame
        hsi_to_feature_local[i, :] = (rotation_hsi[i].inv()).apply(hsi_to_feature_global[i, :])

        # The vector normalized to lie on the virtual plane
        hsi_to_feature_local[i, :] /= hsi_to_feature_local[i, 2]
    
    return hsi_to_feature_local




def interpolate_poses(timestamp_from, pos_from, rot_from, timestamps_to, extrapolate = True):
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
    """
    min_to = np.min(timestamps_to)
    max_to = np.max(timestamps_to)

    min_from = np.min(timestamp_from)
    max_from = np.max(timestamp_from)

    minInd = np.argmin(np.abs(min_from - timestamps_to))
    maxInd = np.argmin(np.abs(max_from - timestamps_to))

    # If navigation data aka "from" is made properly, min_from<<min_to, and max_from >> max_to
    
    if timestamps_to[minInd] < min_from:
        minInd += 1
    if timestamps_to[maxInd] > min_from: # So if the max index is 
        maxInd -= 1
    """

    min_ind = 0
    max_ind = timestamps_to.size


    # Setting use_absolute_position to True means that position calculations are done with absolute
    referenceGeometry = CameraGeometry(pos=pos_from,
                               rot=rot_from,
                               time=timestamp_from)

    # We exploit a method from the camera object
    referenceGeometry.interpolate(time_hsi=timestamps_to,
                          minIndRGB=min_ind,
                          maxIndRGB=max_ind,
                          extrapolate=extrapolate)



    position_to = referenceGeometry.position_nav_interpolated

    quaternion_to = referenceGeometry.rotation_nav_interpolated.as_quat()

    return position_to, quaternion_to




def cartesian_to_polar(xyz):
    """Converts from 3D cartesian coordinates to polar coordinates

    :param xyz: _description_
    :type xyz: (n,3) numpy array
    :return: Radii, Elevations, Azimuths
    :rtype: (n,3) numpy array of radii, theta, phi
    """
    polar = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    polar[:,0] = np.sqrt(xy + xyz[:,2]**2) # Radii
    polar[:,1] = np.arctan2(np.sqrt(xy), np.abs(xyz[:,2])) # for elevation angle defined from Z-axis down [0-90]
    polar[:,2] = np.arctan2(xyz[:,1], xyz[:,0]) # Azimuth

    return polar










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

    R_ecef_ned = rotation_matrix_ecef2ned(lon=lon, lat=lat)
    R_ecef_enu = np.zeros(R_ecef_ned.shape)
    #
    R_ecef_enu[[0, 1]] = R_ecef_ned[[1, 0]] # Swap rows
    R_ecef_enu[2] = -R_ecef_ned[2] # Switch signs to compensate for up and down


    
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
    """
    An object that allows to input positions with arbitrary CRS, and orientations either wrt ECEF or NED. As in other
    places in the project, the orientations are abstracted with rotation objects. The main use case is formatting poses
    to the correct CRS.
    """
    def __init__(self, timestamps, rot_obj, rot_ref, pos, pos_epsg):
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


        self.lat = None
        self.lon = None
        self.hei = None

        self.compute_geodetic_position()
        self.compute_geocentric_orientation()
        self.compute_ned_orientation()
        self.compute_ned_2_ecef()


    def compute_geodetic_position(self, epsg_geod = 4326):
        """
            Function for transforming positions to latitude longitude height
            :param epsg_geod: int
            EPSG code of the transformed geodetic coordinate system
        """
        # If geocentric position has not been defined.
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

        self.pos_geocsc = np.concatenate((x, y, z), axis = 1)

    def compute_geocentric_orientation(self):
        if self.rot_obj_ecef == None:
            # Define rotations from ned to ecef
            if self.lat.any == None:
                # Needed to encode rotation between NED and ECEF
                self.compute_geodetic_position()

            R_body_2_ned = self.rot_obj_ned
            R_ned_2_ecef = self.compute_ned_2_ecef()
            R_body_2_ecef = R_ned_2_ecef*R_body_2_ned

            self.rot_obj_ecef = R_body_2_ecef

        else:
            pass

    def compute_ned_orientation(self):
        if self.rot_obj_ned == None:
            # Define rotations from body to ecef
            R_body_2_ecef = self.rot_obj_ecef

            # Define rotation from ecef 2 ned
            R_ecef_2_ned = self.compute_ned_2_ecef().inv()

            # Compose
            R_body_2_ned = R_ecef_2_ned*R_body_2_ecef

            self.rot_obj_ned = R_body_2_ned


        else:
            pass

    def compute_ned_2_ecef(self):

        N = self.lat.shape[0]
        rot_mats_ned_2_ecef = np.zeros((N, 3, 3), dtype=np.float64)
        
        for i in range(N):
            rot_mats_ned_2_ecef[i,:,:] = rot_mat_ned_2_ecef(lat=self.lat[i], lon = self.lon[i])


        self.rot_obj_ned_2_ecef = RotLib.from_matrix(rot_mats_ned_2_ecef)

        return self.rot_obj_ned_2_ecef

    


def rot_mat_ned_2_ecef(lat, lon):
    """
    Computes the rotation matrix R from north-east-down (NED) to earth-fixed-earth centered (ECEF)
    :param lat: float in range [-90, 90]
    The latitude in degrees
    :param lon: float in range [-180, 180]
    :return R_ned_ecef: numpy array of shape (3,3)
    rotation matrix from NED to ECEF
    """

    # Ensure lat and lon are scalar values
    lat_scalar = lat[0] if isinstance(lat, np.ndarray) else lat
    lon_scalar = lon[0] if isinstance(lon, np.ndarray) else lon

    # Convert to radians
    l = np.deg2rad(lon_scalar)
    mu = np.deg2rad(lat_scalar)

    # Compute rotation matrix
    # TODO: add source
    R_ned_ecef = np.array([[-np.cos(l) * np.sin(mu), -np.sin(l), -np.cos(l) * np.cos(mu)],
                           [-np.sin(l) * np.sin(mu), np.cos(l), -np.sin(l) * np.cos(mu)],
                           [np.cos(mu), 0, -np.sin(mu)]])
    return R_ned_ecef


def read_raster(filename, out_crs="EPSG:3857", use_z=False):
    """Read a raster to a ``pyvista.StructuredGrid``.

    This will handle coordinate transformations.
    """
    from rasterio import transform
    import rioxarray
    # Read in the data
    data = rioxarray.open_rasterio(filename)
    values = np.asarray(data)
    data.rio.nodata
    nans = values == data.rio.nodata
    if np.any(nans):
        # values = np.ma.masked_where(nans, values)
        values[nans] = np.nan
    # Make a mesh
    xx, yy = np.meshgrid(data["x"], data["y"])
    if use_z and values.shape[0] == 1:
        # will make z-comp the values in the file
        zz = values.reshape(xx.shape)
    else:
        # or this will make it flat
        zz = np.zeros_like(xx)
    mesh = pv.StructuredGrid(xx, yy, zz)
    pts = mesh.points
    lon, lat = transform(data.rio.crs, out_crs, pts[:, 0], pts[:, 1])
    mesh.points[:, 0] = lon
    mesh.points[:, 1] = lat
    mesh["data"] = values.reshape(mesh.n_points, -1, order="F")
    return mesh

def dem_2_mesh(path_dem, model_path, config, dem_ref_is_geoid = False, path_geoid= None, config_file_path = None):
    """
    A function for converting a specified DEM to a 3D mesh model (*.vtk, *.ply or *.stl). 
    Consequently, mesh should be thought of as 2.5D representation.
    In the case 

    :param path_dem: _description_
    :type path_dem: _type_
    :param model_path: _description_
    :type model_path: _type_
    :param config: _description_
    :type config: _type_
    """

    ## Step 0: do a size check
    with rasterio.open(path_dem) as src:
        w = src.width
        h = src.height
    
    dem_folder = config['Absolute Paths']['dem_folder']
    
    # Unfortunately chrashes have been observed on a local machine. 1, 2, 3, 4M has worked.
    # Chrashes occur at the delaunay_2d() call further down
    # Observed at 5M, 10M
    n_points_max = 8e6
    if w*h > n_points_max:
        resample_factor = np.ceil(np.sqrt(w*h/n_points_max)).astype(np.int64)
        # Make a new file
        resampled_dem_path = os.path.join(dem_folder, 'dem_resampled.tif')

        _resample_raster(raster_path=path_dem, 
                         resample_factor=resample_factor,
                         output_path=resampled_dem_path)
        
        # Update the path_dem
        path_dem = resampled_dem_path

        # Update for config file
        config['Absolute Paths']['dem_path'] = path_dem

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)
    ##



    # Step 1: If the dem is given with respect to Geoid, the two rasters (DEM and GEoid) need to be added
    if dem_ref_is_geoid:
        # To add reasters, first crop Geoid to grid of DEM
        geoid_cropped_to_dem = os.path.join(dem_folder, 'geoid_cropped_to_dem.tif')

        # Resample geoid to shape of DEM
        geohsi.resample_dem_to_hsi_ortho(path_geoid, path_dem, geoid_cropped_to_dem)

        dem_wrt_ellipsoid = os.path.join(dem_folder, 'dem_wrt_ellipsoid.tif')

        # Add them together to get height wrt ellipsoid
        add_rasters_with_nodata_mask(raster1_path = geoid_cropped_to_dem, raster2_path = path_dem, output_path=dem_wrt_ellipsoid)

        # Update dem patha
        path_dem = dem_wrt_ellipsoid

        # Update for config file
        config['Absolute Paths']['dem_path'] = path_dem

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

        
    
    


    # The desired CRS for the model must be same as positions, orientations
    epsg_geocsc = config['Coordinate Reference Systems']['geocsc_epsg_export']

    # Intermediate point cloud format
    output_xyz = model_path.split(sep = '.')[0] + '.xyz'

    # Open the input raster dataset
    ds = gdal.Open(path_dem)

    if ds is None:
        print(f"Failed to open {path_dem}")
    else:
        # Read the first band (band index is 1)
        band = ds.GetRasterBand(1)
        no_data_value = band.GetNoDataValue()
        if band is None:
            print(f"Failed to open band 1 of {path_dem}")
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
            epsg_proj = None
            if spatial_reference.IsProjected():
                epsg_proj = spatial_reference.GetAttrValue("AUTHORITY", 1)
                is_projected = True
                config.set('Coordinate Reference Systems', 'dem_epsg', str(epsg_proj))
            elif spatial_reference.IsGeographic():
                epsg_proj = spatial_reference.GetAttrValue("AUTHORITY", 0)
                proj = ds.GetProjection()
                is_projected = False


            # Automatically set
            
            
            # Get the band's data as a NumPy array of float64 (important)
            band_data = band.ReadAsArray().astype(np.float64)

            # Create a mask to identify no-data values
            mask = band_data != no_data_value

            # Create and open the output XYZ file for writing if it does not exist:
            if not os.path.exists(output_xyz):
                with open(output_xyz, 'w') as xyz_file:
                    # Write data to the XYZ file using the mask and calculated coordinates
                    for y in range(ds.RasterYSize):
                        for x in range(ds.RasterXSize):
                            if mask[y, x]:
                                x_coord = x_origin + x * x_resolution
                                y_coord = y_origin + y * y_resolution
                                xyz_file.write(f"{x_coord} {y_coord} {band_data[y, x]}\n")
                    



                    # Step 2 insert corners of geoid file to ensure that all rays hit the target. Helps to ensure all ray intersections
                    try:
                        add_geoid_corners = eval(config['General']['add_geoid_corners'])
                    except:
                        add_geoid_corners = False
                    if add_geoid_corners:
                        # Ensure that 3D model is computed with padding, i.e. add some far-away corners at sea level to fill in terrain model
                        geoid_cropped_with_padding_file = os.path.join(dem_folder, 'geoid_cropped_with_padding.tif')

                        crop_geoid_to_pose(geoid_cropped_with_padding_file, config, geoid_path = path_geoid)

                        # Find corners of a DEM in DEM CRS and write to point cloud 
                        corners_3d = _extract_ecef_corners(raster_path=geoid_cropped_with_padding_file, ecef_epsg=epsg_proj)
                        for i in range(4):
                            x_coord, y_coord, z_coord = corners_3d[i]
                            xyz_file.write(f"{x_coord} {y_coord} {z_coord}\n")

            else:
                print('*.xyz already exists, ignoring re-creation')
            # Clean up
            ds = None
            band = None
    

    
    
    # TODO: Apply patch to avoid crash for triangulation when using big DEM files
    # Generate a mesh from points

    ## Old implementation
    #mesh = cloud.delaunay_2d(progress_bar=True)
    
    # Due to some unpredictable errors leading to silent exits, we need to test the delaunay_2d()
    is_working = False

    mesh, points_offset = _run_delaunay_2d_in_separate_process(output_xyz)




    

    
    
    #mesh = cloud.reconstruct_surface(progress_bar=True)

    # Transform the mesh points from projected to geocentric ECEF.
    geocsc = CRS.from_epsg(epsg_geocsc)

    if epsg_proj == 'EPSG': # If a geographic CRS
        pass
    else:
        proj = CRS.from_epsg(epsg_proj)

    

    transformer = Transformer.from_crs(proj, geocsc)

    # Convert to proper coordinates
    points_proj = mesh.points + points_offset

    if is_projected:
        x_proj = points_proj[:, 0].reshape((-1, 1))
        y_proj = points_proj[:, 1].reshape((-1, 1))
    else:
        x_proj = points_proj[:, 1].reshape((-1, 1))
        y_proj = points_proj[:, 0].reshape((-1, 1))
    
    h_proj = points_proj[:, 2].reshape((-1, 1))

    (x_ecef, y_ecef, z_ecef) = transformer.transform(xx=x_proj, yy=y_proj, zz=h_proj)

    # Introduce mesh offsets in ECEF sized numbers seem to cause troubles:
    offset_x = np.mean(x_ecef.reshape(-1))
    offset_y = np.mean(y_ecef.reshape(-1))
    offset_z = np.mean(z_ecef.reshape(-1))

    mesh.points[:, 0] = x_ecef.reshape(-1) - offset_x
    mesh.points[:, 1] = y_ecef.reshape(-1) - offset_y
    mesh.points[:, 2] = z_ecef.reshape(-1) - offset_z
   

    # Save mesh
    mesh.save(model_path)
    # Save mesh meta
    # Define your metadata dictionary
    metadata = {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "offset_z": offset_z,
        "epsg_code": geocsc.to_epsg(),  # Example EPSG code, replace with your actual code
        "data_type": str(mesh.points.dtype),  # Add other metadata entries here
    }

    model_meta_path = model_path.split('.')[0] + '_meta.json' 
    # Open the file in write mode with proper indentation
    with open(model_meta_path, "w") as f:
        # Write the dictionary to the file using JSON dump
        json.dump(metadata, f)


def crop_geoid_to_pose(path_dem, config, geoid_path = 'data/world/geoids/egm08_25.gtx'):
    """Crops out a DEM from the geoid using the pose of the vehicle

    :param path_dem: where to write the new dem
    :type path_dem: str
    :param config: dictionary of configurations
    :type config: dictionary from *.ini file
    :param geoid_path: Where the geoid to use comes from, defaults to 'data/world/geoids/egm08_25.gtx'
    :type geoid_path: str, optional
    """
    # The desired CRS for the model must be same as positions, orientations
    epsg_geocsc = config['Coordinate Reference Systems']['geocsc_epsg_export']

    # Open the input raster dataset
    
    ds = gdal.Open(geoid_path)

    df_pose = pd.read_csv(config['Absolute Paths']['pose_path'])

    # Find CRS of DEM
    spatial_reference = osr.SpatialReference(ds.GetProjection())


    # Get the EPSG code
    epsg_proj = None
    if spatial_reference.IsProjected():
        epsg_proj = spatial_reference.GetAttrValue("AUTHORITY", 1)
    elif spatial_reference.IsGeographic():
        epsg_proj = spatial_reference.GetAttrValue("AUTHORITY", 0)
    
    # Transform points to DEM CRS
    geocsc = CRS.from_epsg(epsg_geocsc)
    proj = ds.GetProjection()
    transformer = Transformer.from_crs(geocsc, proj)

    x_ecef = df_pose[' X'].values.reshape((-1, 1))
    y_ecef = df_pose[' Y'].values.reshape((-1, 1))
    z_ecef = df_pose[' Z'].values.reshape((-1, 1))

    (x_proj, y_proj, z_proj) = transformer.transform(xx=x_ecef, yy=y_ecef, zz=z_ecef)

    coords_horz = np.concatenate((x_proj.reshape((-1,1)), y_proj.reshape((-1,1))), axis = 1)

    # Determine bounding rectangle of poses ()
    polygon = MultiPoint(coords_horz).envelope
    
    # Corners
    x, y = polygon.exterior.xy
    x = np.array(x)
    y = np.array(y)

    # Determine padding
    padding = float(config['General']['max_ray_length'])

    if spatial_reference.IsProjected():
        # Add padding to 
        minx = x.min() - padding
        miny = y.min() - padding
        maxx = x.max() + padding
        maxy = y.max() + padding
    elif spatial_reference.IsGeographic():
        # Must translate metric padding into increments in lon/lat
        (x_new, y_new, z_new) = pm.enu2geodetic(e= np.array([-padding, padding]), n = np.array([-padding, padding]), u = 0, lon0=np.mean(y), lat0=np.mean(x), h0 = 0)
        delta_x = x_new[1] - np.mean(x)
        delta_y = y_new[1] - np.mean(y)

        # Swap x and y because x, y above is latitude, longitude and Rasterio expects opposite
        miny = x.min() - delta_x
        maxx = y.min() - delta_y
        maxy = x.max() + delta_x
        minx = y.max() + delta_y


        

    # Crops the DEM to the appropriate bounds and writes a new file (Copies CRS info from Geoid)
    crop_dem_from_bounds(minx, miny, maxx, maxy, dem_path_source=geoid_path, dem_path_target=path_dem)
     


def crop_dem_from_bounds(minx, miny, maxx, maxy, dem_path_source, dem_path_target):

    dem_dataset = rasterio.open(dem_path_source)
    res_x = dem_dataset.transform.a
    res_y = -dem_dataset.transform.e

    if maxx-minx < res_x:
        minx += -res_x
        maxx += res_x
    
    if maxy-miny < res_y:
        miny += -res_y
        maxy += res_y

    window = from_bounds(minx, miny, maxx, maxy, dem_dataset.transform)

    window = Window(window.col_off, window.row_off, np.ceil(window.width), np.ceil(window.height))

    cropped_dem_data = dem_dataset.read(window=window)

    # Update the transform based on the new window
    new_transform = dem_dataset.window_transform(window)
    new_height, new_width = cropped_dem_data.shape[1], cropped_dem_data.shape[2]

    # Create a new dataset for the cropped DEM
    cropped_dem_profile = dem_dataset.profile.copy()
    cropped_dem_profile.update({
        'width': new_width,
        'height': new_height,
        'transform': new_transform
    })


    with rasterio.open(dem_path_target, 'w', **cropped_dem_profile) as dst:
        dst.write(cropped_dem_data)


def position_transform_ecef_2_llh(position_ecef, epsg_from, epsg_to, config):
    """
    Function for transforming ECEF positions to latitude longitude height
    :param position_ecef: numpy array floats (n,3)
    :param epsg_from: int
    EPSG code of the original geocentric coordinate system (ECEF)
    :param epsg_to: int
    EPSG code of the transformed geodetic coordinate system
    :return lat_lon_hei: numpy array floats (n,3)
    latitude, longitude ellipsoid height.
    """
    geocsc = CRS.from_epsg(epsg_from)
    geod = CRS.from_epsg(epsg_to)
    transformer = Transformer.from_crs(geocsc, geod)

    x_ecef = position_ecef[:, 0].reshape((-1, 1))
    y_ecef = position_ecef[:, 1].reshape((-1, 1))
    z_ecef = position_ecef[:, 2].reshape((-1, 1))

    (lat, lon, hei) = transformer.transform(xx=x_ecef, yy=y_ecef, zz=z_ecef)

    lat_lon_hei = np.zeros(position_ecef.shape)
    lat_lon_hei[:, 0] = lat.reshape((position_ecef.shape[0], 1))
    lat_lon_hei[:, 1] = lon.reshape((position_ecef.shape[0], 1))
    lat_lon_hei[:, 2] = hei.reshape((position_ecef.shape[0], 1))

    return lat_lon_hei

def add_rasters_with_nodata_mask(raster1_path, raster2_path, output_path):
    """Adds two rasters with "NoData" masking. The input rasters must have the same CRS and geotransform

    Args:
        raster1_path (str): Path to the first raster.
        raster2_path (str): Path to the second raster.
        output_path (str): Path to save the output raster.
    """

    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:

        # Create a new dataset with the same profile
        profile = src1.profile.copy()
        with rasterio.open(output_path, 'w', **profile) as dst:
            # Read raster data
            array1 = src1.read(1)
            array2 = src2.read(1)

            # Get NoData values
            nodata1 = src1.nodata
            nodata2 = src2.nodata

            # Create masks
            mask1 = (array1 != nodata1)
            mask2 = (array2 != nodata2)

            # Combine masks
            combined_mask = mask1 & mask2

            # Apply masks and add
            result = (array1 * combined_mask) + (array2 * combined_mask)

            # Write to output raster
            dst.write(result, 1)
def _resample_raster(raster_path, resample_factor, output_path):
    """Resamples a raster by a given factor and saves the result to a new file.

    Args:
        raster_path: Path to the input raster file.
        resample_factor: Factor by which to resample the raster.
        output_path: Path to the output raster file.

    Returns:
        None.
    """

    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        transform = src.transform

        new_width, new_height = int(width / resample_factor), int(height / resample_factor)

        # Calculate new cell size based on resampling and original extent
        original_cell_x = transform[0]
        original_cell_y = transform[4]

        new_cell_x = original_cell_x * resample_factor
        new_cell_y = original_cell_y * resample_factor

        print(f"Resampling raster from {width}x{height} to {new_width}x{new_height}")

        # Create a new dataset with the resampled dimensions
        profile = src.profile.copy()
        profile.update(width=new_width, height=new_height, transform=[
            new_cell_x, transform[1], transform[2],
            transform[3], new_cell_y, transform[5]
        ])
        with rasterio.open(output_path, "w", **profile) as dst:
            data = src.read()
            dst.write(data)

            # Resample the data using nearest neighbor resampling
            #resampled_data = dst.read()
            #dst.write(resampled_data)

def _extract_ecef_corners(raster_path, ecef_epsg = "EPSG:4979"):
    """Extracts the corners of a DEM and returns their ECEF coordinates.

    Args:
        raster_path (str): Path to the DEM raster file.

    Returns:
        list: A list of ECEF coordinates (x, y, z) for the four corners.
    """

    with rasterio.open(raster_path) as src:
        # Get the raster's CRS and bounds
        crs = src.crs
        left, bottom, right, top = src.bounds
        elevation_data = src.read(1)  # Assuming elevation is in the first band


        # Calculate corner coordinates
        corners = [
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom)
        ]
        w,h  = src.width, src.height
        corner_indices = [
            (h-1, 0),
            (0, 0),
            (0, w-1),
            (h-1, w-1)
        ]
        elevations = [elevation_data[int(row), int(col)] for row, col in corner_indices]


        # Convert to ECEF coordinates using pyproj
        ecef_corners = []
        transformer = pyproj.Transformer.from_crs(crs, ecef_epsg, always_xy=True)
        for xy, elevation in zip(corners, elevations):
            x, y, z = transformer.transform(xy[0], xy[1], elevation)
            ecef_corners.append((x, y, z))

    return ecef_corners

def _run_delaunay_2d_in_separate_process(output_xyz):
    """A wrapper around the call of the delaunay_2d(), which keeps on trying with half resolution. 

    :param output_xyz: _description_
    :type output_xyz: _type_
    :return: _description_
    :rtype: _type_
    """
    # Create a subprocess to run the function
    python_cmd_str = f'import pyvista as pv; import numpy as np; points = np.loadtxt(r"{output_xyz}"); points_offset = np.mean(points, axis = 0); cloud = pv.PolyData(points-points_offset); mesh = cloud.delaunay_2d(); print("delaunay_2d completed successfully")'

    command = [
    'python',
    '-c',
    python_cmd_str
    ]

    is_working = False

    while not is_working:
        process = subprocess.Popen(command, stdout=subprocess.PIPE )
    
        for line in process.stdout:
            msg = line.decode().rstrip()
            if msg == 'delaunay_2d completed successfully':
                is_working = True
        
        # If it fails, rewrite the xyz-file with half the points 
        if not is_working:
            points = np.loadtxt(output_xyz)
            # Take 
            point_no_corn_half = points[0:-5:2]
            corners = points[-5::]
            points_half = np.row_stack((point_no_corn_half, corners)) # Re
            n_points = points.shape[0]
            print(f'{n_points} points failed in delaunay triangulation')
            _write_xyz_file(file_path=output_xyz, data = points_half)
        else:
            points = np.loadtxt(output_xyz)
            n_points = points.shape[0]
            print(f'{n_points} points succeeded in delaunay triangulation')

            points_offset = np.mean(points, axis = 0)
            # Create a pyvista point cloud object (just to avoid precision problems)
            cloud = pv.PolyData(points-points_offset)

            mesh = cloud.delaunay_2d()
            is_working = True

    return mesh, points_offset



def _write_xyz_file(file_path, data):
    """Writes a NumPy array to an XYZ file.

    Args:
        file_path (str): Path to the output XYZ file.
        data (np.ndarray): The XYZ data to write.
    """

    with open(file_path, 'w') as f:
        for row in data:
            f.write(' '.join(map(str, row)) + '\n')