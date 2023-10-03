import cv2 as cv
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from osgeo import gdal
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon
from shapely.geometry import mapping
from sklearn.neighbors import NearestNeighbors
from spectral import envi
import os

from colours import Image as Imcol


class GeoSpatialAbstractionHSI():
    def __init__(self, point_cloud, datacube_indices, transect_string, config):
        self.config = config
        self.name = transect_string
        self.points_geocsc = point_cloud
        #self.cube_indices = datacube_indices
        self.is_global = self.config['Coordinate Reference Systems']['ExportEPSG'] != 'Local'
        if self.is_global:
            self.epsg_geocsc = int(config['Coordinate Reference Systems']['ExportEPSG'].split(sep = '::')[1])

            self.epsg_proj = int(config['Coordinate Reference Systems']['gisEPSG'].split(sep = '::')[1])
    def transform_geocentric_to_projected(self):
        self.points_proj  = self.points_geocsc # Remains same if it is local
        if self.is_global:
            geocsc = CRS.from_epsg(self.epsg_geocsc)
            proj = CRS.from_epsg(self.epsg_proj)
            transformer = Transformer.from_crs(geocsc, proj)

            xECEF = self.points_geocsc[:,:,0].reshape((-1, 1))
            yECEF = self.points_geocsc[:, :, 1].reshape((-1, 1))
            zECEF = self.points_geocsc[:, :, 2].reshape((-1, 1))

            self.offX = float(self.config['General']['offsetX'])
            self.offY = float(self.config['General']['offsetY'])
            self.offZ = float(self.config['General']['offsetZ'])

            (east, north, hei) = transformer.transform(xx=xECEF + self.offX, yy=yECEF + self.offY, zz=zECEF + self.offZ)

            self.points_proj[:,:,0] = east.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))
            self.points_proj[:, :, 1] = north.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))
            self.points_proj[:, :, 2] = hei.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))
    def footprint_to_shape_file(self):
        self.edge_start = self.points_proj[0, :, 0:2].reshape((-1,2))
        self.edge_end = self.points_proj[-1, :, 0:2].reshape((-1,2))
        self.side_1 = self.points_proj[:, 0, 0:2].reshape((-1,2))
        self.side_2 = self.points_proj[:, -1, 0:2].reshape((-1,2))


        # Do it clockwise
        self.hull_line = np.concatenate((
            self.edge_start,
            self.side_2,
            np.flip(self.edge_end, axis=0),
            np.flip(self.side_1, axis = 0)

        ), axis = 0)


        self.footprint_shp = Polygon(self.hull_line)

        if self.is_global:
            self.crs = 'EPSG:' + str(self.epsg_proj)
        else:
            # Our frame is a local engineering frame (local tangent plane)
            wkt = self.config['Coordinate Reference Systems']['wktLocal']
            #ellps = self.config['Coordinate Reference Systems']['ellps']
            #geo_dict = {'proj':'utm', 'zone': 10, 'ellps': ellps}
            #self.crs = pyproj.CRS.from_dict(proj_dict=geo_dict)
            self.crs = pyproj.CRS.from_wkt(wkt)

        gdf = gpd.GeoDataFrame(geometry=[self.footprint_shp], crs=self.crs)

        shape_path = self.config['Georeferencing']['footPrintPaths'] + self.name + '.shp'

        gdf.to_file(shape_path, driver='ESRI Shapefile')
    def resample_datacube(self, hyp, rgb_composite, minInd, maxInd, extrapolate = True):
        #
        self.res = float(self.config['Georeferencing']['resolutionHyperspectralMosaic'])
        wl_red = float(self.config['General']['RedWavelength'])
        wl_green = float(self.config['General']['GreenWavelength'])
        wl_blue = float(self.config['General']['BlueWavelength'])

        rgb_composite_path = self.config['Georeferencing']['rgbCompositePaths']
        datacube_path = self.config['Georeferencing']['orthoCubePaths']
        resamplingMethod = self.config['Georeferencing']['resamplingMethod']
        
        # The footprint-shape is a in a vectorized format and needs to be mapped into a raster-mask
        xmin, ymin, xmax, ymax = self.footprint_shp.bounds
        width = int((xmax - xmin) / self.res)
        height = int((ymax - ymin) / self.res)
        transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

        # Create mask from the polygon
        geoms = [mapping(self.footprint_shp)]
        mask = geometry_mask(geoms, out_shape=(height, width), transform=transform)

        # Set custom RGB bands from *.ini file
        wavelength_nm = np.array([wl_red, wl_green, wl_blue])
        band_ind_R = np.argmin(np.abs(wavelength_nm[0] - hyp.band2Wavelength))
        band_ind_G = np.argmin(np.abs(wavelength_nm[1] - hyp.band2Wavelength))
        band_ind_B = np.argmin(np.abs(wavelength_nm[2] - hyp.band2Wavelength))
        n_bands = len(hyp.band2Wavelength)

        if rgb_composite:
            if extrapolate == False:
                datacube = hyp.dataCubeRadiance[minInd:maxInd, :, :].reshape((-1, n_bands))
                rgb_cube = datacube[:, [band_ind_R, band_ind_G, band_ind_B]].reshape((-1, 3))

            elif extrapolate == True:
                datacube = hyp.dataCubeRadiance[:, :, :].reshape((-1, n_bands))
                rgb_cube = datacube[:, [band_ind_R, band_ind_G, band_ind_B]].reshape((-1, 3))
            transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

            # Horizontal coordinates of intersections
            coords = self.points_proj[:, :, 0:2].reshape((-1, 2))
            if resamplingMethod == 'Nearest':
                tree = NearestNeighbors(radius=0.01).fit(coords)
                xi, yi = np.meshgrid(np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height))
                xy = np.vstack((xi.flatten(), yi.flatten())).T
                dist, indexes = tree.kneighbors(xy, 1)

                # Build the RGB cube from the indices
                ortho_rgb = rgb_cube[indexes, :].flatten()
                # Build datacube
                ortho_datacube = datacube[indexes, :].flatten()

                ortho_rgb = np.flip(ortho_rgb.reshape((height, width, 3)).astype(np.float64), axis = 0)
                ortho_datacube = np.flip(ortho_datacube.reshape((height, width, n_bands)).astype(np.float64), axis=0)

                self.width_rectified = width
                self.height_rectified = height
                self.indexes = indexes




            # Set nodata value
            nodata = -9999
            ortho_rgb[mask == 1, :] = nodata
            ortho_datacube[mask == 1, :] = nodata

            # Arange datacube or composite in rasterio-friendly structure
            ortho_rgb = np.transpose(ortho_rgb, axes = [2, 0, 1])
            ortho_datacube = np.transpose(ortho_datacube, axes=[2, 0, 1])


            # Write pseudo-RGB composite to composite folder ../GIS/RGBComposites
            with rasterio.open(rgb_composite_path + self.name + '.tif', 'w', driver='GTiff',
                                   height=height, width=width, count=3, dtype=np.float64,
                                   crs=self.crs, transform=transform, nodata=nodata) as dst:

                dst.write(ortho_rgb)
            # Write pseudo-RGB composite to composite folder ../GIS/RGBComposites
            self.write_datacube_ENVI(ortho_datacube, nodata, transform, datacube_path = datacube_path + self.name, wavelengths=hyp.band2Wavelength)




        else:
            print('The software does not support writing a full datacube yet')

    def write_datacube_ENVI(self, ortho_datacube, nodata, transform, datacube_path, wavelengths):
        nx = ortho_datacube.shape[1]
        mx = ortho_datacube.shape[2]
        k = ortho_datacube.shape[0]

        # Create the bsq file
        with rasterio.open(datacube_path + '.bsq', 'w', driver='ENVI', height=nx, width=mx, count=k, crs=self.crs, dtype=ortho_datacube[0].dtype, transform=transform , nodata=nodata) as dst:
            for i, band_data in enumerate(ortho_datacube, start=1):
                dst.write(band_data, i)


        # Make some simple modifications
        data_file_path = datacube_path + '.bsq'



        # Include meta data regarding the unit
        unit_str = self.config['General']['radiometric_unit']
        header_file_path = datacube_path + '.hdr'
        header = envi.open(header_file_path)
        # Set the unit of the signal in the header
        header.metadata['unit'] = unit_str
        # Set wavelengths array in the header
        header.bands.centers = wavelengths
        # TODO: include support for the bandwidths
        # header.bands.bandwidths = wl
        envi.save_image(header_file_path, header, metadata={}, interleave='bsq', filename=data_file_path, force=True)


        os.remove(datacube_path + '.img')


    def compare_hsi_composite_with_rgb_mosaic(self):
        self.rgb_ortho_path = self.config['Georeferencing']['rgbOrthoPath']
        self.hsi_composite = self.config['Georeferencing']['rgbCompositePaths'] + self.name + '.tif'
        self.rgb_ortho_reshaped = self.config['Georeferencing']['rgbOrthoReshaped'] + self.name + '.tif'
        self.dem_path = self.config['Georeferencing']['demPath']
        self.dem_reshaped = self.config['Georeferencing']['demReshaped'] + self.name + '_dem.tif'


        self.resample_rgb_ortho_to_hsi_ortho()

        self.resample_dem_to_hsi_ortho()


        raster_rgb = gdal.Open(self.rgb_ortho_reshaped, gdal.GA_Update)
        xoff1, a1, b1, yoff1, d1, e1 = raster_rgb.GetGeoTransform()  # This should be equal
        raster_rgb_array = np.array(raster_rgb.ReadAsArray())
        R = raster_rgb_array[0, :, :].reshape((raster_rgb_array.shape[1], raster_rgb_array.shape[2], 1))
        G = raster_rgb_array[1, :, :].reshape((raster_rgb_array.shape[1], raster_rgb_array.shape[2], 1))
        B = raster_rgb_array[2, :, :].reshape((raster_rgb_array.shape[1], raster_rgb_array.shape[2], 1))
        # del raster_array1
        ortho_rgb = np.concatenate((R, G, B), axis=2)
        rgb_image = Imcol(ortho_rgb)

        raster_hsi = gdal.Open(self.hsi_composite)
        raster_hsi_array = np.array(raster_hsi.ReadAsArray())
        xoff2, a2, b2, yoff2, d2, e2 = raster_hsi.GetGeoTransform()
        self.transform_pixel_projected = raster_hsi.GetGeoTransform()
        R = raster_hsi_array[0, :, :].reshape((raster_hsi_array.shape[1], raster_hsi_array.shape[2], 1))
        G = raster_hsi_array[1, :, :].reshape((raster_hsi_array.shape[1], raster_hsi_array.shape[2], 1))
        B = raster_hsi_array[2, :, :].reshape((raster_hsi_array.shape[1], raster_hsi_array.shape[2], 1))

        ortho_hsi = np.concatenate((R, G, B), axis=2)
        #ortho_hsi = (ortho_hsi - ortho_hsi.min()) / (ortho_hsi.max() - ortho_hsi.min())

        max_val = np.percentile(ortho_hsi.reshape(-1), 99)
        ortho_hsi /= max_val
        ortho_hsi[ortho_hsi > 1] = 1
        ortho_hsi = (ortho_hsi * 255).astype(np.uint8)
        ortho_hsi[ortho_hsi == 0] = 255
        hsi_image = Imcol(ortho_hsi)


        # Dem
        self.raster_dem = rasterio.open(self.dem_reshaped)


        # Adjust clahe
        hsi_image.clahe_adjustment()
        rgb_image.clahe_adjustment()
        #ortho1_clahe[ortho1
        # [..., 0] == 0] = 0

        hsi_image.to_luma(gamma=False, image_array = hsi_image.clahe_adjusted)
        rgb_image.to_luma(gamma=False, image_array= rgb_image.clahe_adjusted)

        self.compute_sift_difference(hsi_image.luma_array, rgb_image.luma_array)



    def resample_rgb_ortho_to_hsi_ortho(self):
        """Reproject RGB orthophoto to match the shape and projection of HSI raster.

        Parameters
        ----------
        infile : (string) path to input file to reproject
        match : (string) path to raster with desired shape and projection
        outfile : (string) path to output file tif
        """

        infile = self.rgb_ortho_path
        match = self.hsi_composite
        outfile = self.rgb_ortho_reshaped
        # open input
        with rasterio.open(infile) as src:
            src_transform = src.transform

            # open input to match
            with rasterio.open(match) as match:
                dst_crs = match.crs

                # calculate the output transform matrix
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs,  # input CRS
                    dst_crs,  # output CRS
                    match.width,  # input width
                    match.height,  # input height
                    *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                )

            # set properties for output
            dst_kwargs = src.meta.copy()
            dst_kwargs.update({"crs": dst_crs,
                               "transform": dst_transform,
                               "width": dst_width,
                               "height": dst_height,
                               "nodata": 0})
            #print("Coregistered to shape:", dst_height, dst_width, '\n Affine', dst_transform)
            # open output
            with rasterio.open(outfile, "w", **dst_kwargs) as dst:
                # iterate through bands and write using reproject function
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.cubic)

    def resample_dem_to_hsi_ortho(self):
        """Reproject a file to match the shape and projection of existing raster.

        Parameters
        ----------
        infile : (string) path to input file to reproject
        match : (string) path to raster with desired shape and projection
        outfile : (string) path to output file tif
        """

        infile = self.dem_path
        match = self.hsi_composite
        outfile = self.dem_reshaped
        # open input
        with rasterio.open(infile) as src:
            src_transform = src.transform

            # open input to match
            with rasterio.open(match) as match:
                dst_crs = match.crs

                # calculate the output transform matrix
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs,  # input CRS
                    dst_crs,  # output CRS
                    match.width,  # input width
                    match.height,  # input height
                    *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                )

            # set properties for output
            dst_kwargs = src.meta.copy()
            dst_kwargs.update({"crs": dst_crs,
                               "transform": dst_transform,
                               "width": dst_width,
                               "height": dst_height,
                               "nodata": 0})
            #print("Coregistered to shape:", dst_height, dst_width, '\n Affine', dst_transform)
            # open output
            with rasterio.open(outfile, "w", **dst_kwargs) as dst:
                # iterate through bands and write using reproject function
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.cubic)


    def compute_sift_difference(self, gray1, gray2):
        gray1 = (gray1 - np.min(gray1)) / (np.max(gray1) - np.min(gray1))
        gray2 = (gray2 - np.min(gray2)) / (np.max(gray2) - np.min(gray2))

        gray1 = (gray1 * 255).astype(np.uint8)
        gray2 = (gray2 * 255).astype(np.uint8)


        # Find the keypoints and descriptors with SIFT
        sift = cv.SIFT_create()
        kp2, des2 = sift.detectAndCompute(gray2, None)
        print('Key points found')
        kp1, des1 = sift.detectAndCompute(gray1, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test. We changed 0.8 to 0.85 to get more matches
        good = []
        for m, n in matches:
            if m.distance < 0.80 * n.distance:
                good.append(m)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           flags=2)

        diff_u = np.zeros(len(good))
        diff_v = np.zeros(len(good))
        uv_vec_hsi = np.zeros((len(good), 2))
        uv_vec_rgb = np.zeros((len(good), 2))
        for i in range(len(good)):
            idx2 = good[i].trainIdx
            idx1 = good[i].queryIdx
            uv1 = kp1[idx1].pt  # Slit image
            uv2 = kp2[idx2].pt  # Orthomosaic
            uv_vec_hsi[i,:] = uv1
            uv_vec_rgb[i,:] = uv2

            ## Conversion to global coordinates
            diff_u[i] = uv2[0] - uv1[0]
            diff_v[i] = uv2[1] - uv1[1]

        img3 = cv.drawMatches(gray1, kp1, gray2, kp2, good, None, **draw_params)
        plt.imshow(img3, 'gray')
        plt.show()

        #print(len(good))

        med_u = np.median(diff_u[np.abs(diff_u) < 10])
        med_v = np.median(diff_v[np.abs(diff_u) < 10])
#
        #print(np.mean(np.abs(diff_u[np.abs(diff_u) < 100])))
        #print(np.mean(np.abs(diff_v[np.abs(diff_u) < 100])))
##
        #print(np.median(np.abs(diff_u[np.abs(diff_u) < 100]  - med_u)))
        #print(np.median(np.abs(diff_v[np.abs(diff_u) < 100] - med_v)))
##
        MAD_u = np.median(np.abs(diff_u[np.abs(diff_u) < 100]  - med_u))
        MAD_v = np.median(np.abs(diff_v[np.abs(diff_u) < 100] - med_v))
##
        #MAD_tot = np.median(np.sqrt((diff_v[np.abs(diff_u) < 100] - med_v)**2 + (diff_u[np.abs(diff_u) < 100] - med_u)**2))
        # IF the disagreement is more than 100 pixels, omit it
        diff = np.sqrt(diff_u ** 2 + diff_v ** 2)

        MAE_tot = np.median(diff[diff < 5])
        print(len(good))
        print(MAE_tot)
        self.feature_uv_hsi = uv_vec_hsi[diff < 5, :]
        self.feature_uv_rgb = uv_vec_rgb[diff < 5, :]
        print(len(self.feature_uv_rgb))
#
        #print(med_u)
        #print(med_v)
        #print(MAD_u)
        #print(MAD_v)

        #plt.imshow(gray1)





        #plt.scatter(uv_vec[:,0][np.abs(diff_u) < 100], uv_vec[:,1][np.abs(diff_u) < 100], c = diff_u[np.abs(diff_u) < 100] - np.median(diff_u[np.abs(diff_u) < 100]))
        #plt.colorbar()
        #plt.show()


        #plt.hist(diff_u[np.abs(diff) < 100], 50)
        #plt.title('MAD u: ' + str(np.round(MAD_u,2)))
        #plt.xlim([-100, 100])
        #plt.show()

        plt.hist(diff[diff < 10]*0.002, 50)
        #plt.title('MAD v: ' + str(np.round(MAD_v, 2)))
        #plt.xlim([-100, 100])
        plt.show()
        #
        self.diff = diff



    def map_pixels_back_to_datacube(self, w_datacube):
        """The projected formats can be transformed back with four integer transforms and interpolated accordingly"""
        """As a simple strategy we perform bilinear interpolation"""

        indexes_grid = np.flip(self.indexes.reshape((self.height_rectified, self.width_rectified)), axis = 0)

        v = self.feature_uv_hsi[:, 0]
        u = self.feature_uv_hsi[:, 1]

        v_rgb = self.feature_uv_rgb[:, 0]
        u_rgb = self.feature_uv_rgb[:, 1]

        # Should transform rgb coordinates directly to world coordinates
        ## Conversion to global coordinates
        x = self.feature_uv_rgb[:, 0]
        y = self.feature_uv_rgb[:, 1]

        xoff, a, b, yoff, d, e = self.transform_pixel_projected


        xp = a * x + b * y + xoff + 0.5*a # The origin of the image coordinate system is located at 0.5,0.5
        yp = d * x + e * y + yoff + 0.5*e
        zp = np.zeros(yp.shape)
        for i in range(xp.shape[0]):
            temp = [x for x in self.raster_dem.sample([(xp[i], yp[i])])]
            zp[i] = float(temp[0])

        if self.is_global != True:
            self.features_points = np.concatenate((xp.reshape((-1,1)), yp.reshape((-1,1)), zp.reshape((-1,1))), axis = 1)
        else:
            geocsc = CRS.from_epsg(self.epsg_geocsc)
            proj = CRS.from_epsg(self.epsg_proj)
            transformer = Transformer.from_crs(proj, geocsc)
            self.features_points = np.zeros((xp.shape[0], 3))



            (xECEF, yECEF, zECEF) = transformer.transform(xx=xp, yy=yp, zz=zp)

            self.features_points[:, 0] = xECEF - self.offX
            self.features_points[:, 1] = yECEF - self.offY
            self.features_points[:, 2] = zECEF - self.offZ




        #

        self.v_datacube_hsi = np.zeros((v.shape[0], 4))
        self.u_datacube_hsi = np.zeros((v.shape[0], 4))

        # See wikipedia
        for i in range(4):
            if i == 0:
                u1 = np.floor(u).astype(np.int32)  # row
                v1 = np.floor(v).astype(np.int32)  # col
            elif i == 1:
                u1 = np.floor(u).astype(np.int32)  # row
                v1 = np.ceil(v).astype(np.int32)  # col
            elif i == 2:
                u1 = np.ceil(u).astype(np.int32)  # row
                v1 = np.floor(v).astype(np.int32)  # col
            else:
                u1 = np.ceil(u).astype(np.int32)  # row
                v1 = np.ceil(v).astype(np.int32)  # col

            ind_datacube_hsi = indexes_grid[u1, v1] # 1D Indexer til rå datakube

            self.v_datacube_hsi[:, i] = ind_datacube_hsi % w_datacube
            self.u_datacube_hsi[:, i]  = (ind_datacube_hsi - self.v_datacube_hsi[:, i]) / w_datacube


        self.x1_x_hsi = v - np.floor(v)
        self.y1_y_hsi = u - np.floor(u)

        #self.x1_x_rgb = v_rgb - np.floor(v_rgb)
        #self.y1_y_rgb = u_rgb - np.floor(u_rgb)










