import os
from concurrent.futures import ThreadPoolExecutor

# Third party
import cv2 as cv
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from pyproj import CRS, Transformer
import pyvista as pv
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from osgeo import gdal, osr
from shapely.geometry import Polygon, mapping, MultiPoint
from sklearn.neighbors import NearestNeighbors
from spectral import envi
import spectral as sp
import spectral.io.envi
import h5py


# Local module
from scripts.colours import Image as Imcol
from lib.parsing_utils import Hyperspectral

GRAVITY = 9.81 # m/s^2
# ENVI datatype conversion dictionary
dtype_dict = {1:np.uint8,
             2:np.int16,
             3:np.int32,
             4:np.float32,
             5:np.float64,
             12:np.uint16,
             13:np.uint32,
             14:np.int64,
             15:np.uint64}

class GeoSpatialAbstractionHSI():
    def __init__(self, point_cloud, transect_string, config_crs):
        self.name = transect_string
        self.points_geocsc = point_cloud

        self.offX = config_crs.off_x
        self.offY = config_crs.off_y
        self.offZ = config_crs.off_z

        self.epsg_geocsc = config_crs.epsg_geocsc
        
        self.n_lines = point_cloud.shape[0]
        self.n_pixels = point_cloud.shape[1]
        self.n_bands = point_cloud.shape[1]

        # A clean way of doing things would be to define 
    def transform_geocentric_to_projected(self, config_crs):


        
        self.epsg_proj = config_crs.epsg_proj
        self.crs = 'EPSG:' + str(self.epsg_proj)
        
        

        self.points_proj  = np.zeros(self.points_geocsc.shape) # Remains same if it is local
        
        geocsc = CRS.from_epsg(self.epsg_geocsc)
        proj = CRS.from_epsg(self.epsg_proj)
        transformer = Transformer.from_crs(geocsc, proj)

        xECEF = self.points_geocsc[:,:,0].reshape((-1, 1))
        yECEF = self.points_geocsc[:, :, 1].reshape((-1, 1))
        zECEF = self.points_geocsc[:, :, 2].reshape((-1, 1))

        

        (east, north, hei) = transformer.transform(xx=xECEF + self.offX, yy=yECEF + self.offY, zz=zECEF + self.offZ)

        self.points_proj[:,:,0] = east.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))
        self.points_proj[:, :, 1] = north.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))
        self.points_proj[:, :, 2] = hei.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))

        


    def footprint_to_shape_file(self, footprint_dir):
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

        gdf = gpd.GeoDataFrame(geometry=[self.footprint_shp], crs=self.crs)

        shape_path = footprint_dir + self.name + '.shp'

        gdf.to_file(shape_path, driver='ESRI Shapefile')

    def resample_datacube(self, radiance_cube, wavelengths, fwhm, envi_cube_dir, rgb_composite_dir, config_ortho):
        """Resamples the radiance cube into a geographic grid based on the georeferencing

        :param radiance_cube: The data cube of radiance with the corresponding radiometric_unit of the data
        :type radiance_cube: Often an ndarray(n, m, k) where n-number of lines, m- number of pixels, and k-number of spectral bands
        :param wavelengths: The band's centre wavelengths
        :type wavelengths: ndarray(k, 1)
        :param fwhm: Full Width Half Maximum, descibing the band's widths (often in nanometers)
        :type fwhm: ndarray(k, 1)
        :param envi_cube_dir: Directory to write data cubes
        :type envi_cube_dir: string, path
        :param rgb_composite_dir: Directory to write data cubes
        :type rgb_composite_dir: string, path
        :param config_ortho: The relevant configurations for orthorectification
        :type config_ortho: Dictionary
        """


        
        n_bands = len(wavelengths)
        #
        n = radiance_cube.shape[0]
        m = radiance_cube.shape[1]
        k = radiance_cube.shape[2] # Number of bands

        bytes_per_entry = radiance_cube.itemsize
        chunk_size_GB = config_ortho.chunk_size_cube_GB


        # If chunking is to be applied, we can use square chunks. Round to the nearest thousand because of personal OCD
        chunk_square_length = np.sqrt((chunk_size_GB*1024**3) / (k*bytes_per_entry))
        self.chunk_square_length = int(np.round(chunk_square_length/1000)*1000)

        if self.chunk_square_length == 0:
            self.chunk_square_length = 1000

        self.chunk_area = self.chunk_square_length**2

        self.res = config_ortho.ground_resolution

        wl_red = config_ortho.wl_red
        wl_green = config_ortho.wl_green
        wl_blue = config_ortho.wl_blue

        raster_transform_method = config_ortho.raster_transform_method

        # Set nodata value for ortho-products
        nodata = config_ortho.nodata_value
        self.nodata = nodata

        rgb_composite_only = config_ortho.resample_rgb_only

        

        # Set custom RGB bands from *.ini file
        band_ind_R = np.argmin(np.abs(wl_red - wavelengths))
        band_ind_G = np.argmin(np.abs(wl_green - wavelengths))
        band_ind_B = np.argmin(np.abs(wl_blue - wavelengths))

        # To let ENVI pick up on which bands are used for red-green-blue vizualization
        self.default_bands_string = '{ '+' , '.join([str(band_ind_R), str(band_ind_G), str(band_ind_B)]) + ' }'

        # Some relevant metadata
        metadata_ENVI = {
            'description': 'Radiance converted, georeferenced data',
            'unit': config_ortho.radiometric_unit,
            'wavelength units': config_ortho.wavelength_unit,
            'sensor type': config_ortho.sensor_type,
            'default bands': self.default_bands_string,
            'interleave': config_ortho.interleave,
            'wavelengths': wavelengths
        }
        try:
            # If vector form
            if fwhm.any() == np.nan:
                pass
            else:
                metadata_ENVI['fwhm'] = fwhm
        except AttributeError:
            # If scalar
            if fwhm == np.nan:
                pass
            else:
                metadata_ENVI['fwhm'] = fwhm

            
        
        
        # Extract relevant info from hyp object
        datacube = radiance_cube[:, :, :].reshape((-1, n_bands))
        rgb_cube = datacube[:, [band_ind_R, band_ind_G, band_ind_B]].reshape((-1, 3))

        # Horizontal coordinates of intersections in projected CRS
        coords = self.points_proj[:, :, 0:2].reshape((-1, 2))

        del radiance_cube
        
        # The raster can be rotated optimally (which saves loads of memory) for transects that are long compared to width. 
        # However, north-east oriented rasters is more supported by image visualization
        transform, height, width, indexes, suffix = GeoSpatialAbstractionHSI.cube_to_raster_grid(coords, raster_transform_method, resolution = self.res)

        # Make accessible as attribute because it can be to write ancillary data
        self.indexes = indexes.copy()
        self.transform = transform
        self.width = width
        self.height = height
        self.suffix = suffix

        # Create raster mask from the polygon describing the footprint
        geoms = [mapping(self.footprint_shp)]
        mask = geometry_mask(geoms, out_shape=(height, width), transform=transform)

        self.mask = mask

        # Build datacube
        if not rgb_composite_only:

            # For the later processing, storing the mapping from the rectified grid to the raw datacube makes sense:
            indexes_grid_unmasked = indexes.reshape((height, width))
            # Mask indexes
            indexes_grid_unmasked[mask == 1] = nodata

            # Make masked indices accessible as these allow orthorectification of ancilliary data
            self.index_grid_masked = indexes_grid_unmasked

            # Step 1: Initialize a Memory Map
            filename = 'memmap_array.dat'
            shape = (n_bands, height, width)
            dtype = np.float32
            
            memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

            # In grid form, identify whether partitioning is needed
            area_idx_grid = width*height

            if area_idx_grid > self.chunk_area:
                
                # Subdivide along horizontal axis:
                delta_width = int(self.chunk_area/height)
                # Number of slices
                num_delta_widths = int(width/delta_width) + 1

                for i in range(num_delta_widths):
                    
                    if i != num_delta_widths-1:
                        sub_indices = self.index_grid_masked[:, i*delta_width:(i+1)*delta_width].reshape((-1,1))
                        dw = delta_width
                        
                    else:
                        sub_indices = self.index_grid_masked[:, i*delta_width:width].reshape((-1,1))
                        dw = np.arange(i*delta_width, width).size
                    
                    # Only extraxt valid data:
                    sub_ortho_cube = np.ones((sub_indices.shape[0]*sub_indices.shape[1], n_bands))*nodata
                    
                    # From the grid, the only valid pixels are
                    valid_pixels = (sub_indices != nodata).reshape(-1)
                    valid_pixels_raw = sub_indices[valid_pixels].reshape(-1)
                    valid_pixels_geo = np.arange(sub_indices.size)[valid_pixels]

                    sub_ortho_cube[valid_pixels_geo, :] = datacube[valid_pixels_raw, :]

                    sub_ortho_cube = sub_ortho_cube.reshape((height, dw, n_bands))

                    # Form to Rasterio friendly
                    sub_ortho_cube = np.transpose(sub_ortho_cube, axes=[2, 0, 1])

                    memmap_array[:, :, i*delta_width:i*delta_width + dw] = sub_ortho_cube
                # Free memory
                del datacube
                del sub_ortho_cube
            else:
                # Resample
                ortho_datacube = datacube[self.indexes, :]

                # Free memory
                del datacube

                # Reshape to datacube form
                ortho_datacube = ortho_datacube.reshape((height, width, n_bands))

                # Mask
                ortho_datacube[mask == 1, :] = nodata

                # Form to Rasterio friendly
                ortho_datacube = np.transpose(ortho_datacube, axes=[2, 0, 1])

                # Write to memory map
                memmap_array[:] = ortho_datacube

                # Free memory
                del ortho_datacube
                

                
            GeoSpatialAbstractionHSI.write_datacube_ENVI(memmap_array, 
                                nodata, 
                                transform, 
                                datacube_path = envi_cube_dir + self.name + suffix, 
                                wavelengths=wavelengths,
                                fwhm=fwhm,
                                metadata = metadata_ENVI,
                                crs=self.crs,
                                interleave = config_ortho.interleave)

            

        # Resample RGB image data 
        ortho_rgb = rgb_cube[self.indexes, :].flatten()

        # Reshape image
        ortho_rgb = ortho_rgb.reshape((height, width, 3))
    
        # Mask
        ortho_rgb[mask == 1, :] = nodata

        # Arange datacube or composite in rasterio-friendly structure
        ortho_rgb = np.transpose(ortho_rgb, axes = [2, 0, 1])
        
        # Write pseudo-RGB composite to composite folder ../GIS/RGBComposites
        with rasterio.open(rgb_composite_dir + self.name + suffix + '.tif', 'w', driver='GTiff',
                                height=height, width=width, count=3, dtype=ortho_rgb.dtype,
                                crs=self.crs, transform=transform, nodata=nodata) as dst:

            dst.write(ortho_rgb)
        

    def resample_ancillary(self, h5_filename, anc_dir, anc_dict, interleave = 'bil'):

        band_counter = 0
        band_names = []

        # Define one hyperspectral data to 
        with h5py.File(h5_filename, 'r', libver='latest') as f:
            for attribute_name, h5_hierarchy_item_path in anc_dict.items():
                if attribute_name != 'folder':
                    data = f[h5_hierarchy_item_path][()]


                    if data.ndim == 2:
                        if data.shape[1]  != self.n_pixels:
                            # For data of shape n_lines x j, for example camera positions, reshape to n_lines x n_pixels x j
                            j = data.shape[1]
                            data = np.einsum('ijk, ik -> ijk', np.ones((self.n_lines, self.n_pixels, j), dtype=np.float32), data)
                        else:
                            # For data with dimension n_lines x n_pixels, add third dimension
                            data = data.reshape((data.shape[0], data.shape[1], 1))

                    data = data.astype(dtype = np.float32)
                    
                    # For first layer
                    if band_counter == 0:
                        anc_data_array = data
                    # For other layers, concatenate with existing layers
                    else:
                        anc_data_array = np.concatenate((anc_data_array, data), axis = 2)

                    # Necessary to modify for data with multiple bands
                    k = data.shape[2]
                    # If data has more than one band:
                    if k > 1:
                        for i in range(k):
                            band_name_data = attribute_name + '_' + str(i)
                            band_names.append(band_name_data)
                    else:
                        band_name_data = attribute_name
                        band_names.append(band_name_data)
                    
                    band_counter += k
                    
                    
            
        
        
        metadata_anc = {
            'description': 'Ancillary data',
            'band names': '{ '+' , '.join(band_names) + ' }'
        }
        n_bands = band_counter

        # Resample
        anc_data_array = anc_data_array.reshape((-1, n_bands))
        ortho_anc = anc_data_array[self.indexes.flatten(), :]

        # Free memory
        del anc_data_array

        # Reshape to anc_cube form
        ortho_anc = ortho_anc.reshape((self.height, self.width, n_bands))

        # Mask
        ortho_anc[self.mask == 1, :] = self.nodata

        # Form to Rasterio friendly
        ortho_anc = np.transpose(ortho_anc, axes=[2, 0, 1])

        if not os.path.isdir(anc_dir):
            os.mkdir(anc_dir)

        GeoSpatialAbstractionHSI.write_ancillary_ENVI(ortho_anc, 
                                                      nodata = self.nodata, 
                                                      transform = self.transform, 
                                                      crs = self.crs,
                                                      anc_path = anc_dir + self.name + self.suffix,
                                                      metadata=metadata_anc,
                                                      interleave=interleave)

        

    @staticmethod
    def cube_to_raster_grid(coords, raster_transform_method, resolution):
        if raster_transform_method == 'north_east':
            # Creates the minimal area (and thus memory) rectangle around chunk
            polygon = MultiPoint(coords).envelope

            # extract coordinates
            x, y = polygon.exterior.xy

            idx_ul = 3
            idx_ur = 2
            idx_ll = 4

            suffix = '_north_east'
            

            
        elif raster_transform_method == 'minimal_rectangle':
            
            # Creates the minimal area (and thus memory) rectangle around chunk
            polygon = MultiPoint(coords).minimum_rotated_rectangle

            # extract coordinates
            x, y = polygon.exterior.xy

            # Increasing indices are against the clock
            
            # Determine basis vectors from data
            idx_ul = 3
            idx_ur = 2
            idx_ll = 4

            suffix = '_rotated'

        x_ul = x[idx_ul]
        y_ul = y[idx_ul]

        x_ur = x[idx_ur]
        y_ur = y[idx_ur]

        x_ll = x[idx_ll]
        y_ll = y[idx_ll]

        # The vector from upper-left corner aka origin to the upper-right equals lambda*e_basis_x
        e_basis_x = np.array([x_ur-x_ul, y_ur-y_ul]).reshape((2,1))
        w_transect = np.linalg.norm(e_basis_x)
        e_basis_x /= w_transect

        # The y basis vector is the vector to the other edge
        e_basis_y = np.array([x_ll-x_ul, y_ll-y_ul]).reshape((2,1))
        h_transect = np.linalg.norm(e_basis_y)
        e_basis_y /= h_transect
        e_basis_y *= -1 # Ensuring Right handedness (image storage y direction is o)

        R = np.hstack((e_basis_x, e_basis_y)) # 2D rotation matrix by definition
        
        # Define origin/translation vector
        o = np.array([x[idx_ul], y[idx_ul]]).reshape((2))

        # Transformation matrix rigid body 2D
        Trb = np.zeros((3,3))
        Trb[0:2,0:2] = R
        Trb[0:2,2] = o
        Trb[2,2] = 1

        s_x = resolution
        s_y = resolution

        S = np.diag(np.array([s_x, s_y, 1]))

        # Reflection to account for opposite row/up direction
        Ref = np.diag(np.array([1, -1, 1]))

        # The affine transform is then expressed:
        Taff = Trb.dot(S).dot(Ref)

        # The metric width and length of the transect can give us the number of pixels
        width = int(np.ceil(w_transect/s_x))
        height = int(np.ceil(h_transect/s_y))

        # Rasterio operates with a conventional affine
        a, b, c, d, e, f = Taff[0,0], Taff[0,1], Taff[0,2], Taff[1,0], Taff[1,1], Taff[1,2]

        transform = rasterio.Affine(a, b, c, d, e, f)

        # Pixel centers reside at half coordinates.
        xi, yi = np.meshgrid(np.arange(width) + 0.5, 
                                np.arange(height) + 0.5)
        zi = np.ones(xi.shape)

        # Form homogeneous vectors
        x_r = np.vstack((xi.flatten(), yi.flatten(), zi.flatten())).T

        x_p = np.matmul(Taff, x_r.T).T # Map pixels to geographic

        xy = np.vstack((x_p[:,0].flatten(), x_p[:,1].flatten())).T
        

        # Locate nearest neighbors in a radius defined by the resolution
        tree = NearestNeighbors(radius=resolution).fit(coords)

        # Calculate the nearest neighbors. Here we only use one neighbor, but other approaches could be employed
        dist, indexes = tree.kneighbors(xy, 1)
        
        return transform, height, width, indexes, suffix

    @staticmethod
    def write_datacube_ENVI(ortho_datacube, nodata, transform, datacube_path, wavelengths, fwhm, metadata, interleave, crs):
        """_summary_

        :param ortho_datacube: _description_
        :type ortho_datacube: _type_
        :param nodata: _description_
        :type nodata: _type_
        :param transform: _description_
        :type transform: _type_
        :param datacube_path: _description_
        :type datacube_path: _type_
        :param wavelengths: _description_
        :type wavelengths: _type_
        :param fwhm: _description_
        :type fwhm: _type_
        :param envi_hdr_dict: _description_
        :type envi_hdr_dict: _type_
        """

        nx = ortho_datacube.shape[1]
        mx = ortho_datacube.shape[2]
        k = ortho_datacube.shape[0]

        # Make some simple modifications
        data_file_path = datacube_path + '.' + interleave
        header_file_path = datacube_path + '.hdr'

        # Clean the files generated by rasterio
        def write_band(args):
            band_data, index, dst = args
            dst.write(band_data, index + 1)
        
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
            os.remove(header_file_path)

        
        # Assuming ortho_datacube is a 3D NumPy array with shape (k, nx, mx)
        with rasterio.open(data_file_path, 'w', driver='ENVI', height=nx, width=mx, count=k, crs=crs, dtype=ortho_datacube.dtype, transform=transform, nodata=nodata) as dst:
            # Create a ThreadPoolExecutor with as many threads as bands
            with ThreadPoolExecutor(max_workers=k) as executor:
                # Use executor.map to parallelize the band writing process
                executor.map(write_band, [(band_data, i, dst) for i, band_data in enumerate(ortho_datacube)])
            
        
        header = sp.io.envi.read_envi_header(datacube_path + '.hdr') # Open for extraction

        
        header.pop('band names')

        for meta_key, value in metadata.items():
            header[meta_key] = value

        sp.io.envi.write_envi_header(fileName=header_file_path, header_dict=header)

    @staticmethod
    def write_ancillary_ENVI(anc_data, nodata, transform, anc_path, metadata, interleave, crs):
        """_summary_

        :param anc_data: An ancilliary data cube 
        :type anc_data: j x m x n where j are the number of bands, and m, n are the raster dimensions 
        :param nodata: _description_
        :type nodata: _type_
        :param transform: _description_
        :type transform: _type_
        :param datacube_path: _description_
        :type datacube_path: _type_
        :param metadata: _description_
        :type metadata: _type_
        :param interleave: _description_
        :type interleave: _type_
        :param crs: _description_
        :type crs: _type_
        """
        nx = anc_data.shape[1]
        mx = anc_data.shape[2]
        k = anc_data.shape[0]

        # Make some simple modifications
        data_file_path = anc_path + '.' + interleave
        header_file_path = anc_path + '.hdr'

        if os.path.exists(data_file_path):
            os.remove(data_file_path)
            os.remove(header_file_path)

        # Hack to exploit rasterio's generation of map info

        # Clean the files generated by rasterio
        def write_band(args):
            band_data, index, dst = args
            dst.write(band_data, index + 1)

        # Assuming ortho_datacube is a 3D NumPy array with shape (k, nx, mx)
        with rasterio.open(data_file_path, 'w', driver='ENVI', height=nx, width=mx, count=k, crs=crs, dtype=anc_data.dtype, transform=transform, nodata=nodata) as dst:
            # Create a ThreadPoolExecutor with as many threads as bands
            with ThreadPoolExecutor(max_workers=k) as executor:
                # Use executor.map to parallelize the band writing process
                executor.map(write_band, [(band_data, i, dst) for i, band_data in enumerate(anc_data)])
            
        
        header = sp.io.envi.read_envi_header(anc_path + '.hdr') # Open for modif
        #header['interleave'] = interleave
        header.pop('band names') # Remove defaults

        for meta_key, value in metadata.items():
            header[meta_key] = value

        sp.io.envi.write_envi_header(fileName=header_file_path, header_dict=header)

    def compare_hsi_composite_with_rgb_mosaic(self):
        self.rgb_ortho_path = self.config['Absolute Paths']['rgbOrthoPath']
        self.hsi_composite = self.config['Absolute Paths']['rgbCompositePaths'] + self.name + '.tif'
        self.rgb_ortho_reshaped = self.config['Absolute Paths']['rgbOrthoReshaped'] + self.name + '.tif'
        self.dem_path = self.config['Absolute Paths']['demPath']
        self.dem_reshaped = self.config['Absolute Paths']['demReshaped'] + self.name + '_dem.tif'


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

        max_val = np.percentile(ortho_hsi.reshape(-1), 99)
        ortho_hsi /= max_val
        ortho_hsi[ortho_hsi > 1] = 1
        ortho_hsi = (ortho_hsi * 255).astype(np.uint8)
        ortho_hsi[ortho_hsi == 0] = 255
        hsi_image = Imcol(ortho_hsi)


        # Dem
        self.raster_dem = rasterio.open(self.dem_reshaped)


        # Adjust Clahe
        hsi_image.clahe_adjustment()
        rgb_image.clahe_adjustment()

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









