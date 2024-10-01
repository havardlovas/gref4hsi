# Python standard lib
import os
from concurrent.futures import ThreadPoolExecutor
import sys

# Third party
import cv2 as cv
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer
import pyproj
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from osgeo import gdal, osr
from shapely.geometry import Polygon, mapping, MultiPoint
from sklearn.neighbors import NearestNeighbors
from spectral import envi
import spectral as sp
import h5py
from scipy.spatial.transform import Rotation as RotLib

# Lib modules
from gref4hsi.utils.colours import Image as Imcol

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

        x_ecef = self.points_geocsc[:,:,0].reshape((-1, 1))
        y_ecef = self.points_geocsc[:, :, 1].reshape((-1, 1))
        z_ecef = self.points_geocsc[:, :, 2].reshape((-1, 1))

        

        (east, north, hei) = transformer.transform(xx=x_ecef, yy=y_ecef, zz=z_ecef)

        self.points_proj[:,:,0] = east.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))
        self.points_proj[:, :, 1] = north.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))
        self.points_proj[:, :, 2] = hei.reshape((self.points_proj.shape[0], self.points_proj.shape[1]))


        


    def footprint_to_shape_file(self, footprint_dir):
        """Uses the georeferenced data points (in projected form) to describe the footprint as a shape file

        :param footprint_dir: Where to put the shape files describing the footprint
        :type footprint_dir: string
        """
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

        # The swiped ground area is defined by the convex hull
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

        # If chunking is to be applied, we can use square chunks
        chunk_square_length = np.sqrt((chunk_size_GB*1024**3) / (k*bytes_per_entry))
        self.chunk_square_length = int(np.round(chunk_square_length/1000)*1000)

        if self.chunk_square_length == 0:
            self.chunk_square_length = 1000

        self.chunk_area = self.chunk_square_length**2

        self.res = config_ortho.ground_resolution

        wl_red = config_ortho.wl_red
        wl_green = config_ortho.wl_green
        wl_blue = config_ortho.wl_blue
        
        # North-east or memory optimal
        raster_transform_method = config_ortho.raster_transform_method

        # Set nodata value for ortho-products
        
        self.nodata = config_ortho.nodata_value
        
        # If dtype is integer and not same type of int as radiance cube
        # This avoids annoying error
        if np.issubdtype(radiance_cube.dtype, np.integer):
            if self.nodata.dtype != radiance_cube.dtype:
                # If they are incompatible use the max value to fix the problem
                self.nodata = _get_max_value(radiance_cube.dtype)
                
                # To avoid calling nodata on saturated values we do
                radiance_cube[radiance_cube == self.nodata] = self.nodata - 1

        
        #
        rgb_composite_only = config_ortho.resample_rgb_only

        

        # Set custom RGB bands from *.ini file
        band_ind_R = np.argmin(np.abs(wl_red - wavelengths))
        band_ind_G = np.argmin(np.abs(wl_green - wavelengths))
        band_ind_B = np.argmin(np.abs(wl_blue - wavelengths))

        # To let ENVI pick up on which bands are used for red-green-blue vizualization
        self.default_bands_string = '{ '+' , '.join([str(band_ind_R), str(band_ind_G), str(band_ind_B)]) + ' }'

        # Some relevant metadata.
        # See https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html for documentation of the entries
        metadata_ENVI = {
            'description': 'Radiance converted, georeferenced data',
            'unit': config_ortho.radiometric_unit,
            'wavelength units': config_ortho.wavelength_unit,
            'sensor type': config_ortho.sensor_type,
            'default bands': self.default_bands_string,
            'interleave': config_ortho.interleave,
            'wavelength': wavelengths
        }
        try:
            # If vector form is avai
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

        

        # Horizontal coordinates of intersections in projected CRS
        coords = self.points_proj[:, :, 0:2].reshape((-1, 2))

        del radiance_cube
        
        # The raster can be rotated optimally (which saves loads of memory) for transects that are long compared to width. 
        # However, north-east oriented rasters is more supported by image visualization
        transform, height, width, indexes, suffix, mask_nn = GeoSpatialAbstractionHSI.cube_to_raster_grid(coords, raster_transform_method, resolution = self.res)

        # Make accessible as attribute because it can be to write ancillary data
        self.indexes = indexes.copy()
        self.transform = transform
        self.width = width
        self.height = height
        self.suffix = suffix

        # Create raster mask from the polygon describing the footprint (currently not used for anything)
        
        # Recommend using the nearest neighbor if transects has turns
        # the footprint method will render a mosaic without holes, but may lead to a lot of interpolation in rugged terrain
        mask_method = config_ortho.pixel_mask_method       
        
        geoms = [mapping(self.footprint_shp)]
        mask_footprint = geometry_mask(geoms, out_shape=(height, width), transform=transform)

        if mask_method == 'nn':
            mask = mask_nn.reshape((height, width))
        elif mask_method == 'footprint':
            mask = mask_footprint.reshape((height, width))


        
        
        
        self.mask = mask

        # Build datacube
        if not rgb_composite_only:

            # For the later processing, storing the mapping from the rectified grid to the raw datacube makes sense:
            indexes_grid_unmasked = indexes.reshape((height, width))
            
            # Mask indexes
            indexes_grid_unmasked[mask == 1] = self.nodata

            # Make masked indices accessible as these allow orthorectification of ancilliary data
            self.index_grid_masked = indexes_grid_unmasked

            # To do the writing of the rasters we use a memory map, and writes chunks at a time.
            # For this writing process, we need to know all the below
            memmap_gen_params = {
                'indexes': indexes,
                'index_grid_masked': self.index_grid_masked,
                'mask': mask,
                'nodata': self.nodata,
                'height': height,
                'width': width,
                'datacube': datacube,
                'chunk_area': self.chunk_area
            }

            GeoSpatialAbstractionHSI.write_datacube_ENVI(memmap_gen_params, 
                                self.nodata, 
                                transform, 
                                datacube_path = envi_cube_dir + self.name + suffix, 
                                wavelengths=wavelengths,
                                fwhm=fwhm,
                                metadata = metadata_ENVI,
                                crs=self.crs,
                                interleave = config_ortho.interleave)

            
        
        # RBG Composite as list
        rgb_cube = datacube[:, [band_ind_R, band_ind_G, band_ind_B]].reshape((-1, 3))
        
        # Resample RGB image data 
        ortho_rgb = rgb_cube[self.indexes, :].flatten()

        # Reshape image
        ortho_rgb = ortho_rgb.reshape((height, width, 3))
    
        # Mask RGB
        ortho_rgb[mask == 1, :] = self.nodata

        # Arange composite in rasterio-friendly structure
        ortho_rgb = np.transpose(ortho_rgb, axes = [2, 0, 1])
        
        # Write pseudo-RGB composite to composite folder ../GIS/RGBComposites
        with rasterio.open(rgb_composite_dir + self.name + suffix + '.tif', 'w', driver='GTiff',
                                height=height, width=width, count=3, dtype=ortho_rgb.dtype,
                                crs=self.crs, transform=transform, nodata=self.nodata) as dst:

            dst.write(ortho_rgb)
        
    @staticmethod
    def write_datacube_memmap(memmap_array, indexes, index_grid_masked, mask, nodata, height, width, datacube, chunk_area):  
        """Function for writing data cube arrays to memory maps, effectively avoiding memory issues.

        :param memmap_array: _description_
        :type memmap_array: _type_
        :param indexes: _description_
        :type indexes: _type_
        :param index_grid_masked: _description_
        :type index_grid_masked: _type_
        :param mask: _description_
        :type mask: _type_
        :param nodata: _description_
        :type nodata: _type_
        :param height: _description_
        :type height: _type_
        :param width: _description_
        :type width: _type_
        :param datacube: _description_
        :type datacube: _type_
        :param chunk_area: _description_
        :type chunk_area: _type_
        """

        # How the hell is this logical?
        n_bands = datacube.shape[1]

        # In grid form, identify whether partitioning is needed
        area_idx_grid = width*height

        if area_idx_grid > chunk_area:
            
            # Subdivide along horizontal axis:
            delta_width = int(chunk_area/height)
            # Number of slices
            num_delta_widths = int(width/delta_width) + 1
            # Iterate slices
            for i in range(num_delta_widths):
                # Last slice has different thickness
                if i != num_delta_widths-1:
                    sub_indices = index_grid_masked[:, i*delta_width:(i+1)*delta_width].reshape((-1,1))
                    dw = delta_width
                    
                else:
                    sub_indices = index_grid_masked[:, i*delta_width:width].reshape((-1,1))
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
                #sub_ortho_cube = np.transpose(sub_ortho_cube, axes=[2, 0, 1])

                memmap_array[:, i*delta_width:i*delta_width + dw, :] = sub_ortho_cube
            # Free memory
            del datacube
            del sub_ortho_cube
        else:
            # Resample
            ortho_datacube = datacube[indexes, :]

            # Free memory
            del datacube

            # Reshape to datacube form
            ortho_datacube = ortho_datacube.reshape((height, width, n_bands))

            # Mask
            ortho_datacube[mask == 1, :] = nodata

            # Form to Rasterio friendly
            #ortho_datacube = np.transpose(ortho_datacube, axes=[2, 0, 1])

            # Write to memory map
            memmap_array[:] = ortho_datacube

            # Free memory
            del ortho_datacube

        return  

    def resample_ancillary(self, h5_filename, anc_dir, anc_dict, interleave = 'bsq'):

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
                            data = np.einsum('ijk, ik -> ijk', np.ones((self.n_lines, self.n_pixels, j), dtype=np.float64), data)
                        else:
                            # For data with dimension n_lines x n_pixels, add third dimension
                            data = data.reshape((data.shape[0], data.shape[1], 1))

                    data = data.astype(dtype = np.float64)
                    
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
        
        # Grid of indices
        indexes_grid_unmasked = self.indexes.copy().reshape((self.height, self.width))

        # Mask indexes
        indexes_grid_unmasked[self.mask == 1] = self.nodata

        # Make masked indices accessible as these allow orthorectification of ancilliary data
        self.index_grid_masked = indexes_grid_unmasked

        # Alternative 2
        memmap_gen_params = {
            'indexes': self.indexes,
            'index_grid_masked': self.index_grid_masked,
            'mask': self.mask,
            'nodata': self.nodata,
            'height': self.height,
            'width': self.width,
            'datacube': anc_data_array,
            'chunk_area': self.chunk_area
        }
        
        GeoSpatialAbstractionHSI.write_ancillary_ENVI_envi(nodata = self.nodata, 
                                                    transform = self.transform, 
                                                    crs = self.crs,
                                                    anc_path = anc_dir + self.name + self.suffix,
                                                    metadata = metadata_anc,
                                                    interleave=interleave,
                                                    memmap_gen_params=memmap_gen_params)

        

    @staticmethod
    def cube_to_raster_grid(coords, raster_transform_method, resolution):
        """Function that takes projected coordinates (e.g. UTM 32 east and north) of ray intersections and computes an image grid

        :param coords: _description_
        :type coords: _type_
        :param raster_transform_method: How the raster grid is calculated. "north_east" is standard and defines a rectangle along north/east. "minimal_rectangle" is memory optimal as it finds the smallest enclosing rectangle that wraps the points.
        :type raster_transform_method: _type_
        :param resolution: _description_
        :type resolution: _type_
        :return: _description_
        :rtype: _type_
        """


        if raster_transform_method == 'north_east':
            # Creates the minimal area (and thus memory) rectangle around points
            polygon = MultiPoint(coords).envelope

            # extract coordinates
            x, y = polygon.exterior.xy

            suffix = '_north_east'
            

            
        elif raster_transform_method == 'minimal_rectangle':
            
            # Creates the minimal area (and thus memory) rectangle around points
            polygon = MultiPoint(coords).minimum_rotated_rectangle

            # extract coordinates
            x, y = polygon.exterior.xy

            

            suffix = '_rotated'

        # Indices increase against the clock
        # ul means upper left corner of polygon, ll means lower left and so on
        idx_ul = 3
        idx_ur = 2
        idx_ll = 4

        x_ul = x[idx_ul]
        y_ul = y[idx_ul]

        x_ur = x[idx_ur]
        y_ur = y[idx_ur]

        x_ll = x[idx_ll]
        y_ll = y[idx_ll]

        # The vector from the upper-left corner aka origin to the upper-right equals lambda*e_basis_x
        e_basis_x = np.array([x_ur-x_ul, y_ur-y_ul]).reshape((2,1))
        w_transect = np.linalg.norm(e_basis_x)
        e_basis_x /= w_transect

        # The y basis vector is the vector to the other edge
        e_basis_y = np.array([x_ll-x_ul, y_ll-y_ul]).reshape((2,1))
        h_transect = np.linalg.norm(e_basis_y)
        e_basis_y /= h_transect
        e_basis_y *= -1 # Ensuring right handedness (image storage y direction is opposite)

        R = np.hstack((e_basis_x, e_basis_y)) # 2D rotation matrix by definition
        
        # Define origin/translation vector as the upper left corner
        o = np.array([x[idx_ul], y[idx_ul]]).reshape((2))

        # Transformation matrix rigid body 2D
        Trb = np.zeros((3,3))
        Trb[0:2,0:2] = R
        Trb[0:2,2] = o
        Trb[2,2] = 1

        # We operate with same resolution in X and Y. Note that for "minimal_rectangle" X, Y are NOT East and North. 
        s_x = resolution
        s_y = resolution

        S = np.diag(np.array([s_x, s_y, 1])) # Scale

        # Reflection matrix to account for opposite row/up direction
        Ref = np.diag(np.array([1, -1, 1]))

        # The affine transform is then expressed:
        Taff = Trb.dot(S).dot(Ref)

        # The metric width and length of the transect can give us the number of pixels
        width = int(np.ceil(w_transect/s_x))
        height = int(np.ceil(h_transect/s_y))

        # Rasterio operates with a conventional affine geotransform
        a, b, c, d, e, f = Taff[0,0], Taff[0,1], Taff[0,2], Taff[1,0], Taff[1,1], Taff[1,2]
        transform = rasterio.Affine(a, b, c, d, e, f)

        # Define local orthographic pixel grid. Pixel centers reside at half coordinates.
        xi, yi = np.meshgrid(np.arange(width) + 0.5, 
                                np.arange(height) + 0.5)
        # To get homogeneous vector (not an actual z coordinate)
        zi = np.ones(xi.shape)

        # Form homogeneous vectors (allows translation by multiplication)
        x_r = np.vstack((xi.flatten(), yi.flatten(), zi.flatten())).T

        # Map orthographic pixels to projected system
        x_p = np.matmul(Taff, x_r.T).T 

        # Make a point vector of grid points to be used in nearest neighbor
        xy = np.vstack((x_p[:,0].flatten(), x_p[:,1].flatten())).T
        
        # Define NN search tree from intersection points
        tree = NearestNeighbors(radius=resolution).fit(coords)

        # Calculate the nearest intersection point (in "coords") for each grid cell ("xy").
        
        # Here we only use one neighbor, and indexes is a vector of len(xy) where an element indexes(i)
        # says that the closest point to xy[i] is coords[indexes[i]]. Since coords is just a flattened/reshaped version of intersection points
        #, the data cube can be resampled by datacube_flat=datacube.reshape((dim1*dim2, dim3)) and 
        # geographic_datacube_flat = datacube.reshape((dim1_geo*dim2_geo, dim3)) so that geographic_datacube_flat = datacube_flat[indexes,:]
        n_neighbors = 1
        dist, indexes = tree.kneighbors(xy, n_neighbors)

        # We can mask the data by allowing points within a radius of 2x the resolution
        mask_nn = dist > 2*resolution
        
        return transform, height, width, indexes, suffix, mask_nn

    @staticmethod
    def write_datacube_ENVI(memmap_gen_params, nodata, transform, datacube_path, wavelengths, fwhm, metadata, interleave, crs):
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

        nx = memmap_gen_params['height']
        mx = memmap_gen_params['width']
        k = memmap_gen_params['datacube'].shape[1] # is collapsed datacube

        dtype_cube = memmap_gen_params['datacube'].dtype # is collapsed datacube

        # Make some simple modifications
        data_file_path = datacube_path + '.' + interleave
        header_file_path = datacube_path + '.hdr'
        

        if os.path.exists(data_file_path):
            os.remove(data_file_path)
            os.remove(header_file_path)

        
        # Create metadata for the output raster dataset
        crs_gdal = osr.SpatialReference()
        
        crs_gdal.ImportFromEPSG(int(crs.split(':')[1]))

        # Create 1x1x1 dummy file to exploit builtin driver
        with rasterio.open(data_file_path, 'w', driver='ENVI', height=1, width=1, count=1, crs=crs, dtype=dtype_cube, transform=transform, nodata=nodata) as dst:
            pass

        # Then remove 
        os.remove(data_file_path)

        header = sp.io.envi.read_envi_header(datacube_path + '.hdr') # Open for extraction
        
        # Since dummy image was made as 1x1x1 data cube
        metadata_dim = {
            'lines': nx,
            'samples': mx,
            'bands': k
        }

        # Write all meta data to header
        for meta_key, value in metadata_dim.items():
            header[meta_key] = value

        # Reshape the ortho_datacube to (nx, mx, k)
        #ortho_datacube_reshaped = ortho_datacube.transpose(1, 2, 0)

        # Create the output raster dataset
        # Create ENVI image without using a context manager
        dst = envi.create_image(datacube_path + '.hdr', interleave='bsq', metadata=header, force=True)

        mm = dst.open_memmap(writable=True)

        GeoSpatialAbstractionHSI.write_datacube_memmap(memmap_array=mm, **memmap_gen_params)
        
        header = sp.io.envi.read_envi_header(datacube_path + '.hdr') # Open for extraction
        header.pop('band names')

        # Nobody in the history of the world could have come up with a more annoying bug.
        # Apparently, the CRS string is written with white spaces by rasterio/GDAL, wheras it should have none.
        header['coordinate system string'] = '{' + ",".join(header['coordinate system string']) + '}'
        
        # Write all meta_data to header
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

        header.pop('band names') # Remove defaults

        for meta_key, value in metadata.items():
            header[meta_key] = value

        sp.io.envi.write_envi_header(fileName=header_file_path, header_dict=header)

    @staticmethod
    def write_ancillary_ENVI_envi(nodata, transform, anc_path, metadata, interleave, crs, memmap_gen_params):
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
        nx = memmap_gen_params['height']
        mx = memmap_gen_params['width']
        k = memmap_gen_params['datacube'].shape[1] # is collapsed datacube

        dtype_cube = memmap_gen_params['datacube'].dtype # is collapsed datacube
        # Make some simple modifications
        data_file_path = anc_path + '.' + interleave
        header_file_path = anc_path + '.hdr'

        #if os.path.exists(header_file_path):
        #    return
        

        if os.path.exists(data_file_path):
            os.remove(data_file_path)
            os.remove(header_file_path)


        # Create metadata for the output raster dataset
        crs_gdal = osr.SpatialReference()
        
        crs_gdal.ImportFromEPSG(int(crs.split(':')[1]))

        # Create 1x1x1 dummy file to exploit builtin driver
        with rasterio.open(data_file_path, 'w', driver='ENVI', height=1, width=1, count=1, crs=crs, dtype=dtype_cube, transform=transform, nodata=nodata) as dst:
            pass

        # Then remove 
        os.remove(data_file_path)

        # Open for extraction
        header = sp.io.envi.read_envi_header(anc_path + '.hdr')

        # Since dummy image was made as 1x1x1 data cube
        metadata_dim = {
            'lines': nx,
            'samples': mx,
            'bands': k
        }

        # Write all dimensions to header
        for meta_key, value in metadata_dim.items():
            header[meta_key] = value
        
        # Create ENVI image without using a context manager
        dst = envi.create_image(anc_path + '.hdr', interleave='bsq', metadata=header, force=True)

        mm = dst.open_memmap(writable=True)

        # Write to the memory map
        GeoSpatialAbstractionHSI.write_datacube_memmap(memmap_array=mm, **memmap_gen_params)
        
        header = sp.io.envi.read_envi_header(anc_path + '.hdr') # Open for extraction
        header.pop('band names')

        # Nobody in the history of the world could have come up with a more annoying bug.
        # Apparently, the CRS string is written with white spaces by rasterio/GDAL, wheras it should have none.
        header['coordinate system string'] = '{' + ",".join(header['coordinate system string']) + '}'
        
        # Write all meta_data to header
        for meta_key, value in metadata.items():
            header[meta_key] = value

        sp.io.envi.write_envi_header(fileName=header_file_path, header_dict=header)



    @staticmethod
    def compare_hsi_composite_with_rgb_mosaic(hsi_composite_path, ref_ortho_reshaped_path):
        """Compares an HSI orthomosaic """
        
        

        
        # The RGB orthomosaic after reshaping (the reference)
        raster_rgb = gdal.Open(ref_ortho_reshaped_path, gdal.GA_Update)
        xoff1, a1, b1, yoff1, d1, e1 = raster_rgb.GetGeoTransform()  # This should be equal
        raster_rgb_array = np.array(raster_rgb.ReadAsArray())
        R = raster_rgb_array[0, :, :].reshape((raster_rgb_array.shape[1], raster_rgb_array.shape[2], 1))
        G = raster_rgb_array[1, :, :].reshape((raster_rgb_array.shape[1], raster_rgb_array.shape[2], 1))
        B = raster_rgb_array[2, :, :].reshape((raster_rgb_array.shape[1], raster_rgb_array.shape[2], 1))
        # del raster_array1
        ortho_rgb = np.concatenate((R, G, B), axis=2)
        rgb_image = Imcol(ortho_rgb)

        # The HSI composite raster (the match)
        raster_hsi = gdal.Open(hsi_composite_path)
        raster_hsi_array = np.array(raster_hsi.ReadAsArray())
        xoff2, a2, b2, yoff2, d2, e2 = raster_hsi.GetGeoTransform()
        transform_pixel_projected = raster_hsi.GetGeoTransform()
        R = raster_hsi_array[0, :, :].reshape((raster_hsi_array.shape[1], raster_hsi_array.shape[2], 1))
        G = raster_hsi_array[1, :, :].reshape((raster_hsi_array.shape[1], raster_hsi_array.shape[2], 1))
        B = raster_hsi_array[2, :, :].reshape((raster_hsi_array.shape[1], raster_hsi_array.shape[2], 1))
        
        ortho_hsi = np.concatenate((R, G, B), axis=2).astype(np.float64)

        # Some form of image processing

        no_data_value = raster_hsi.GetRasterBand(1).GetNoDataValue()

        ortho_valid = ortho_hsi[ortho_hsi[:,:,0] != no_data_value]

        # Iterate the colors
        for i in range(3):
            # Enhancing colors somewhat
            max_val = np.percentile(ortho_valid[:,i].reshape(-1), 95)
            min_val = 0

            ortho_hsi[:,:,i] = (ortho_hsi[:,:,i]-min_val)/(max_val - min_val)

            # Remove out-of-bounds
            ortho_hsi[ortho_hsi[:,:,i] > 1, i] = 1
            ortho_hsi[ortho_hsi[:,:,i] < 0, i] = 0
        
        
        
        ortho_hsi = ortho_hsi * 255
        ortho_hsi[ortho_hsi == 0] = 255
        hsi_image = Imcol(ortho_hsi)

        # Adjust Clahe
        #hsi_image.clahe_adjustment()
        rgb_image.clahe_adjustment()

        # Radiance is equivalent to a linear "color space". We run it through an inverse gamma to match the RGB data
        hsi_image.to_luma(gamma=False, 
                          image_array = hsi_image.image_array, 
                          gamma_inverse=True, 
                          gamma_value=0.45)
        
        hsi_image.clahe_adjustment(is_luma = True)

        # Tendency that RGB images are already transformed
        rgb_image.to_luma(gamma=False, image_array= rgb_image.clahe_adjusted)

        uv_vec_hsi, uv_vec_rgb, diff_AE_pixels = GeoSpatialAbstractionHSI.compute_sift_difference(hsi_image.luma_array, rgb_image.luma_array)


        image_resolution = a2
        diff_AE_meters = diff_AE_pixels*image_resolution
        return uv_vec_hsi, uv_vec_rgb, diff_AE_meters, transform_pixel_projected


    @staticmethod
    def resample_rgb_ortho_to_hsi_ortho(ref_ortho_path, hsi_composite_path, ref_ortho_reshaped_path):
        """Reproject RGB orthophoto to match the shape and projection of HSI raster.

        Parameters
        ----------
        infile : (string) path to input file to reproject
        match : (string) path to raster with desired shape and projection
        outfile : (string) path to output file tif
        """

        infile = ref_ortho_path
        match = hsi_composite_path
        outfile = ref_ortho_reshaped_path
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
    
    @staticmethod
    def resample_dem_to_hsi_ortho(dem_path, hsi_composite_path, dem_reshaped):
        """Reproject a file to match the shape and projection of existing raster.
        see https://rasterio.readthedocs.io/en/stable/topics/reproject.html

        Parameters
        ----------
        infile : (string) path to input file to reproject
        match : (string) path to raster with desired shape and projection
        outfile : (string) path to output file tif
        """

        infile = dem_path
        matchfile = hsi_composite_path
        outfile = dem_reshaped
        # open input
        
        with rasterio.open(infile) as src:
            src_transform = src.transform # DEM

            # open input to match
            with rasterio.open(matchfile) as match:
                dst_crs = match.crs # desired crs and transform

                # calculate the output transform matrix

                if src.crs.is_geographic:
                    tmp_file = 'tmp.tif'

                    # calculate the output transform matrix
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs,  # input CRS
                        dst_crs,  # output CRS
                        src.width,  # input width
                        src.height,  # input height
                        *src.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                    )

                    # Set properties for tmp:
                    dst_kwargs = src.meta.copy()
                    dst_kwargs.update({"crs": dst_crs,
                                    "transform": dst_transform,
                                    "width": dst_width,
                                    "height": dst_height,
                                    "nodata": 0})

                    with rasterio.open(tmp_file, 'w', **dst_kwargs) as src_prj:
                        # Reproject the DEM data in-place
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(src_prj, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=None,# No cropping
                            dst_crs=dst_crs,
                            resampling=Resampling.cubic)

                    src.close()
                    match.close()
                    
                    # Now that the file is written, we can recursively call
                    GeoSpatialAbstractionHSI.resample_dem_to_hsi_ortho(tmp_file, matchfile, outfile)

                    os.remove(tmp_file)
                    return 


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

    @staticmethod
    def compute_sift_difference(gray1, gray2):
        """Simply matches two grayscale images using SIFT"""

        gray1 = (gray1 - np.min(gray1)) / (np.max(gray1) - np.min(gray1))
        gray2 = (gray2 - np.min(gray2)) / (np.max(gray2) - np.min(gray2))

        gray1 = (gray1 * 255).astype(np.uint8)
        gray2 = (gray2 * 255).astype(np.uint8)


        # Find the keypoints and descriptors with SIFT
        sift = cv.SIFT_create()
        kp2, des2 = sift.detectAndCompute(gray2, None)
        kp1, des1 = sift.detectAndCompute(gray1, None)


        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        good = []
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        # store all the good matches as per Lowe's ratio test (0.7). 
        # Cranking up gives more. We changed 0.8 to 0.85 to get more matches
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
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
        """plt.imshow(img3, 'gray')
        plt.show()"""
        print(len(good))
##
        # The absolute errors
        diff_AE = np.sqrt(diff_u ** 2 + diff_v ** 2)
        
        return uv_vec_hsi, uv_vec_rgb, diff_AE


    def compute_reference_points_ecef(uv_vec_ref, transform_pixel_projected, dem_resampled_path, epsg_proj, epsg_geocsc=4978):
    
    
        """Computes ECEF reference points from a features detected on the orthomosaic and the DEM"""
        x = uv_vec_ref[:, 0] # In range 0 -> w
        y = uv_vec_ref[:, 1] # In range 0 -> h

        # Sample the terrain raster to get a projected position of data 
        raster_dem = rasterio.open(dem_resampled_path)
        xoff, a, b, yoff, d, e = transform_pixel_projected

        # Convert the pixel coordinates into true coordinates (e.g. UTM N/E)
        xp = a * x + b * y + xoff
        yp = d * x + e * y + yoff
        zp = np.zeros(yp.shape)
        for i in range(xp.shape[0]):
            temp = [x for x in raster_dem.sample([(xp[i], yp[i])])]
            zp[i] = float(temp[0])

        # Transform points to true 3D via pyproj
        geocsc = CRS.from_epsg(epsg_geocsc)
        proj = CRS.from_epsg(epsg_proj)
        transformer = Transformer.from_crs(proj, geocsc)
        ref_points_ecef = np.zeros((xp.shape[0], 3))
        (x_ecef, y_ecef, z_ecef) = transformer.transform(xx=xp, yy=yp, zz=zp)

        ref_points_ecef[:, 0] = x_ecef
        ref_points_ecef[:, 1] = y_ecef
        ref_points_ecef[:, 2] = z_ecef
        
        return ref_points_ecef

    def compute_position_orientation_features(uv_vec_hsi, pixel_nr_image, unix_time_image, position_ecef, quaternion_ecef, time_pose, nodata):
        """Returns the positions, orientations and pixel numbers corresponding to the features. 
        Also computes a feature mask identifying features that are invalid"""
        rows = uv_vec_hsi[:, 1]
        cols = uv_vec_hsi[:, 0]

        # Determine mask
        x0 = np.floor(cols).astype(int)
        x1 = x0 + 1
        y0 = np.floor(rows).astype(int)
        y1 = y0 + 1

        # All 4 neighbors (as used by bilinear interpolation) should be valid data points
        feature_mask = np.all([pixel_nr_image[y0, x0].reshape(-1) != nodata,
               pixel_nr_image[y1, x0].reshape(-1) != nodata,
               pixel_nr_image[y0, x1].reshape(-1) != nodata,
               pixel_nr_image[y1, x1].reshape(-1) != nodata], axis=0)


        pixel_nr_vec = GeoSpatialAbstractionHSI.bilinear_interpolate(pixel_nr_image, x = cols, y = rows)
        time_vec = GeoSpatialAbstractionHSI.bilinear_interpolate(unix_time_image, x = cols, y = rows)

        pixel_vec_valid = pixel_nr_vec[feature_mask]
        time_vec_valid = time_vec[feature_mask]

        
        from scipy.interpolate import interp1d
        from gref4hsi.utils import geometry_utils as geom
        

        rotation_ecef = RotLib.from_quat(quaternion_ecef)

        # Interpolates positions linearly and quaterinons by Slerp
        position_vec, quaternion_vec = geom.interpolate_poses(time_pose, 
                                            position_ecef, 
                                            rotation_ecef,  
                                            timestamps_to=time_vec_valid)
        
        return pixel_vec_valid, time_vec_valid, position_vec, quaternion_vec, feature_mask
        

        




    # COPIED from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    @staticmethod
    def bilinear_interpolate(im, x, y):
        
        x = np.asarray(x)
        y = np.asarray(y)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1);
        x1 = np.clip(x1, 0, im.shape[1]-1);
        y0 = np.clip(y0, 0, im.shape[0]-1);
        y1 = np.clip(y1, 0, im.shape[0]-1);

        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        return wa*Ia + wb*Ib + wc*Ic + wd*Id


    

def _get_max_value(dtype):
    """Gets the maximum value for a given data type.

    Args:
        dtype: The data type.

    Returns:
        The maximum value for the data type.
    """

    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).max
    else:
        raise ValueError("Unsupported data type: {}".format(dtype))

