# import required libraries
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import rasterio
from PIL import Image
from osgeo import ogr, gdal, osr
import numpy as np
import h5py
def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

class DataLogger:
    def __init__(self, filename, header):
        self.filename = filename
        #Generate file with header
        with open(filename,'w') as fh:
            fh.write(header + '\n')
    def append_data(self, data):
        #Append data line to file
        with open(self.filename,'a') as fh:
            #Generate data line to file
            if data[0] != None:
                line = ','.join([str(el) for el in data])

                #Append line to file
                fh.write(line + '\n')

class HyperspectralH5():
    def __init__(self, filename):

        f = h5py.File(filename, 'r', libver='latest')
        # Here we get all the keys and can write all attrinutes to corresponding Pythonic structures
        keys_data = get_dataset_keys(f)

        for i in range(len(keys_data)):
            dataset = f[keys_data[i]]
            names = keys_data[i].split(sep='/')
            if names[len(names)-1] == 'timestamp':
                setattr(self, names[len(names) - 2]+names[len(names) - 1], dataset[()])
            else:
                setattr(self, names[len(names)-1], dataset[()])


        # We will transform datacube to radiance
        DF = self.darkFrame
        t_exp = self.exposureTime[0]
        RF = self.radiometricFrame
        self.dataCubeRadiance = np.zeros(self.dataCube.shape)
        for i in range(self.dataCube.shape[0]):
            self.dataCubeRadiance[i, :, :] = (self.dataCube[i, :, :] - DF) / (
                    RF * t_exp)


        self.hyperspectraltimestamp[self.hyperspectraltimestamp < np.min(self.rgbtimestamp)] = 0
        self.hyperspectraltimestamp[self.hyperspectraltimestamp > np.max(self.rgbtimestamp)] = 0


        self.datacube_timestamp = self.hyperspectraltimestamp[self.hyperspectraltimestamp !=0 ]
        self.dataCubeRadiance_valid = self.dataCubeRadiance[self.hyperspectraltimestamp !=0 , :, :]



def pixel2coord(x, y):
    raster = gdal.Open("rasters/raster.tif")
    raster_array = np.array(raster.ReadAsArray())
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return (xp, yp)
def coord2pix(xp, yp):
    raster = gdal.Open("rasters/raster.tif")
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()
    A = np.array([a, b], [d, e])
    vp = np.array([xp -  xoff, yp -  yoff]).reshape((3, 1))
    v = np.inv(A)*vp
    return (v[0], v[1])

def compute_sift_matching(img1, img2):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    plt.imshow(img2)
    plt.show()
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray')
    plt.show()

    # Since the let us try and extract the pixel coordinates for the big image
    print(len(good))
def clahe_adjustment(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    R = clahe.apply(img[:,:,0]).reshape((img.shape[0], img.shape[1], 1))
    G = clahe.apply(img[:, :, 1]).reshape((img.shape[0], img.shape[1], 1))
    B = clahe.apply(img[:, :, 2]).reshape((img.shape[0], img.shape[1], 1))
    img_adjusted = np.concatenate((R, G, B), axis = 2)
    return img_adjusted

def adjust_gamma(img, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv.LUT(img, table)

def compute_control_points():
    # Write all control points to the same file
    args = sys.argv[1:]


    filename_csv = "D:/SeaBee/control_points_all_clahe_silly.csv"
    filename_csv = args[0]
    dlogr_control_points = DataLogger(filename_csv,
                                      'ControlPointidx, column_x_u, row_y_v, North, East, ElevationEllipsoid')

    gis_folder = args[1]  #



    # Use digital surface model
    raster_dsm = rasterio.open(gis_folder + "DEM.tif")
    raster_dsm_geoid = rasterio.open(gis_folder + "DEMGeoid.tif")
    print('Geoid Loaded')

    sift = cv.SIFT_create()

    if use_one_orthomosaic:
        # Import original TIF-file
        print('Image loaded')
        fn_raster = gis_folder + "OrthoMosaic.tif"
        # Match the formats in terms of resolution and offsets to create equal images
        raster = gdal.Open(fn_raster)  # used for extracting images
        print('Image Raster Loaded')
        # raster_array = np.array(raster.ReadAsArray())
        xoff, a, b, yoff, d, e = raster.GetGeoTransform()

        raster_array = np.array(raster.ReadAsArray())
        R = raster_array[0, :, :].reshape((raster_array.shape[1], raster_array.shape[2], 1))
        G = raster_array[1, :, :].reshape((raster_array.shape[1], raster_array.shape[2], 1))
        B = raster_array[2, :, :].reshape((raster_array.shape[1], raster_array.shape[2], 1))
        img2 = np.concatenate((R, G, B), axis=2)

        kp2, des2 = sift.detectAndCompute(img2, None)
        print('Key points found')

    ind = np.array([[4200, 6800], [7900, 10400], [11400, 15900], [17000, 22600], [23700, 29100], [30200, 36300],
                   [37300, 41300]])

    # read two input images as grayscale





    for i in range(2,ind.shape[0]):
        line_offset = ind[i, 0]  # Start index of image

        raster = gdal.Open('D:/SeaBee/FolderOfCroppedDEMs/' + str(line_offset) + '.tif')  # used for extracting images
        print('Image Raster Loaded')
        # raster_array = np.array(raster.ReadAsArray())
        xoff, a, b, yoff, d, e = raster.GetGeoTransform()
        del raster

        img2 = cv.imread('D:/SeaBee/FolderOfCroppedDEMs/' + str(line_offset) + '.tif')
        img2 = np.flip(img2, axis=2)
        #img2 = clahe_adjustment(img2)
        print('Image loaded')

        kp2, des2 = sift.detectAndCompute(img2, None)
        print('Key points found')


        imgScan = cv.imread('D:/SeaBee/RGB_images/' + str(line_offset) +
        '_s.jpg')

        # imgScan_resize = cv.resize(imgScan, (imgScan.shape[1]*2, imgScan.shape[0]*2))
        imgScan_resize = imgScan

        img1 = np.rot90(imgScan_resize, 3)
        img1 = np.flip(img1, axis = 2)
        #img1 = clahe_adjustment(img1)
        img1 = adjust_gamma(img1, 2.2)


        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           flags=2)
        img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        print(len(good))
        plt.imshow(img3, 'gray')
        plt.show()

        # Since the let us try and extract the pixel coordinates for the big image


        for i in range(len(good)):
            idx2 = good[i].trainIdx
            idx1 = good[i].queryIdx
            uv1 = kp1[idx1].pt # Slit image
            uv2 = kp2[idx2].pt # Orthomosaic

            ## Conversion to global coordinates
            x = uv2[0]
            y = uv2[1]
            xp = a * x + b * y + xoff
            yp = d * x + e * y + yoff

            # Sample the dsm= [x for x in src.sample(coord_list)]
            zp = [x for x in raster_dsm.sample([(xp, yp)])] # NN value
            z_geoid = [x for x in raster_dsm_geoid.sample([(xp, yp)])] # The height of the NN ellipsoid

            dlogr_control_points.append_data(np.array([i, uv1[0], uv1[1] + line_offset, yp, xp, float(zp[0]) + float(z_geoid[0])]))



        #plt.scatter(np.array([0,0]), np.array([0,0]))
        #plt.show()

hyp_path = 'E:/NansenLegacyUHI/Transect48/uhi_20210506_083101_2.h5'
hyp = HyperspectralH5(hyp_path)

import spectral as sp

hyp_rgb_im = hyp.dataCubeRadiance_valid[:, :, [150, 97, 50]]
hyp_rgb_im *= (255/hyp_rgb_im.max())



rgb_im = np.asarray(Image.open('E:/NansenLegacyUHI/GISHongbo/mesh_contrast_adjusted.png')).astype(np.uint8)

hyp_rgb_im = adjust_gamma(hyp_rgb_im.astype(np.uint8), gamma = 1/1.5)

compute_sift_matching(img1=rgb_im, img2=hyp_rgb_im.astype(np.uint8))


#plt.imshow(rgb_im/rgb_im.max())
#plt.show()
#sp.imshow(np.flip(hyp.dataCubeRadiance_valid, axis = 0), [150, 97, 50])
#plt.pause(10)

#compute_control_points()























