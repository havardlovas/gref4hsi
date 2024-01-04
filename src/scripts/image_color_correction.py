# creata a class called Images
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import time
import argparse
import glob
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='Process .h5 files to extract RGB data and IMU data.')
    parser.add_argument('--image_dir_raw', type=str, default="", help='Path to the folder containing uncorrected images.')
    parser.add_argument('--image_dir_corrected', type=str, default="", help='Path to the folder where corrected images are.')
    parser.add_argument('--mean_desired_intensity', type=int, default=90, help='Desired mean after correction.')
    parser.add_argument('--std_desired_intensity', type=int, default=30, help='Desired std after correction.')
    return parser.parse_args()

class GrayWorldCorrection:
    def __init__(self, image_dir, image_dir_write):
        """This class is ment to perform gray-world correction of images in a directory and write the corrected files to a different directory"""
        self.image_dir = image_dir
        self.image_dir_write = image_dir_write
        # Find all .h5 files in the folder
        self.image_paths = glob.glob(os.path.join(self.image_dir, '*.png'))
        self.n_images = len(self.image_paths)

        # Define self.n_rows, self.n_columns
        path = self.image_paths[0]

        self.n_rows, self.n_columns, _ = np.asarray(Image.open(path)).shape

    def calculate_avg_image_x(self):
        # Initiate an index to be used for division
        idx = 0


        # Initiate an average image array
        avg_sum = np.zeros((self.n_rows, self.n_columns, 3), dtype = 'float64')

        for filename in self.image_paths:
            
            # Create file path by merging the directory and filename
            path = filename
            img_x = np.asarray(Image.open(path))
            avg_sum += img_x[:,:,0:3]
            idx += 1
            if idx % 100:
                print(100*idx/self.n_images)
        self.mu_x = (avg_sum / idx).astype('float64')
        self.n_images = idx
        #np.save('mu_x', self.mu_x)
    def calculate_sigma_image_x(self):

        sigma_sum = np.zeros((self.n_rows, self.n_columns, 3), dtype='float64')
        idx = 0
        for filename in self.image_paths:
            # Create file path by merging the directory and filename
            path = filename
            img = np.asarray(Image.open(path))
            sigma_sum += (img[:,:,0:3]-self.mu_x)**2
            idx += 1
            if idx % 100:
                print(100*idx/self.n_images)
        self.sigma_x = np.sqrt(sigma_sum / idx).astype('float64')
        np.save('sigma_x', self.sigma_x)

    def grey_world_correction(self, mean_intensity = 90, std_intensity = 30):

        self.mu_y = np.ones(self.sigma_x.shape)*mean_intensity
        self.sigma_y = np.ones(self.mu_x.shape)*std_intensity
        # Define first the average image as defined by the function

        # The intensity scaling of each pixel and waveband
        self.m = np.multiply(self.sigma_y, 1/self.sigma_x)
        # Apply an offset to the brightness.
        self.n = self.mu_y - np.multiply(self.m, self.mu_x)

        idx = 0
        t_start = time.time()
        for filename in self.image_paths:
            path = filename
            img_x = np.asarray(Image.open(path))[:, :, 0:3]

            # Apply transformation
            img_y = (self.m*(img_x - self.mu_x) + self.mu_y)
            # Make sure values are bounded
            img_y[img_y > 255] = 255
            img_y[img_y < 0] = 0
            img_y = img_y.astype('uint8')

            filename_no_ext = filename.split(sep = '\\')[1].split(sep='.')[0]


            matplotlib.image.imsave(self.image_dir_write + filename_no_ext + '.png', img_y)
            if idx % 100 == 0 and idx != 0:
                t = time.time()
                #print('Final transformation')
                print(100*idx/self.n_images)
                print('Time left = ' + str((self.n_images-idx)*((t-t_start)/idx)) + ' s')
            idx += 1



def remove_lf_component(image_path_raw):
    

    # Load the image
    image = cv2.imread(image_path_raw)

    # Convert the image to float32 for better precision during calculations
    image_float = image.astype(np.float32) / 255.0

    # Separate the image into color channels
    b, g, r = cv2.split(image_float)

    # Apply the process to each color channel
    def extract_high_frequency(channel):
        # Apply a Gaussian filter to obtain the low-frequency component
        low_freq = cv2.GaussianBlur(channel, (15, 15), 0)

        # Subtract the low-frequency component from the original channel to get the high-frequency component
        high_freq = channel - low_freq

        return high_freq

    # Extract high-frequency components for each color channel
    high_freq_r = extract_high_frequency(r)
    high_freq_g = extract_high_frequency(g)
    high_freq_b = extract_high_frequency(b)

    # Combine the high-frequency components to obtain the color image
    high_freq_image = cv2.merge([high_freq_b, high_freq_g, high_freq_r])

    # Display the original image and the color image equivalent of the high-frequency component
    cv2.imshow('Original Image', image)
    cv2.imshow('High-Frequency Component (Color)', (high_freq_image * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image_dir_raw, image_dir_corrected, mean_desired_intensity = 90, std_desired_intensity = 30):
    gwd = GrayWorldCorrection(image_dir=image_dir_raw, 
                              image_dir_write=image_dir_corrected)
    gwd.calculate_avg_image_x()
    gwd.calculate_sigma_image_x()
    gwd.grey_world_correction(mean_intensity=mean_desired_intensity, 
                              std_intensity=std_desired_intensity)

if __name__ == '__main__':
    args = get_args()
    main(image_dir_raw=args.image_dir_raw, 
         image_dir_corrected=args.image_dir_corrected,
         mean_desired_intensity = args.mean_desired_intensity,
         std_desired_intensity=args.std_desired_intensity)