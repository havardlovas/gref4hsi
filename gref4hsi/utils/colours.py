
import numpy as np

import cv2 as cv

class Image():
    def __init__(self, array):
        """Takes in an image array with nxmx3 of type uint8"""
        self.image_array = array

    def clahe_adjustment(self, is_luma = False):
        tile_size = 8
        clip_lim = 3
        clahe = cv.createCLAHE(clipLimit=clip_lim, tileGridSize=(tile_size, tile_size))
        if is_luma:
            img = (self.luma_array*255).astype(np.uint8) # 0->1

            # Apply equalization and adjust luma array
            self.luma_array = clahe.apply(img).reshape((img.shape[0], img.shape[1], 1))

        else:
            
            img = self.image_array
            R = clahe.apply(img[:, :, 0]).reshape((img.shape[0], img.shape[1], 1))
            G = clahe.apply(img[:, :, 1]).reshape((img.shape[0], img.shape[1], 1))
            B = clahe.apply(img[:, :, 2]).reshape((img.shape[0], img.shape[1], 1))

            self.clahe_adjusted = np.concatenate((R, G, B), axis=2)

    def to_luma(self, gamma, image_array, gamma_inverse = False, gamma_value = 0.45):
        """Takes in an image array with nxmx3 of type uint8 and transforms it to luminance with or without gamma compensation"""
        # Image is transformed into sRGB luminance
        R = (image_array[:, :, 2] / 255)

        G = (image_array[:, :, 1] / 255)

        B = (image_array[:, :, 0] / 255)

        # It is possible to apply a Gamma Conversion to the results as examplified below
        if gamma:
            R[R <= 0.04045] = R[R < 0.04045] / 12.92
            R[R > 0.04045] = ((R[R > 0.04045] + 0.055) / 1.055) ** 2.4
            G[G <= 0.04045] = G[G < 0.04045] / 12.92
            G[G > 0.04045] = ((G[G > 0.04045] + 0.055) / 1.055) ** 2.4
            B[B <= 0.04045] = B[B < 0.04045] / 12.92
            B[B > 0.04045] = ((B[B > 0.04045] + 0.055) / 1.055) ** 2.4

        # Linear approach for calculating CIE luminance rec. 709, ref Eq. (7).
        self.luma_array =  0.2125 * R + 0.7154 * G + 0.0721 * B

        if gamma_inverse:
            self.luma_array = np.power(self.luma_array, gamma_value).clip(0,1)

