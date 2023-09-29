
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
from radiometry import Radiometry
from collections import namedtuple
import visualize
from georeference import Hyperspectral
import configparser
from sklearn.neighbors import NearestNeighbors
import spectral as sp
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use('TkAgg')

def linear_regressor(x, y, fit_intercept = True):
    """
    Fits a linear regression model to the given data and returns the intercept and slope.

    Parameters:
    x (numpy array): The x-coordinates of the data points.
    y (numpy array): The y-coordinates of the data points.

    Returns:
    float: The intercept of the linear regression line.
    float: The slope of the linear regression line.
    """
    # Create a LinearRegression object
    model = LinearRegression(fit_intercept=fit_intercept)

    # Reshape x to a 2D array (required by scikit-learn)
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]

    x = x.reshape(-1, 1)
    if x.shape[0] != 0:
        # Fit the model to the data
        model.fit(x, y)

        y_pred = model.predict(x)
        r_squared = r2_score(y, y_pred)

        # Extract the intercept and slope from the model
        if fit_intercept:
            intercept = model.intercept_
        else:
            intercept = 0

        slope = model.coef_[0]
    else:
        intercept = -1
        slope = -1
        r_squared = -1

    return intercept, slope, r_squared

def save_rgb_composite(path, datacube):
    R_1 = datacube[:, :, 150]
    R_1 = (R_1 - R_1.min()) / (R_1.max() - R_1.min())

    G_1 = datacube[:, :, 97]
    G_1 = (G_1 - G_1.min()) / (G_1.max() - G_1.min())

    B_1 = datacube[:, :, 50]
    B_1 = (B_1 - B_1.min()) / (B_1.max() - B_1.min())

    image = np.concatenate((R_1.reshape((-1, 960, 1)), G_1.reshape((-1, 960, 1)), B_1.reshape((-1, 960, 1))), axis=2)*255
    plt.imsave(path, image.astype(np.uint8))

def image2mask(fn):
    image = cv2.imread(fn)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to convert to binary mask
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Binary Mask', binary_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return binary_mask

def optimize_function(radObj, G, X, Y, alpha, I_0, sigma_l):
    radObj.set_parameters(G, X, Y, alpha, I_0, sigma_l)

    radObj.run_forward_model(is_skogn=True)

    error = np.sum((radObj.refl_1 - radObj.refl_2)**2, axis = 1)
    print(np.mean(error))
    print([G, X, Y, alpha, I_0, sigma_l])



    return error

def optimize_function_WP(radObjList, G, X, Y, alpha, I_0, sigma_l):
    radObjWP = radObjList.WhitePlateObj
    radObj = radObjList

    radObj.set_parameters(G, X, Y, alpha, I_0, sigma_l)
    radObjWP.set_parameters(G, X, Y, alpha, I_0, sigma_l)

    radObj.run_forward_model(is_skogn=True, model = 'two way')
    # White plate data
    radObjWP.run_forward_model(is_skogn=True, model='two way')
    refl_gt = radObjWP.Const.refl_gt.reshape((1, -1))

    use_log_err = False
    if use_log_err:
        error = (np.log(radObj.refl_1[:, 25:175]) - np.log(radObj.refl_2[:, 25:175])).reshape((1, -1))
        error1 = (np.log(radObjWP.refl_1[:, 25:175]) - np.log(refl_gt[:, 25:175])).reshape((1, -1))
        error2 = (np.log(radObjWP.refl_2[:, 25:175]) - np.log(refl_gt[:, 25:175])).reshape((1, -1))
    else:

        error = (radObj.refl_1[:, 25:175] - radObj.refl_2[:, 25:175]).reshape((1, -1))
        error1 = (radObjWP.refl_1[:, 25:175] - refl_gt[:, 25:175]).reshape((1, -1))
        error2 = (radObjWP.refl_2[:, 25:175] - refl_gt[:, 25:175]).reshape((1, -1))



    #plt.plot(bands, np.mean(radObj.refl_1, axis = 0))
    #s = np.std(radObj.refl_1, axis = 0)
    #plt.plot(bands, np.mean(radObj.refl_1 + s, axis=0))
    #plt.plot(bands, np.mean(radObj.refl_1 - s, axis=0))
    #plt.scatter(bands, radObj.refl_1[100, :])
    #plt.plot(bands, radObj.Const.refl_gt)
    #plt.show()







    print(np.median(np.abs(error1)))
    print(np.median(np.abs(error2)))
    print(np.median(np.abs(error)))

    print([G, X, Y, alpha, I_0, sigma_l])

    err_tot = np.concatenate((error1.reshape(-1), error2.reshape(-1), error.reshape(-1)))

    return np.log(np.abs(err_tot) + 1) + 10



def optimize_function_lsq(param, radObjList):
    G = param[0]
    X = param[1]
    Y = param[2]
    B = param[3]
    alpha = param[4]
    I_0 = param[5]
    sigma_l = param[6]
    k_l = param[7]


    radObjWP = radObjList.WhitePlateObj
    radObj = radObjList

    radObj.set_parameters(G, X, Y, B, alpha, I_0, sigma_l, k_l)
    radObjWP.set_parameters(G, X, Y, B, alpha, I_0, sigma_l, k_l)

    radObj.run_forward_model(is_skogn=True, model = 'two way')

    # White plate data
    radObjWP.run_forward_model(is_skogn=True, model='two way')
    refl_gt = radObjWP.Const.refl_gt.reshape((1, -1))

    use_log_err = False
    eps = 0.0001
    channel = np.arange(20, 180)
    if use_log_err:
        error1 = (np.log(radObjWP.refl_1[:, channel] + eps) - np.log(refl_gt[:, channel] + eps)).reshape((1, -1))
        error2 = (np.log(radObjWP.refl_2[:, channel] + eps) - np.log(refl_gt[:, channel] + eps)).reshape((1, -1))



        error = (np.log(radObj.refl_1[:, channel] + eps) - np.log(radObj.refl_2[:, channel]+ eps)).reshape((1, -1))
    else:

        error = (radObj.refl_1[:, channel] - radObj.refl_2[:, channel]).reshape((1, -1))
        error1 = (radObjWP.refl_1[:, channel] - refl_gt[:, channel]).reshape((1, -1))
        error2 = (radObjWP.refl_2[:, channel] - refl_gt[:, channel]).reshape((1, -1))
        error3 = (radObjWP.refl_2[:, channel] - radObjWP.refl_1[:, channel]).reshape((1, -1))


    bands = radObj.Var.bands1

    mu = np.median(radObj.refl_1 - radObj.refl_2, axis = 0)
    s = np.std(radObj.refl_1 - radObj.refl_2, axis = 0)

    print(bands[180])
    print(bands[50])
    plt.plot(bands, mu)
    plt.plot(bands, mu+s)
    plt.plot(bands, mu-s)
    plt.xlim([370, 720])
    plt.ylim([-1, 1])
    plt.show()
    muWP1 = np.median(radObjWP.refl_1, axis = 0)
    muWP2 = np.median(radObjWP.refl_2, axis = 0)
    plt.plot(bands, muWP1)
    plt.plot(bands, muWP2)
    plt.plot(bands, refl_gt.reshape((225, 1)))
# plt.plot(bands, np.mean(radObj.refl_1 - s, axis=0))
    # plt.scatter(bands, radObj.refl_1[100, :])
    plt.plot(bands, radObj.Const.refl_gt)
    plt.xlim([370, 720])
    plt.ylim([0, 1])
    plt.show()


    print(np.mean(np.abs(error1)))
    print(np.mean(np.abs(error2)))
    #print(np.mean(np.abs(error3)))
    print(np.mean(np.abs(error)))

    print([G, X, Y, B, alpha, I_0, sigma_l, k_l])

    err_tot = np.concatenate((error1.reshape(-1), error2.reshape(-1), error.reshape(-1)))

    return err_tot

def optimize_function_lsq_tautra(param, radObjList, plot_it, is_skogn):
    G = param[0]
    X = param[1]
    Y = param[2]
    B = param[3]
    alpha = param[4]
    I_0 = param[5]
    sigma_l = param[6]
    k_l = param[7]

    radObjWP = radObjList.WhitePlateObj
    radObj = radObjList

    radObj.set_parameters(G, X, Y, B, alpha, I_0, sigma_l, k_l)
    radObjWP.set_parameters(G, X, Y, B, alpha, I_0, sigma_l, k_l)

    radObj.run_forward_model(is_skogn=is_skogn, model='two way')

    radObjWP.run_forward_model(is_skogn=is_skogn, model='two way')



    refl_gt_coral = radObjWP.Const.refl_gt_coral.reshape((1, 224))
    bands = radObj.Var.bands1

    use_log_err = False
    eps = 0.0001
    channel = np.arange(10, 39) # 400-700 nm
    if use_log_err:
        error = (np.log(radObj.refl_1[:, channel] + eps) - np.log(radObj.refl_2[:, channel] + eps)).reshape((1, -1))
    else:
        refl_1_hat = radObj.refl_1
        refl_2_hat = radObj.refl_2
        error = (refl_1_hat[:, channel] - refl_2_hat[:, channel]).reshape(-1)
        error1 = (radObjWP.refl_1[:, channel] - refl_gt_coral[:, channel]).reshape(-1)
        error2 = (radObjWP.refl_2[:, channel] - refl_gt_coral[:, channel]).reshape(-1)
        #error_norm2 = radObj.refl_2.reshape((-1)) - 0.1

        #smoothed = radObj.Meas.rad1
        #from scipy.ndimage import gaussian_filter1d
        #for i in range(smoothed.shape[0]):
        #    smoothed[i,: ] = gaussian_filter1d(radObj.Meas.rad1[i, :], sigma=10)
        #    #radObj.Meas.rad1[i, :] = gaussian_filter1d(radObj.Meas.rad1[i, :], sigma=3)
        #    #radObj.Meas.rad2[i, :] = gaussian_filter1d(radObj.Meas.rad2[i, :], sigma=3)



        # Take radiance value 1:
        R1 = np.linalg.norm(radObj.Var.points1, axis=1).reshape((-1, 1))
        R2 = np.linalg.norm(radObj.Var.points2, axis=1).reshape((-1, 1))
        k = bands.reshape(-1).shape[0]
        a_arr = np.zeros(k)
        for i in range(k):
            x = R1
            y = -np.log((radObj.Meas.rad1[:, i]) / (radObj.Meas.rad1[:, 100]))
            b, a, r = linear_regressor(x = x, y = y)
            a_arr[i] = a/2
        #plt.plot(bands, a_arr- a_arr[150], label = 'Single Transect 1')



        #k = bands.reshape(-1).shape[0]
        #a_arr = np.zeros(k)
        #for i in range(k):
        #    x = R2
        #    y = -np.log((radObj.Meas.rad2[:, i]) / (radObj.Meas.rad2[:, 100]))
        #    b, a, r = linear_regressor(x=x, y=y)
        #    a_arr[i] = a / 2
        #plt.plot(bands, a_arr - a_arr[150], label='Single Transect 2')
#
        #k = bands.reshape(-1).shape[0]
        #a_arr = np.zeros(k)
        #for i in range(k):
        #    x = R2 - R1
        #    y = -np.log((radObj.Meas.rad2[:, i]) / (radObj.Meas.rad2[:, 100])) + np.log((radObj.Meas.rad1[:, i]) / (radObj.Meas.rad1[:, 100]))
        #    b, a, r = linear_regressor(x=x, y=y, fit_intercept=False)
        #    a_arr[i] = a/2
#
        #plt.plot(bands, a_arr - a_arr[150], label = 'Overlap norm')

#
        if is_skogn:
            k = bands.reshape(-1).shape[0]
            a_arr = np.zeros(k)
            r_arr = np.zeros(k)
            c1 = radObj.cos_gamma_s1_t1.reshape(-1)
            c2 = radObj.cos_gamma_s1_t2.reshape(-1)
            BP1 = radObj.BP1_norm_t1.reshape(-1)
            BP2 = radObj.BP1_norm_t2.reshape(-1)
            RS1 = radObj.R_s1_t1.reshape(-1)
            RS2 = radObj.R_s1_t2.reshape(-1)
            L1 = radObj.Meas.rad1
            L2 = radObj.Meas.rad2
            for i in range(k):
                x = (R2.reshape(-1) - R1.reshape(-1)) + (RS2 - RS1)
                y = np.log(L1[:, i]/L2[:, i]) - np.log((c1*BP1)/(c2*BP2)) + 2*np.log(RS1/RS2)

                x = (R1.reshape(-1) - R2.reshape(-1)) + (RS1 - RS2)
                y = np.log(L2[:, i] / L1[:, i]) - np.log((c2 * BP2 / (c1 * BP1))) + 2 * np.log(RS2 / RS1)
                # Say we ignore entries where L < 0.001

                valid_ind = np.where((L1[:, i].reshape(-1) > 0.000) & (L2[:, i].reshape(-1) > 0.000))[0]


                if len(valid_ind) > 1:
                    b, a, r = linear_regressor(x=x[valid_ind], y=y[valid_ind])
                    a_arr[i] = a
                    r_arr[i] = r

            print('Mean r2 score: ' + str(np.mean(r_arr[10:191])))

            plt.scatter(bands, r_arr, label='R2 score')
            plt.plot(bands, (radObj.Const.a_w + 0.5 * radObj.Const.b_w).reshape((-1, 1)), label='Pure Water [1/m]')
            plt.plot(bands, a_arr, label = 'Overlap-Estimated Attenuation [1/m]')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Attenuation [1/m]')
            plt.legend()
            plt.ylim([-0.2, 0.8])
            plt.xlim([400, 700])
            plt.pause(100)
            plt.show()

        else:

            k = bands.reshape(-1).shape[0]
            a_arr = np.zeros(k)
            r_arr = np.zeros(k)
            c1 = (radObj.cos_gamma_s1_t1.reshape(-1) + radObj.cos_gamma_s2_t1.reshape(-1))/2
            c2 = (radObj.cos_gamma_s1_t2.reshape(-1) + radObj.cos_gamma_s2_t2.reshape(-1))/2
            BP1 = (radObj.BP1_norm_t1.reshape(-1) + radObj.BP2_norm_t1.reshape(-1))/2
            BP2 = (radObj.BP1_norm_t2.reshape(-1) + radObj.BP2_norm_t2.reshape(-1))/2
            RS1 = (radObj.R_s1_t1.reshape(-1) + radObj.R_s2_t1.reshape(-1))/2
            RS11 = radObj.R_s1_t1.reshape(-1)
            RS12 = radObj.R_s2_t1.reshape(-1)
            RS21 = radObj.R_s1_t2.reshape(-1)
            RS22 = radObj.R_s2_t2.reshape(-1)
            RS2 = (radObj.R_s1_t2.reshape(-1) + radObj.R_s2_t2.reshape(-1))/2
            from scipy.ndimage import gaussian_filter1d
            L1 = radObj.Meas.rad1
            L2 = radObj.Meas.rad2
            R1 = np.linalg.norm(radObj.Var.points1, axis=1).reshape((-1))
            R2 = np.linalg.norm(radObj.Var.points2, axis=1).reshape((-1))
            ind = 2000


            plt.plot(bands, np.log(L1[ind, :]), label = 'Cam distance 1, source 11 and source 12 of ' + str(R1[ind])+','+ str(RS11[ind])+','+ str(RS12[ind]) + ','+ str(RS1[ind]))
            plt.plot(bands, np.log(L2[ind, :]), label = 'Cam distance 2, source 21 and source 22 of ' + str(R2[ind])+','+ str(RS21[ind])+','+ str(RS22[ind])+','+ str(RS2[ind]))
            plt.legend()
            plt.show()


            x = R2 - R1 + (RS2 - RS1)

            for i in range(k):
                y = (np.log(L2[:, i]*RS2**2 /(BP2*c2))) - (np.log(L1[:, i]*RS1**2 /(BP1*c1)))


                b, a, r = linear_regressor(x=x[np.abs(x) < 5], y=y[np.abs(x) < 5], fit_intercept=False)
                a_arr[i] = -a
                r_arr[i] = r
            #    if i % 10 == 0:
            #        plt.scatter(x[np.abs(x) < 5], y[np.abs(x) < 5], alpha = 0.05, label='Logarithm of difference')
            #        y_pred = a*(x[np.abs(x) < 5] + b)
            #        r_squared = r
            #        x_pred = np.linspace(-0.8,0.6, 10)
            #        plt.plot(x_pred, a*x_pred + b,'r',
            #                    label='Regression fit with attenuation: ' + "{:.2f}".format(-a/2) + ' and r2: ' + "{:.2f}".format(r_squared))
            #        plt.xlim([-0.8, 0.6])
            #        #plt.ylim([-0.4, 0.5])
            #
#
            #        plt.legend()
             #       plt.show()

            plt.scatter(bands, r_arr, label='R2 score')
            kappa_w = (radObj.Const.a_w + 0.5 * radObj.Const.b_w).reshape((-1, 1))
            plt.plot(bands, kappa_w, label='Pure Water [1/m]')
            plt.plot(bands, a_arr, label='Overlap-Estimated Attenuation [1/m]')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Attenuation [1/m]')
            plt.legend()
            plt.ylim([-0.2, 0.8])
            plt.xlim([400, 700])
            plt.pause(100)
            plt.show()



            for i in range(k):
                x = RS2 + R2 - (R1 + RS1)
                y = -(np.log(L2[:, i].reshape(-1)) + np.log(RS2**2) - np.log(BP2*c2)) + \
                    (np.log(L1[:, i].reshape(-1)) + np.log(RS1**2) - np.log(BP1*c1))

                valid_ind = np.where((L1[:, i].reshape(-1) > 0.00) & (L2[:, i].reshape(-1) > 0.00))[0]
                b, a, r = linear_regressor(x=x[valid_ind], y=y[valid_ind])
                a_arr[i] = a
                r_arr[i] = r
            print('Mean r2 score 2: ' + str(np.mean(r_arr[10:191])))
            corr_coef = np.corrcoef(L1[:,10:191].reshape(-1), L2[:,10:191].reshape(-1))[0, 1]

            print("Correlation coefficient between arr1 and arr2:", corr_coef)

            plt.scatter(bands, r_arr, label='R2 score')
            plt.plot(bands, (radObj.Const.a_w + 0.5 * radObj.Const.b_w).reshape((-1, 1)), label='Pure Water [1/m]')
            plt.plot(bands, a_arr, label='Overlap-Estimated Attenuation [1/m]')



            for i in range(k):
                x = RS1 + R1
                y = -(np.log(L1[:, i].reshape(-1)/L1[:, 100].reshape(-1)))

                #valid_ind = np.where((L1[:, i].reshape(-1) > 0.000) & (L2[:, i].reshape(-1) > 0.000))[0]
                b, a, r = linear_regressor(x=-x, y=-y)
                a_arr[i] = a
                r_arr[i] = r
            print('Mean r2 score 1: ' + str(np.mean(r_arr[10:191])))

            plt.scatter(bands, r_arr, label='R2 score')
            #plt.plot(bands, (radObj.Const.a_w + 0.5 * radObj.Const.b_w).reshape((-1, 1)), label='Pure Water [1/m]')
            plt.plot(bands, a_arr, label='Single Transect [1/m]')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Attenuation [1/m]')
            plt.legend()
            plt.ylim([-0.2, 0.8])
            plt.xlim([400, 700])
            plt.pause(100)
            plt.show()
#
#
#
#
        ## Take radiance value 1:
        #R1 = np.linalg.norm(radObjWP.Var.points1, axis=1).reshape((-1, 1))
        bands = radObjWP.Var.bands1
        #R2 = np.linalg.norm(radObj.Var.points2, axis=1).reshape((-1, 1))
##
        k = bands.reshape(-1).shape[0]
        #a_arr = np.zeros(k)
        #for i in range(k):
        #    x = R1
        #    y = -np.log(radObjWP.Meas.rad1[:, i] / (radObjWP.Meas.rad1[:, 100]))
        #    b, a, r = linear_regressor(x=x, y=y)
        #    a_arr[i] = a/2
        #plt.plot(bands, a_arr- a_arr[150], label = 'Using Lophelia 1')
        #plt.show()

        r_arr = np.zeros(k)

        # Averaging for the sources
        #c = ((radObjWP.cos_gamma_s1_t1.reshape(-1) + radObjWP.cos_gamma_s2_t1.reshape(-1))/2).reshape(-1)
        #BP = ((radObjWP.BP1_norm_t1.reshape(-1) + radObjWP.BP2_norm_t1.reshape(-1))/2).reshape(-1)
        #RS = ((radObjWP.R_s1_t1.reshape(-1) + radObjWP.R_s2_t1.reshape(-1))/2).reshape(-1)
        #R1 = np.linalg.norm(radObjWP.Var.points1, axis=1).reshape((-1))
        #L1 = radObjWP.Meas.rad1
        #k = bands.reshape(-1).shape[0]
        #a_arr = np.zeros(k)
        #for i in range(k):
        #    x = R1 + RS
        #    y = -(np.log(L1[:, i].reshape(-1)) + 2*np.log(RS) - np.log(c*BP))
        #    b, a, r = linear_regressor(x=x, y=y)
        #    a_arr[i] = a
        #    r_arr[i] = r
#
        #plt.plot(bands, (radObjWP.Const.a_w + 0.5*radObjWP.Const.b_w).reshape((-1, 1)), label = 'Pure Water [1/m]')
        #plt.plot(bands, a_arr, label = 'Estimated attenuation [1/m]')
        #plt.scatter(bands, r_arr, label='R2 score [-]')
        #plt.xlabel('Wavelength [nm]')
        #plt.ylabel('Attenuation [1/m]')
        #print('Mean r2 score Lophelia: ' + str(np.mean(r_arr[10:191])))
        #plt.legend()
        #plt.ylim([-0.2, 0.8])
        #plt.xlim([400, 700])
        #plt.show()

        # Averaging for the sources
        #c1 = ((radObjWP.cos_gamma_s1_t1.reshape(-1) + radObjWP.cos_gamma_s2_t1.reshape(-1)) / 2).reshape(-1)
        #c2 = ((radObjWP.cos_gamma_s1_t2.reshape(-1) + radObjWP.cos_gamma_s2_t2.reshape(-1)) / 2).reshape(-1)
        #BP1 = ((radObjWP.BP1_norm_t1.reshape(-1) + radObjWP.BP2_norm_t2.reshape(-1)) / 2).reshape(-1)
        #RS = ((radObjWP.R_s1_t2.reshape(-1) + radObjWP.R_s2_t2.reshape(-1)) / 2).reshape(-1)
        #R2 = np.linalg.norm(radObjWP.Var.points2, axis=1).reshape((-1))
        #L2 = radObjWP.Meas.rad2
        #k = bands.reshape(-1).shape[0]
        #a_arr = np.zeros(k)
        #for i in range(k):
        #    x = R2 + RS
        #    y = -(np.log(L2[:, i].reshape(-1)) + 2 * np.log(RS) - np.log(c * BP))
        #    b, a, r = linear_regressor(x=x, y=y)
        #    a_arr[i] = a
        #    r_arr[i] = r
#
        #plt.plot(bands, (radObjWP.Const.a_w + 0.5 * radObjWP.Const.b_w).reshape((-1, 1)), label='Pure Water [1/m]')
        #plt.plot(bands, a_arr, label='Estimated attenuation [1/m]')
        #plt.scatter(bands, r_arr, label='R2 score [-]')
        #plt.xlabel('Wavelength [nm]')
        #plt.ylabel('Attenuation [1/m]')
        #print('Mean r2 score Lophelia: ' + str(np.mean(r_arr[10:191])))
        #plt.legend()
        #plt.ylim([-0.2, 0.8])
        #plt.xlim([400, 700])
        #plt.pause(1000)
        ##plt.show()
        ## Take radiance value 1:
        #R2 = np.linalg.norm(radObjWP.Var.points2, axis=1).reshape((-1, 1))
        ## R2 = np.linalg.norm(radObj.Var.points2, axis=1).reshape((-1, 1))
        ##
        #k = bands.reshape(-1).shape[0]
        #a_arr = np.zeros(k)
        #for i in range(k):
        #    x = R2
        #    y = -np.log(radObjWP.Meas.rad2[:, i] / (radObjWP.Meas.rad2[:, 100]))
        #    b, a, r = linear_regressor(x=x, y=y)
#
        #    a_arr[i] = a/2
        #plt.plot(bands, a_arr - a_arr[150], label = 'Using Lophelia 2')
        #plt.legend()
        #plt.ylim([-0.2, 0.8])
        #plt.xlim([400, 700])
        ##plt.scatter(R1 - R2, np.log(radObj.Meas.rad1[:, 100] / radObj.Meas.rad1[:, 170]) -
        ##            np.log(radObj.Meas.rad2[:, 100] / radObj.Meas.rad2[:, 170]), alpha= 0.05)
        ##plt.scatter(np.linalg.norm(radObjWP.Var.points2, axis = 1).reshape((-1, 1)), np.log(radObjWP.Meas.rad2[:, 100]/radObjWP.Meas.rad2[:, 150]))
        #plt.pause(1000)
        #plt.show(block = False)


    if plot_it == True:


        mu = np.mean(radObjWP.refl_1, axis = 0)
        s = np.std(radObjWP.refl_1, axis = 0)
        from scipy.ndimage import gaussian_filter1d
        #plt.plot(bands, gaussian_filter1d(radObj.refl_1[127, :], sigma = 2))
        #plt.plot(bands, gaussian_filter1d(radObj.refl_2[127, :], sigma = 2))
        #plt.plot(bands, mu + s)
        plt.plot(bands, mu)
        #plt.plot(bands, mu - s)
        plt.plot(bands, refl_gt_coral.reshape((224,1)))
    ##
        mu = np.mean(radObjWP.refl_2, axis=0)
        s = np.std(radObjWP.refl_2, axis=0)
        #plt.plot(bands, mu + s)
        plt.plot(bands, mu)
        #plt.plot(bands, mu - s)
        plt.xlim([400, 700])
        plt.ylim([-1, 1])
        plt.show()

    print(np.median(np.abs(error1)))
    print(np.median(np.abs(error2)))
    print(np.median(error1))
    print(np.median(error2))
    print(np.mean(error1))
    print(np.mean(error2))

    #print(np.median(np.abs(error_norm1)))
    #print(np.median(np.abs(error_norm2)))

    print([G, X, Y, B, alpha, I_0, sigma_l, k_l])

    #err_tot = np.concatenate((error, error_norm1, error_norm2)).reshape(-1)

    #print(len(np.nanmedian(error, axis = 0)))



    return np.concatenate((error1, error2))


def optimize_function_WP_forward(radObj, G, X, Y, alpha, I_0, sigma_l):
    radObj.set_parameters(G, X, Y, alpha, I_0, sigma_l)

    radObj.run_forward_model(is_skogn=True, model = 'forward')

    error1 = np.log(np.abs(radObj.diff_dc1) + 1)
    error2 = np.log(np.abs(radObj.diff_dc2) + 1)

    print( np.median(np.abs(radObj.diff_dc1)) )

    print(np.median(np.abs(radObj.diff_dc2)))

    print([G, X, Y, alpha, I_0, sigma_l])

    return np.concatenate((error1.reshape(-1), error2.reshape(-1)))

def compute_variables_measurements(iniPath, pathH51, pathH52, radPicklePath):
    config = configparser.ConfigParser()
    config.read(iniPath)
    pathPcl1 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/3Dmodels/RGBCompositePointClouds/120820_2.ply'
    pathPcl2 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/3Dmodels/RGBCompositePointClouds/121343_2.ply'
    #



    hyp1 = Hyperspectral(filename=pathH51, config = config)
    hyp1.DN2Radiance(config)
    points_1 = hyp1.points_global.reshape((-1, 3))


    #plt.plot(hyp1.band2Wavelength, np.min(hyp1.dataCubeRadiance[:, pix, :], axis = 0), label =  'Pixel ' + str(pix) )
    #pix = 480
    #plt.plot(hyp1.band2Wavelength, np.min(hyp1.dataCubeRadiance[:, pix, :], axis=0), label='Pixel ' + str(pix))
    #pix = 959
    #plt.plot(hyp1.band2Wavelength, np.min(hyp1.dataCubeRadiance[:, pix, :], axis=0), label='Pixel ' + str(pix))
    #plt.legend()
    #plt.show()





    print('Loaded 1')
    hyp2 = Hyperspectral(filename=pathH52, config = config)
    hyp2.DN2Radiance(config)

    visualize.show_point_clouds(pathPcl1=pathPcl1, pathPcl2=pathPcl2, hyp1 = hyp1, hyp2 = hyp2)
    pix = 0
    band = 20
    print('Loaded 2')
    points_2 = hyp2.points_global.reshape((-1,3))
    #min_image1 = np.min(hyp1.dataCubeRadiance, axis=0).reshape((960, 224, 1))
    #min_image2 = np.min(hyp2.dataCubeRadiance, axis=0).reshape((960, 224, 1))
    #min_image = np.min(np.concatenate((min_image1, min_image2), axis = 2), axis = 2)
    #plt.imshow(min_image, aspect='auto')
    #plt.colorbar()
    #plt.pause(100)
    #plt.show()

    #hyp1.dataCubeRadiance -= min_image.reshape((1, 960, 224))
    #hyp2.dataCubeRadiance -= min_image.reshape((1, 960, 224))


    #min_image1 = np.min(hyp1.dataCubeRadiance, axis=0).reshape((960, 224, 1))
    #min_image2 = np.min(hyp2.dataCubeRadiance, axis=0).reshape((960, 224, 1))
    #min_image = np.min(np.concatenate((min_image1, min_image2), axis=2), axis=2)

    #plt.imshow(min_image, aspect='auto')
   # #plt.colorbar()
    #plt.pause(100)
    #plt.show()




    # Apply

    n = hyp2.dataCubeRadiance.shape[0]
    m = hyp2.dataCubeRadiance.shape[1]

    """Finding the neares neighbor for all data"""
    print('Nearest Neighbor interpolation')
    max_dist = 0.02
    tree = NearestNeighbors(radius=max_dist).fit(points_1)
    dist, indexes = tree.kneighbors(points_2, 1)

    # 1D indices for datacube 1
    indexes1 = indexes[dist < max_dist]
    w_datacube_1 = 960
    v_datacube_1 = (indexes1 % w_datacube_1).astype(np.uint64)
    u_datacube_1  = ((indexes1 - v_datacube_1) / w_datacube_1).astype(np.uint64)

    # 1D indices for datacube 2
    indexes2 = np.arange(len(indexes)).reshape((-1,1))[dist < max_dist]
    w_datacube_2 = 960
    v_datacube_2 = (indexes2 % w_datacube_2).astype(np.uint64)
    u_datacube_2  = ((indexes2 - v_datacube_2) / w_datacube_2).astype(np.uint64)



    # Disse skriver vi til masker.

    mask1 = np.zeros((hyp1.dataCubeRadiance.shape[0],hyp1.dataCubeRadiance.shape[1])).astype(np.float64)
    mask2 = np.zeros((hyp2.dataCubeRadiance.shape[0], hyp2.dataCubeRadiance.shape[1])).astype(np.float64)
    mask1[u_datacube_1, v_datacube_1] = 1
    mask2[u_datacube_2, v_datacube_2] = 1
    
    import matplotlib.pyplot as plt
    dir = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Matlab/mat_tautra_1203_1208'

    data = {
        'u_datacube_1': u_datacube_1,
        'u_datacube_2': u_datacube_2,
        'v_datacube_1': v_datacube_1,
        'v_datacube_2': v_datacube_2,
        'datacube_1_name': 'Tautra_20170307_120309_3',
        'datacube_2_name': 'Tautra_20170307_120820_1',

    }

    import scipy.io

    save_path = dir + '120309_3__120820_1.mat'


    scipy.io.savemat(save_path, data)

    # Masking does not work


    a = 7
    # Utilize a criterion rather than a gridding. Difference in distance is an obvious one

    #dist1 = np.linalg.norm(hyp1.points_local[u_datacube_1, v_datacube_1, :].reshape((u_datacube_1.shape[0], 3)), axis=1)
    #pix_1 = v_datacube_1.astype(np.uint16)
#
#
#
    #dist2 = np.linalg.norm(hyp2.points_local[u_datacube_2, v_datacube_2, :].reshape((u_datacube_2.shape[0], 3)), axis=1)
    #pix_2 = v_datacube_2.astype(np.uint16)
#
#
    #plt.hist(dist1, 100)
    #plt.show()
#
#
    #np.random.seed(10)
#
    ## Sample variable domain uniformly
    #data = dist1
    #hist, bin_edges = np.histogram(data, bins=960)
    #print(bin_edges.max())
    #print(bin_edges.min())
    #n_per_pixel = 10
    #samples = []
    #indices = []
    #for i in range(len(hist)):
    #    bin_data_mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
    #    bin_data_indices = np.where(bin_data_mask)[0]
    #    bin_data = data[bin_data_mask]
    #    if len(bin_data) > 0:
    #        for i in range(n_per_pixel):
    #            idx = np.random.choice(np.arange(len(bin_data)))
    #            samples.append(bin_data[idx])
    #            indices.append(bin_data_indices[idx])
    #samples_array_1 = np.array(samples)
    #ind_valid_arr_1 = np.array(indices)
#
#
#
#
#
#
    #data = pix_1
    #hist, bin_edges = np.histogram(data, bins=960)
    #samples = []
    #indices = []
    #for i in range(len(hist)):
    #    bin_data_mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
    #    bin_data_indices = np.where(bin_data_mask)[0]
    #    bin_data = data[bin_data_mask]
    #    if len(bin_data) > 0:
    #        for i in range(n_per_pixel):
    #            idx = np.random.choice(np.arange(len(bin_data)))
    #            samples.append(bin_data[idx])
    #            indices.append(bin_data_indices[idx])
    #samples_array_2 = np.array(samples)
    #ind_valid_arr_2 = np.array(indices)
#
    #ind_tot = np.concatenate((ind_valid_arr_1.reshape(-1), ind_valid_arr_2.reshape(-1)))
    #ind_tot = ind_valid_arr_2.reshape(-1)
#
    ##gray_scale = hyp1.dataCubeRadiance[:, :, 100]
    ##gray_scale[u_datacube_1, v_datacube_1] = -0.01
    ##plt.imshow(gray_scale)
    ##plt.show()
##
    ##gray_scale = hyp2.dataCubeRadiance[:, :, 100]
    ##gray_scale[u_datacube_2, v_datacube_2] = -0.01
    ##plt.imshow(gray_scale)
    ##plt.show()
#
    #save_rgb_composite(path = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/rgb_composite_1.png', datacube=hyp1.dataCubeRadiance)
    #save_rgb_composite(
    #    path='C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/rgb_composite_2.png',
    #    datacube=hyp2.dataCubeRadiance)





    #plt.imshow(hyp1.dataCubeRadiance[:, :, 100])
    #plt.pause(100)
    #plt.show()
    ind_tot = np.linspace(0, u_datacube_1.shape[0]-1, 10000).astype(np.uint)
    n = u_datacube_1[ind_tot].shape[0]
    k = hyp1.band2Wavelength.reshape(-1).shape[0]
    m = hyp1.dataCube.shape[1]

    angles_1 = np.arctan(hyp1.points_local[0, :, 0] / hyp1.points_local[0, :, 2]).reshape((m, 1))
    rad_1 = hyp1.dataCubeRadiance[u_datacube_1[ind_tot], v_datacube_1[ind_tot], :].reshape((n, k))
    points_local_1 = hyp1.points_local[u_datacube_1[ind_tot], v_datacube_1[ind_tot], :].reshape((n, 3))
    normals_local_1 = hyp1.normals_local[u_datacube_1[ind_tot], v_datacube_1[ind_tot], :].reshape((n, 3))
    pixel_1 = v_datacube_1[ind_tot].reshape((n, 1))
    bands_1 = hyp1.band2Wavelength.reshape((k, 1))

    n = u_datacube_2[ind_tot].shape[0]
    k = hyp2.band2Wavelength.reshape(-1).shape[0]
    m = hyp2.dataCube.shape[1]

    angles_2 = np.arctan(hyp2.points_local[0, :, 0] / hyp2.points_local[0, :, 2]).reshape((m, 1))
    rad_2 = hyp2.dataCubeRadiance[u_datacube_2[ind_tot], v_datacube_2[ind_tot], :].reshape((n, k))
    points_local_2 = hyp2.points_local[u_datacube_2[ind_tot], v_datacube_2[ind_tot], :].reshape((n, 3))
    normals_local_2 = hyp2.normals_local[u_datacube_2[ind_tot], v_datacube_2[ind_tot], :].reshape((n, 3))
    pixel_2 = v_datacube_2[ind_tot].reshape((n, 1))
    bands_2 = hyp2.band2Wavelength.reshape((k, 1))

    RadData = namedtuple('RadData', ['rad', 'points', 'normals', 'pixel_nr', 'bands', 'angles', 'RF', 't_exp'])
    RF1 = hyp1.radiometricFrame
    t_exp1 = hyp1.t_exp
    data1 = RadData(rad=rad_1, points=points_local_1, normals=normals_local_1, pixel_nr=pixel_1, bands=bands_1,
                    angles=angles_1, RF=RF1, t_exp=t_exp1)

    RF2 = hyp2.radiometricFrame
    t_exp2 = hyp2.t_exp
    data2 = RadData(rad=rad_2, points=points_local_2, normals=normals_local_2, pixel_nr=pixel_2, bands=bands_2,
                    angles=angles_2, RF=RF2, t_exp=t_exp2)

    corr_coef = np.corrcoef(rad_1.reshape(-1), rad_2.reshape(-1))[0, 1]

    print("Correlation coefficient between arr1 and arr2:", corr_coef)

    plt.scatter((rad_1/rad_1[:,100].reshape((-1,1))).reshape(-1),
                (rad_2/rad_2[:,100].reshape((-1,1))).reshape(-1), alpha = 0.005)
    plt.show()













    #from matplotlib.colors import LogNorm
    #plt.hist2d(dist1, dist2, bins=50, cmap=plt.cm.jet, norm=LogNorm())
    #plt.hist2d(pix_1, pix_2, 50)
    #cb = plt.colorbar()
    #plt.show()
    rad = Radiometry(config, data1=data1, data2=data2)
    #rad.DarkCount = hyp1.darkFrame
    #rad.t_exp_1 = hyp1.t_exp
    #rad.t_exp_2 = hyp2.t_exp

    del hyp2
    del hyp1

    import dill

    with open(radPicklePath, 'wb') as file_pi:
        dill.dump(rad, file_pi)


def compute_variables_measurements_white_plate(iniPath, pathH51, pathH52, radPicklePath):
    config = configparser.ConfigParser()
    config.read(iniPath)

    pathPcl1 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/3Dmodels/RGBCompositePointClouds/111150_1.ply'
    pathPcl2 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/3Dmodels/RGBCompositePointClouds/111440_1.ply'

    # visualize.show_point_clouds(pathPcl1=pathPcl1, pathPcl2=pathPcl2)

    hyp1 = Hyperspectral(filename=pathH51, config=config)
    hyp1.DN2Radiance(config)
    points_1 = hyp1.points_global.reshape((-1, 3))

    print('Loaded 1')
    hyp2 = Hyperspectral(filename=pathH52, config=config)
    hyp2.DN2Radiance(config)

    min_image1 = np.min(hyp1.dataCubeRadiance, axis=0).reshape((960, 225, 1))
    min_image2 = np.min(hyp2.dataCubeRadiance, axis=0).reshape((960, 225, 1))

    print('Loaded 2')
    points_2 = hyp2.points_global.reshape((-1, 3))

    n = hyp2.dataCubeRadiance.shape[0]
    m = hyp2.dataCubeRadiance.shape[1]

    # Read
    filename1 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/mask_output1.jpg'
    filename2 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/mask_output2.jpg'
    mask1 = image2mask(filename1)
    mask2 = image2mask(filename2)

    """Finding the neares neighbor for white plate"""

    print('Nearest Neighbor interpolation')
    tree = NearestNeighbors(radius=0.01).fit(points_1)
    dist, indexes = tree.kneighbors(points_2, 1)

    # 1D indices for datacube 1
    indexes1 = indexes[dist < 0.01]
    w_datacube_1 = 960
    v_datacube_1 = (indexes1 % w_datacube_1).astype(np.uint64)
    u_datacube_1 = ((indexes1 - v_datacube_1) / w_datacube_1).astype(np.uint64)

    # 1D indices for datacube 2
    indexes2 = np.arange(len(indexes)).reshape((-1, 1))[dist < 0.01]
    w_datacube_2 = 960
    v_datacube_2 = (indexes2 % w_datacube_2).astype(np.uint64)
    u_datacube_2 = ((indexes2 - v_datacube_2) / w_datacube_2).astype(np.uint64)

    RadData = namedtuple('RadData', ['rad', 'points', 'normals', 'pixel_nr', 'bands', 'angles', 'RF', 't_exp'])

    # Find values of v_datacube within mask
    ind = (mask1[u_datacube_1, v_datacube_1] == 255)
    n = u_datacube_1[ind].shape[0]
    k = hyp1.band2Wavelength.shape[0]
    m = hyp1.dataCube.shape[1]

    angles_1 = np.arctan(hyp1.points_local[0, :, 0] / hyp1.points_local[0, :, 2]).reshape((m, 1))
    rad_1 = hyp1.dataCubeRadiance[u_datacube_1[ind], v_datacube_1[ind], :].reshape((n, k))
    points_local_1 = hyp1.points_local[u_datacube_1[ind], v_datacube_1[ind], :].reshape((n, 3))
    normals_local_1 = hyp1.normals_local[u_datacube_1[ind], v_datacube_1[ind],].reshape((n, 3))
    pixel_1 = v_datacube_1[ind].reshape((n, 1))
    bands_1 = hyp1.band2Wavelength.reshape((k, 1))
    RF1 = hyp1.radiometricFrame
    t_exp1 = hyp1.t_exp

    data1 = RadData(rad=rad_1, points=points_local_1, normals=normals_local_1, pixel_nr=pixel_1, bands=bands_1,
                    angles=angles_1, RF = RF1, t_exp = t_exp1)



    print(n)
    n = u_datacube_2[ind].shape[0]
    k = hyp2.band2Wavelength.shape[0]
    m = hyp2.dataCube.shape[1]

    angles_2 = np.arctan(hyp2.points_local[0, :, 0] / hyp1.points_local[0, :, 2]).reshape((m, 1))
    rad_2 = hyp2.dataCubeRadiance[u_datacube_2[ind], v_datacube_2[ind], :].reshape((n, k))
    points_local_2 = hyp2.points_local[u_datacube_2[ind], v_datacube_2[ind], :].reshape((n, 3))
    normals_local_2 = hyp2.normals_local[u_datacube_2[ind], v_datacube_2[ind], :].reshape((n, 3))
    pixel_2 = v_datacube_2[ind].reshape((n, 1))
    bands_2 = hyp2.band2Wavelength.reshape((-1, 1))
    RF2 = hyp2.radiometricFrame
    t_exp2 = hyp2.t_exp


    data2 = RadData(rad=rad_2, points=points_local_2, normals=normals_local_2, pixel_nr=pixel_2, bands=bands_2,
                    angles=angles_2, RF = RF2, t_exp = t_exp2)

    rad = Radiometry(config, data1=data1, data2=data2)
    rad.DarkCount = hyp1.darkFrame
    rad.t_exp_1 = hyp1.t_exp
    rad.t_exp_2 = hyp2.t_exp

    del hyp2
    del hyp1

    import dill

    with open(radPicklePath, 'wb') as file_pi:
        dill.dump(rad, file_pi)

def compute_variables_measurements_white_coral(iniPath, pathH51, pathH52, radPicklePath):
    config = configparser.ConfigParser()
    config.read(iniPath)

    pathPcl1 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/3Dmodels/RGBCompositePointClouds/111150_1.ply'
    pathPcl2 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/3Dmodels/RGBCompositePointClouds/111440_1.ply'

    # visualize.show_point_clouds(pathPcl1=pathPcl1, pathPcl2=pathPcl2)

    hyp1 = Hyperspectral(filename=pathH51, config=config)
    hyp1.DN2Radiance(config)
    points_1 = hyp1.points_global.reshape((-1, 3))

    print('Loaded 1')
    hyp2 = Hyperspectral(filename=pathH52, config=config)
    hyp2.DN2Radiance(config)

    min_image1 = np.min(hyp1.dataCubeRadiance, axis=0).reshape((960, hyp1.dataCubeRadiance.shape[2], 1))
    min_image2 = np.min(hyp2.dataCubeRadiance, axis=0).reshape((960, hyp1.dataCubeRadiance.shape[2], 1))
    min_image = np.min(np.concatenate((min_image1, min_image2), axis=2), axis=2)

    #hyp1.dataCubeRadiance -= min_image.reshape((1, 960, hyp1.dataCubeRadiance.shape[2]))
    #hyp2.dataCubeRadiance -= min_image.reshape((1, 960, hyp1.dataCubeRadiance.shape[2]))

    hyp1.dataCubeRadiance /= 1000
    hyp2.dataCubeRadiance /= 1000



    print('Loaded 2')
    points_2 = hyp2.points_global.reshape((-1, 3))

    n = hyp2.dataCubeRadiance.shape[0]
    m = hyp2.dataCubeRadiance.shape[1]

    # Read
    filename1 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/mask_composite_1.jpg'
    filename2 = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/mask_composite_2.jpg'
    mask1 = image2mask(filename1)
    mask2 = image2mask(filename2)
    rows1, cols1 = np.where(mask1 == 0)
    rows2, cols2 = np.where(mask2 == 0)


    RadData = namedtuple('RadData', ['rad', 'points', 'normals', 'pixel_nr', 'bands', 'angles', 'RF', 't_exp'])

    # Find values of v_datacube within mask
    n = rows1.shape[0]
    k = hyp1.band2Wavelength.shape[1]
    m = hyp1.dataCube.shape[1]

    angles_1 = np.arctan(hyp1.points_local[0, :, 0] / hyp1.points_local[0, :, 2]).reshape((m, 1))
    rad_1 = hyp1.dataCubeRadiance[mask1 == 0, :].reshape((n, k))
    points_local_1 = hyp1.points_local[mask1 == 0, :].reshape((n, 3))
    normals_local_1 = hyp1.normals_local[mask1 == 0,].reshape((n, 3))
    pixel_1 = cols1.reshape((n, 1))
    bands_1 = hyp1.band2Wavelength.reshape((k, 1))
    RF1 = hyp1.radiometricFrame
    t_exp1 = hyp1.t_exp

    data1 = RadData(rad=rad_1, points=points_local_1, normals=normals_local_1, pixel_nr=pixel_1, bands=bands_1,
                    angles=angles_1, RF = RF1, t_exp = t_exp1)



    print(n)
    n = rows2.shape[0]
    k = hyp2.band2Wavelength.shape[1]
    m = hyp2.dataCube.shape[1]

    angles_2 = np.arctan(hyp2.points_local[0, :, 0] / hyp1.points_local[0, :, 2]).reshape((m, 1))
    rad_2 = hyp2.dataCubeRadiance[mask2 == 0, :].reshape((n, k))
    points_local_2 = hyp2.points_local[mask2 == 0, :].reshape((n, 3))
    normals_local_2 = hyp2.normals_local[mask2 == 0, :].reshape((n, 3))
    pixel_2 = cols2.reshape((n, 1))
    bands_2 = hyp2.band2Wavelength.reshape((-1, 1))
    RF2 = hyp2.radiometricFrame
    t_exp2 = hyp2.t_exp


    data2 = RadData(rad=rad_2, points=points_local_2, normals=normals_local_2, pixel_nr=pixel_2, bands=bands_2,
                    angles=angles_2, RF = RF2, t_exp = t_exp2)

    rad = Radiometry(config, data1=data1, data2=data2)
    rad.DarkCount = hyp1.darkFrame
    rad.t_exp_1 = hyp1.t_exp
    rad.t_exp_2 = hyp2.t_exp

    del hyp2
    del hyp1

    import dill

    with open(radPicklePath, 'wb') as file_pi:
        dill.dump(rad, file_pi)



iniPathTautra = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/configuration.ini'
pathH51 = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_120820_2.hdf'
pathH52 = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_121343_2.hdf'

#pathH51 = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_131705_4.hdf'
#pathH52 = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_132330_1.hdf'
#
#pathH51 = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_120309_3.hdf'
#pathH52 = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_120820_1.hdf'
#pathH51 = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_120820_1.hdf'
#pathH52 = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_120309_4.hdf'
radPicklePathTautra = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/Pickle/Radiometry.pkl'
radPicklePathTautraWP ='C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/Pickle/RadiometryWhiteCoral.pkl'

iniPathSkogn = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/configuration.ini'
pathH51_skogn = 'E:/Skogn/Skogn_h5/uhi_20210122_111150_1.h5'
pathH52_skogn = 'E:/Skogn/Skogn_h5/uhi_20210122_111440_1.h5'

radPicklePathSkogn = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/Pickle/Radiometry.pkl'
radPicklePathSkognWP = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/Pickle/RadiometryWhitePlate.pkl'

# Create data structure
path = 'E:/TautraUHI/ROV_UHI/Tautra_20170307_120820_1.hdf'
config = configparser.ConfigParser()
config.read(iniPathTautra)
#config.read(iniPathSkogn)
hyp1 = Hyperspectral(filename=pathH51, config = config)
hyp1.DN2Radiance(config)
#
sp.imshow(hyp1.dataCubeRadiance, bands =(150, 97, 50))
plt.savefig('C:/Users/haavasl/Downloads/image.png', dpi=600)
plt.pause(100)
##
#
#from pylab import *
#from scipy.optimize import curve_fit
#
#band = 100
#startidx = 480
#y = hyp1.dataCubeRadiance[:, startidx, 40]
##u = np.arange(startidx, stopidx)
#import dill
#with open(radPicklePathSkogn, 'rb') as f:
#    radObj = dill.load(f)
#
#ang = radObj.Var.angles1
#cos_norm = np.abs(hyp1.normals_local[:,:,2])
#cos_norm_med = np.median(cos_norm, axis = 0)
##y *= 1/cos_norm_med # Cosine compensation
##x = (ang[u]*180/np.pi).reshape(-1)
#x = hyp1.points_local[:,startidx, 2]
#print(x.shape)
#print(y.shape)
#
#plt.scatter(x, y)
#plt.show()
#
##x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
#
#def gauss(x,mu,sigma,A):
#    return A*exp(-(x-mu)**2/2/sigma**2)
#
#def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
#    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
#
#expected=(0, 17, 0.5)
#params,cov=curve_fit(gauss, x, y, expected)
#sigma=sqrt(diag(cov))
#print(params,'/n',sigma)
#plot(x, gauss(x,*params),color='red',lw=3,label= 'Gaussian with $/sigma$=' + "%.0f" %params[1] + '$^{/circ}$')
#plt.plot(x, y, label =  "%.0f" % hyp1.band2Wavelength[band] + ' nm')
#plt.xlabel('View angle [$^{/circ}$]')
#plt.ylabel('Relative radiance [-]')
#legend()
#plt.pause(1000)
#
#
#
##plt.plot(np.arange(960), np.median(hyp1.dataCubeRadiance[:, :, 0], axis = 0)/np.max(np.median(hyp1.dataCubeRadiance[:, :, 0], axis = 0)), label = '0')
##plt.plot(np.arange(960), np.median(hyp1.dataCubeRadiance[:, :, 50]/np.max(np.median(hyp1.dataCubeRadiance[:, :, 50], axis = 0)), axis = 0), label = '50')
##plt.plot(np.arange(960), np.median(hyp1.dataCubeRadiance[:, :, 100]/np.max(np.median(hyp1.dataCubeRadiance[:, :, 100], axis = 0)), axis = 0), label = '100')
##plt.plot(np.arange(960), np.median(hyp1.dataCubeRadiance[:, :, 150]/np.max(np.median(hyp1.dataCubeRadiance[:, :, 150], axis = 0)), axis = 0), label = '150')
##plt.plot(np.arange(960), np.median(hyp1.dataCubeRadiance[:, :, 200]/np.max(np.median(hyp1.dataCubeRadiance[:, :, 200], axis = 0)), axis = 0), label = '200')
##plt.legend()
##plt.pause(1000)
compute_variables_measurements(iniPath = iniPathTautra, pathH51=pathH51, pathH52=pathH52, radPicklePath = radPicklePathTautra)
##compute_variables_measurements(iniPath = iniPathSkogn, pathH51=pathH51_skogn, pathH52=pathH52_skogn, radPicklePath = radPicklePathSkogn)
##compute_variables_measurements_white_plate(iniPath = iniPathSkogn, pathH51=pathH51_skogn, pathH52=pathH52_skogn, radPicklePath = radPicklePathSkognWP)
##compute_variables_measurements_white_coral(iniPath = iniPathTautra, pathH51=pathH51, pathH52=pathH52, radPicklePath = radPicklePathTautraWP)
## Try plotting something



import dill


with open(radPicklePathSkogn, 'rb') as f:
    radObj = dill.load(f)

with open(radPicklePathTautraWP, 'rb') as f:
    radObjWP = dill.load(f)































from lmfit import Model


# Try and optimize for the white plate
mod = Model(optimize_function_WP)
pars = mod.make_params(G = {'value': 0.1, 'min': 0, 'max': 0.8}, X = {'value': 0.1, 'min': 0, 'max':4}, Y = {'value': 1.0, 'min': 0, 'max': 2.5}, alpha = {'value': 0, 'min': -0.1, 'max': 0.1}, I_0 = {'value': 1.0, 'min': 10e-9}, sigma_l = {'value': 0.3, 'min': 0.1, 'max': 1})

#pars = mod.make_params(G = {'value': 0.06751024759978295, 'min': 0, 'max': 0.8}, X = {'value': 2.4587210892842823, 'min': 0, 'max': 4}, Y = {'value': 6.128846694042345e-08, 'min': 0, 'max': 2.5}, alpha = {'value': -0.09237675342363894, 'min': -0.1, 'max': 0.1}, I_0 = {'value': 0.8125099355576991, 'min': 0.001}, sigma_l = {'value': 0.28309351049442016, 'min': 0.25, 'max': 0.35})

#pars = mod.make_params(G = {'value': 0.06751024759978295, 'min': 0, 'max': 0.8}, X = {'value': 2.4587210892842823, 'min': 0, 'max': 4}, Y = {'value': 6.128846694042345e-08, 'min': 0, 'max': 2.5}, alpha = {'value': -0.09237675342363894, 'min': -0.1, 'max': 0.1}, I_0 = {'value': 0.8125099355576991, 'min': 0.001}, sigma_l = {'value': 0.28309351049442016, 'min': 0.25, 'max': 0.35})

#radObjList = [radObj, radObjWP]

# For Skogn plate, for Tautra Lophelia
radObj.WhitePlateObj = radObjWP
y = np.zeros(1956375)
# 8.952127927841503e-12, 2.1998122630011183, 6.128846694042345e-08, -0.02498735037403657, 0.3676992313342894, 0.3189460251301856
y = np.zeros(434750)
#result = mod.fit(y, pars, x = None, radObjList = radObj)


param0 = np.array([8.952127927841503e-12, 2.1998122630011183, 6.128846694042345e-08, -0.02498735037403657, 0.3676992313342894, 0.3189460251301856])

# For Tautra
bounds = ([0.01, 0.00001,   0,   0.0001, -0.1, 0.0002893740434756385-10e-9, 0.42, -0.0004211428088137218- 10e-10],
          [   1,    1, 2.5,         0.1,     0.1, 0.0002893740434756385 + 10e-9, 0.45,  -0.0004211428088137218 + 10e-10])


#param0 = np.array([0.010000000025911933, 0.010000000001549297, 1.0832315576606701, 0.0001000000000026569, 0.0999999999378357, 6306211339.961953, 0.38028705111554584])
param150 = np.array([0.24814819994033516, 0.04050087923130993, 0.613601775506194, 0.09950049999997332, 0.0009999999999999998, 5385959787.638721, 0.41683328397878794, 0.28236223526826254, -0.017526328697549144])

param175 = np.array([0.02881653540218901, 0.032154413827011334, 2.2898877707328884, 0.09950051490116119, 0.0009999999999999998, 4537705557.9405575, 0.4912645193472945, 0.3086163681991014, -0.0008903047457070495])

param25_175 = np.array([0.010000000000000002, 0.010000000188352181, 0.6905054185347994, 0.09950049999999887, 0.0009999999999999998, 5504179968.749561, 0.4011185463694464, 0.30731214475070306, -0.0009999850714969517])

param0 = np.array([0.1, 0.1, 1.25, 0.05, 0, 10000000000, 0.3, 0])

param0NoBackscatter = np.array([0.08488915096028826, 0.04104581291379647, 1.17259845183904, 0.0005995000000000028, 0.0009999999009490347, 0.0003733011302949693, 0.4297444574564481, -0.00087912460668984])

# Essentially fixed light source intensity
param0NoBackscatter = np.array([0.04418578411743638, 0.0006477041538701926, 1.8489297407148666, 0.0005995000000000012, 0.0009999999999999593, 0.0002893740434756385, 0.4439086413363198, -0.0004211428088137218])

param0NoBackscatter = np.array([0.04418578411743638, 0.0006477041538701926, 1.8489297407148666, 0.0005995000000000012, 0.0009999999999999593, 0.0002893740434756385, 0.4439086413363198, -0.0004211428088137218])

param0 = np.array([0.1, 0.1, 1.25, 0.05, 0, 0.00037139755851829133, 0.3, 0])
from scipy.optimize import least_squares


# Results from calibration plate

#param0NoBackscatter = np.array([0.019271216190914348, 0.029402956109935777, 1.8489297407148666, 0.0024026482388451323, 0, 0.000375, 0.3, -0.0004211418088414382])
#param0NoBackscatter = np.array([0.019271216190914348, 0.029402956109935777, 1.8489297407148666, 0.0024026482388451323, 0, 0.000375, 0.4, -0.0004211418088414382])


bounds = ([0, 0.00001,   0,   0.0001, -0.1, param0NoBackscatter[5] - 10e-9, param0NoBackscatter[6] - 10e-9, param0NoBackscatter[7] - 10e-2],
          [   1,    0.1, 2.5,         0.1,     0.1, param0NoBackscatter[5] + 10e-9, param0NoBackscatter[6] + 10e-9,  param0NoBackscatter[7] + 10e-2])
param0 = np.array([0.05, 0.1, 1.25, 0.05, 0, 0.000371, 0.42, -0.0004211428088137218 ])
param0_low_median=np.array([0.017067907155297223, 0.026375918498475467, 1.8489297407148666, 0.0024026482388451323, -0.007211252841370041, 0.00037479278517702034, 0.4489142733738463, -0.0032797332407744424])
param0 = np.array([0.015192139811619662, 0.02476192092992111, 1.8489297407148666, 0.0024026482388451323, -0.0035120881879565136, 0.000374335325024145, 0.4479263251303151, -0.0027323728901613607])

#param0NoBackscatter = np.array([-0.2810709334303472, 0.32568671272926286, 0.7724243882025822, 0.0001645759844978038, -0.09998921221803489, 9.079535883615591e-05, 0.6488366624806513, 0.019202837659108848])
plot_it = False
#res = least_squares(fun = optimize_function_lsq_tautra, x0 = param0NoBackscatter, args= (radObj, plot_it,) , x_scale='jac', bounds = bounds)#,
param0NoBackscatter = np.array([0.019271216190914348, 0.029402956109935777, 1.8489297407148666, 0.0024026482388451323, 0, 0.000375, 0.4439086413363198, -0.0004211418088414382])
param0NoBackscatter[6] = 0.45
optimize_function_lsq_tautra(param0NoBackscatter, radObj, plot_it=False, is_skogn=False)
#res = least_squares(fun = optimize_function_lsq_tautra, x0 = res.x, args= (radObj,plot_it,) , x_scale='jac', bounds = bounds)#,
#dist1 = np.linalg.norm(radObj.Var.points1, axis = 1)

#dist2 = np.linalg.norm(radObj.Var.points2, axis = 1)
#
#rad1_170 = radObj.Meas.rad1[:, 170]/radObj.Meas.rad1[:, 50]
#rad2_170 = radObj.Meas.rad2[:, 170]/radObj.Meas.rad2[:, 50]













#plt.scatter(dist1 - dist2, np.log(rad1_170) - np.log(rad2_170), alpha= 0.01)
#plt.show()
#
#plt.scatter(dist1, np.log(rad1_170), alpha= 0.05)
#plt.show()

# Nå kan du laste inn objektet og studere













#plotCloudsGeometry = False
#if plotCloudsGeometry:
#    p = BackgroundPlotter(window_size=(600, 400))
#    pcd = o3d.io.read_point_cloud(pathPcl1)
#    color_arr = np.asarray(pcd.colors)
#    hyp_pcl = pv.read(pathPcl1)
#    #hyp_pcl['points'] = points_1[indexes[dist < 0.01]]
#
#
#    point_cloud1 = pv.PolyData(points_1[indexes1])
#    p.add_mesh(point_cloud1, point_size=2, color='red')
#
#    pcd = o3d.io.read_point_cloud(pathPcl2)
#    color_arr = np.asarray(pcd.colors)
#    hyp_pcl2 = pv.read(pathPcl2)
#    #hyp_pcl2['points'] = points_2
#    point_cloud2 = pv.PolyData(points_2[indexes2])
#    p.add_mesh(point_cloud2,  point_size=2, color='blue')
#
#    p.show()
#    p.app.exec_()
#
#print()






