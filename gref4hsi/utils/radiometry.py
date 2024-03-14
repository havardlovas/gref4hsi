



from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import expi
from scipy.special import erf
from scipy.special import erfi
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def fresnel(theta_i, n_i, n_t, is_transmittance, method):
    # This is used for computing the transmittance at the interfaces
    # First Snells:
    if method == 1:
        theta_t = np.arcsin((n_i / n_t)*np.sin(theta_i))

        c_i = np.cos(theta_i)
        c_t = np.cos(theta_t)

        R_s = np.abs((n_i*c_i - n_t*c_t)/(n_i*c_i + n_t*c_t))**2

        R_p = np.abs((n_i*c_t - n_t*c_i)/(n_i*c_t + n_t*c_i))**2

        # Then snells:
        R_F = 0.5*(R_s + R_p)


    elif method == 2:
        theta_t = np.arcsin((n_i / n_t) * np.sin(theta_i))

        r1 = (np.sin(theta_i - theta_t)/np.sin(theta_i + theta_t))**2
        r2 = (np.tan(theta_i - theta_t) / np.tan(theta_i + theta_t)) ** 2

        R_F = 0.5 * (r1 + r2)
    if is_transmittance:
        return 1 - R_F
    else:
        return R_F




def solid_angle_ratio(theta_w, n_w):
    """Computes the in-air angle corresponding to an in-water angle using Snell's law"""
    n_a = 1 # Approximate n_a = 1
    theta_a = np.arcsin((n_w / n_a) * np.sin(theta_w)) # Snells law
    return theta_a

def immersion_factor(theta_w, n_w, n_g, pixel_nr):
    """Calculates the immersion factor for imager with planar glass port in underwater hyperspectral imaging"""
    n_a = 1

    theta_a = np.arcsin((n_w / n_a) * np.sin(theta_w))

    t_ag = fresnel(theta_i=theta_a, n_i = n_a, n_t = n_g, is_transmittance=1, method=1)

    t_wg = fresnel(theta_i=theta_w, n_i=n_w, n_t=n_g, is_transmittance=1, method=1)

    I_f_LU = ((n_w**2)/(n_a**2) ) * (t_ag/t_wg)

    I_f = I_f_LU[pixel_nr.reshape(-1), :]

    return I_f



def radiance_conversion(RF, t_exp, pixel_nr):

    rad_2_DN_LU = RF*t_exp

    rad_2_DN = rad_2_DN_LU[pixel_nr.reshape(-1)] # Evaluate for relevant pixels
    return rad_2_DN



def beam_pattern(dir_s0, dir, sigma, I_0, I_hat):
    n = dir.shape[0]
    k = I_hat.shape[0]
    # A gaussian beam pattern
    dot_prod = np.dot(dir_s0, np.transpose(dir))

    len_vec = np.linalg.norm(dir_s0)*np.linalg.norm(dir, axis = 1)

    theta_si = np.arccos(dot_prod/len_vec)



    z = (theta_si)/sigma

    Intensity = I_0*np.exp(-0.5*z**2)

    Intensity_Spectral = np.multiply(Intensity.reshape((-1, 1)), I_hat.reshape((1, -1)))

    return Intensity/I_0, Intensity_Spectral

def compute_gamma(dir, normals):
    """Compute the angle between a normal vector and an incoming ray"""
    dot_prod = np.einsum('ij, ij -> i', normals, -dir)
    len_vec = np.linalg.norm(dir, axis = 1) * np.linalg.norm(normals, axis = 1)
    gamma = np.arccos(dot_prod/len_vec)
    return gamma.reshape((-1, 1))

def compute_n_g(wlen):
    n_g = 1.4424 + (7.1661/(wlen - 144.170))
    return n_g

def compute_n_w(wlen, method):
    if method == 1:
        n_w = 1.325147 + (6.6096/(wlen - 137.1924))
    elif method == 2:
        T = 10
        a = -0.000001501562500
        b = 0.000000107084865
        c = -0.000042759374989
        d = -0.000160475520686
        e = 1.398067112092424
        # Function of Slocum

        n_w = a * T ** 2 + b * wlen ** 2 + c * T + d * wlen + e

    return n_w

def compute_scattering_integral(k, x):

    Ei = expi(x = -2*x*k)

    integral = -(2*k*x*Ei + np.exp(-k*x)) / x

    return integral

def compute_backscatter(B_p, b, b_w, b_p, d_pix, pos_seabed, p_si, dir_s0, sigma_l, I_0, I_hat, k, pixel_nr):
    # Step 1 create a lookup table for pixelnumber and perp distance
    z_res = 0.05

    z_max = pos_seabed[:, 2].max()

    # Partition
    z = (np.arange(0, np.ceil(z_max/z_res) + 1)*z_res).reshape((-1, 1, 1))

    r_si_b = z * d_pix - p_si # t by m by 3

    l_si_b = np.linalg.norm(r_si_b, axis =2).reshape((r_si_b.shape[0], r_si_b.shape[1], 1)) # t by m by 1 or something

    r_b_h = -z*d_pix # t by m by 3

    l_b_h = np.linalg.norm(r_b_h, axis=2).reshape((r_si_b.shape[0], r_si_b.shape[1], 1))  # t by m by 1 or something

    BP_norm, BP = beam_pattern(dir_s0=dir_s0, dir=r_si_b.reshape((-1, 3)), sigma=sigma_l, I_0=I_0, I_hat=I_hat).reshape((r_si_b.shape[0], r_si_b.shape[1], I_hat.shape[0]))

    E_i = BP*(1/(l_si_b**2))*np.exp(-k*l_si_b) # Values of incident irradiance for all directions and all perpendicular


    # distances t by m by k. Part of the integrand

    # Compute psi for all t by m cells. Check that angles are on the right side of 90 deg

    dot_prod = np.einsum('jk, ijk -> ij', d_pix, r_si_b)
    len_vec = np.linalg.norm(d_pix, axis=1) * np.linalg.norm(r_si_b, axis=2)
    psi = np.pi - np.arccos(dot_prod / len_vec)

    beta_p_interp = fournier_forand(B_p)  # Assumed constant backscatter fraction across wl. Is a log-log interpolator

    beta_p = np.exp( beta_p_interp(np.log(psi)) ).reshape((r_si_b.shape[0], r_si_b.shape[1], 1)) # Due to the interpolation in log-log domain. t by m

    beta_w = compute_beta_w(psi).reshape((r_si_b.shape[0], r_si_b.shape[1], 1)) # t by m

    beta_tot = (1/b)*(b_p*beta_p + b_w*beta_w) # t by m by k

    integrand = E_i*beta_tot*np.exp(-k*l_b_h) # t by m by k

    integral = np.cumsum(integrand, axis = 0)*z_res # Summing along the z axis weighted by z_res. Should return a t by m by k.



    L_b_LU = b*integral # Should return a t by m by k.

    # Can compute row of a given measurement by:
    rows = np.round(pos_seabed[:, 2]/z_res).astype(np.int64).reshape(-1)
    cols = pixel_nr.reshape(-1)

    L_b = L_b_LU[rows, cols, :] # Returns the appropriate spectra n by k

    #print(stop - start)

    return L_b









def compute_light_source_integral(sigma):
    """Computes the relation between spectral radiant flux P [W] and peak intensity I_0 [W/sr] for a directionally gaussian beam
    with sigma standard decviation"""
    constant = np.exp(-0.5 * sigma ** 2) * sigma

    term1 = -0.626657j * erf(0 + 0.707107j * sigma)
    term2 = 0.626657j * erf(1.11072 / sigma + 0.707107j * sigma)
    term3 = -0.626657 * erfi((0.707107 * (sigma ** 2 + 1.5708j)) / sigma)
    term4 = 0.626657 * erfi(0.707107 * sigma + 0j)

    return np.real(constant * (term1 + term2 + term3 + term4))*2*np.pi

def fournier_forand(B_p):
    """Calculates the phase function values for a given backscatter fraction"""

    # Partition data into equal 100 bins from 10 degrees to 100:
    logPsi = np.linspace(np.log(10*np.pi/180), np.log(180*np.pi/180), 100)
    psi = np.exp(logPsi)
    # Convert B_p to FF parameters
    mu, n = compute_FF_params(B_p)
    #
    nu = (3 - mu) / 2

    delta = (4 * (np.sin(psi / 2)) ** 2) / (3 * (n - 1) ** 2) # Correct, Mobley 2002 eq 2

    delta_180 = 4 / (3 * (n - 1) ** 2)

    beta_ff_t1 = 1 / (4 * np.pi * ((1 - delta) ** 2) * (delta ** nu))  # Coeff 1, correct

    beta_ff_t2 = nu * (1 - delta) - (1 - delta ** nu) # Correct

    beta_ff_t3 = delta * (1 - delta ** nu) - nu * (1 - delta) # *Correct

    beta_ff_t4 = 1 / (np.sin(psi / 2) ** 2) # Correct

    beta_ff_t5 = (1 - delta_180 ** nu) / (16 * np.pi * (delta_180 - 1) * delta_180 ** nu)

    beta_ff_t6 = 3 * np.cos(psi) ** 2 - 1

    beta_ff = beta_ff_t1 * (beta_ff_t2 + beta_ff_t3 * beta_ff_t4) \
              + beta_ff_t5 * beta_ff_t6  ## Formula t1*(t2+t3*t4) + t5*t6

    # Instantiate an interpolation object that describes the function
    beta_ff_interp = interp1d(x = logPsi, y = np.log(beta_ff), kind = 'cubic', fill_value='extrapolate')

    return beta_ff_interp

def compute_FF_params(B_p):
    mu_0 = 4 # Midpoint between max and min allowed values
    mu = fsolve(eq_backscatter_param, mu_0, args = (B_p,))
    n = 1.01 + 0.1542*(mu-3) # Eq 621, p 202 ocean optics
    return mu, n

    #

def eq_backscatter_param(mu, B_p_true):
    n = 1.01 + 0.1542*(mu-3) # Eq 621, p 202 ocean optics
    nu = (3-mu)/2 # eq 2 mobley 2002

    psi_90 = np.pi/2

    delta_90 = (4/( 3*(n-1)**2 ))*np.sin(psi_90/2)**2  # eq 2 mobley 2002

    num = (1 - delta_90**(nu+1) - 0.5*(1 - delta_90**nu))
    den = (1 - delta_90)*delta_90**nu

    B_p = 1 - num/den

    return B_p_true - B_p



def compute_beta_w(psi):
    """The scattering phase function of pure water"""
    # Psi is an array of scattering angles
    # Analytical Formula from Ocean Optics p. 180
    return 0.0608*(1 + 0.925*(np.cos(psi))**2)




class Radiometry():
    def __init__(self, config, data1, data2):
        # Read in necessary information. i.e. water absorption and water scattering coefs
        path_iop_water = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Misc/iop_pure_water.txt'
        df_light = pd.read_csv('C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Misc/DataMultiSeaLite.txt',
                         header=None)

        #df_light = pd.read_csv(
        #    'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Misc/Multi_SeaLite_spectrum.csv',
        #    header=0)

        df_iop = pd.read_csv(path_iop_water, header=9, sep=',') # Should be interpolated

        df_coral = pd.read_csv(
            'C:/Users\haavasl\PycharmProjects/newGit\TautraReflectanceTools\Misc\Coral_spectra_lab.csv', sep=';')

        df_refl_plaque = pd.read_csv('C:/Users\haavasl\PycharmProjects/newGit\TautraReflectanceTools\Misc/reflectance_plaque.csv')

        interpol_coral = interp1d(df_coral.iloc[:, 0].astype(np.float64),
                                np.transpose(df_coral.iloc[:, 2:4].astype(np.float64)), fill_value='extrapolate')
        interpol_iop = interp1d(df_iop.iloc[:, 0].astype(np.float64), np.transpose(df_iop.iloc[:, 1:3].astype(np.float64)))
        interpol_light = interp1d(df_light.iloc[:, 0], np.transpose(df_light.iloc[:, 1]), fill_value='extrapolate')

        interpol_refl = interp1d(df_refl_plaque.iloc[:, 0], np.transpose(df_refl_plaque.iloc[:, 1]))

        refl_gt = interpol_refl(data1.bands)

        I_hat = interpol_light(data1.bands) # Total spectral radiant flux
        a_w, b_w = interpol_iop(data1.bands)
        s1, s2 = interpol_coral(data1.bands)
        refl_gt_coral = 0.5*(s1 + s2) # Mean of measurements

        n_w = compute_n_w(wlen = data1.bands, method = 2)
        n_g = compute_n_g(wlen = data1.bands)

        # data contains RadData(rad = rad_2, points = points_local_2, normals=normals_local_2, pixel_nr=pixel_2, bands=bands_2, angles)

        Var = namedtuple('Var', ['points1', 'normals1', 'pixel_nr1', 'bands1', 'angles1','RF1','t_exp1', 'points2', 'normals2', 'pixel_nr2', 'bands2', 'angles2', 'RF2', 't_exp2'])

        self.Var =  Var(points1 = data1.points, normals1=data1.normals, pixel_nr1=data1.pixel_nr, bands1=data1.bands, angles1=data1.angles, RF1 = data1.RF, t_exp1 = data1.t_exp,
                         points2 = data2.points, normals2=data2.normals, pixel_nr2=data2.pixel_nr, bands2=data2.bands, angles2=data2.angles, RF2 = data2.RF, t_exp2 = data2.t_exp)


        # Structure the data in 1) constants, 2) measurements y, 3) variables x and 4) parameters beta
        Meas = namedtuple('Meas', ['rad1', 'rad2'])
        self.Meas = Meas(rad1 = data1.rad, rad2=data2.rad)



        Const = namedtuple('Const', ['S', 'a_w', 'b_w', 'n_g', 'n_w', 'dir_s1', 'dir_s2', 'pos_s1', 'pos_s2', 'I_hat', 'refl_gt', 'refl_gt_coral'])
        self.Const = Const(S = 0.015, a_w = a_w, b_w = b_w, n_g = n_g, n_w=n_w, dir_s1=np.array([0,0,1]), dir_s2=np.array([0,0,1]),
                           pos_s1 = np.array([0.35, 0, 0]), pos_s2 = np.array([-0.35, 0, 0]), I_hat = I_hat, refl_gt= refl_gt, refl_gt_coral = refl_gt_coral)


        Param = namedtuple('Param', ['G', 'X', 'Y', 'B_p', 'alpha', 'I_0', 'sigma_l', 'k_l'])
        self.Param = Param(G = 0.1, X = 0.1, Y = 1, B_p=0.01, alpha = 0, I_0 = 1, sigma_l = 1, k_l = 0)
    def run_forward_model(self, is_skogn, model = 'forward'):
        #print('This function is yet to be made')
        if model == 'two way':
            self.refl_1 = self.two_way_model_simple(id = 1, is_skogn = is_skogn)
            self.refl_2 = self.two_way_model_simple(id = 2, is_skogn = is_skogn)
        if model == 'forward':
            self.diff_dc1 = self.forward_model_simple(id=1, is_skogn=is_skogn)
            self.diff_dc2 = self.forward_model_simple(id=2, is_skogn=is_skogn)



            # Could also propagate the uncertainty into the model by perturbing


    def set_parameters(self, G, X, Y, B_p, alpha, I_0, sigma_l, k_l):
        #self.Param = Param(G=0.1, X=0.1, Y=1, alpha=0, I_0=1, sigma_l=1)
        Param = namedtuple('Param', ['G', 'X', 'Y', 'B_p', 'alpha', 'I_0', 'sigma_l', 'k_l'])
        self.Param = Param(G=G, X=X, Y=Y, B_p = B_p, alpha=alpha, I_0=I_0, sigma_l=sigma_l, k_l = k_l)





    def two_way_model_simple(self, id, is_skogn):
        # The formward model computes
        # Parameters
        G = self.Param.G
        X = self.Param.X
        Y = self.Param.Y
        B_p = self.Param.B_p


        I_0 = self.Param.I_0
        sigma_l = self.Param.sigma_l
        alpha = self.Param.alpha
        #B_inf1 = self.Param.B_inf
        #B_inf2 = self.Param.B_inf
        #p1  = self.Param.p1
        #p2 = self.Param.p2
        #theta_s1 = self.Param.theta_s1


        if id == 1:
            # Measurements
            L_air = self.Meas.rad1
            # Variables
            wlen = self.Var.bands1
            pos_seabed = self.Var.points1
            norm_vec = self.Var.normals1
            theta_w_dot = self.Var.angles1
            pixel_nr = self.Var.pixel_nr1
        elif id == 2:
            # Measurements
            L_air = self.Meas.rad2
            # Variables
            wlen = self.Var.bands2
            pos_seabed = self.Var.points2
            norm_vec = self.Var.normals2
            theta_w_dot = self.Var.angles2
            pixel_nr = self.Var.pixel_nr2


        # Constants
        a_ph = 0
        a_w = self.Const.a_w
        b_w = self.Const.b_w
        S = self.Const.S
        dir_s10 = self.Const.dir_s1
        dir_s20 = self.Const.dir_s2
        pos_s1 = self.Const.pos_s1
        pos_s2 = self.Const.pos_s2
        n_w = self.Const.n_w.reshape((1, -1))
        n_g = self.Const.n_g.reshape((1, -1))


        I_hat_non_scaled = self.Const.I_hat


        k_scale = self.Param.k_l

        I_hat = I_hat_non_scaled * (1 +  k_scale*(wlen - wlen[100]))





        # Eq. 9a)-9f)
        b_bw = 0.5 * b_w #Water is isotrophic
        a_g = G*np.exp(-S*(wlen - 440))
        b_bp = X * (400 / wlen) ** Y
        b_p = b_bp/B_p
        a = a_w + a_ph + a_g
        b = b_w + b_p

        b_b = b_bw + b_bp

        k = (a + b_b).reshape((1, -1))

        import matplotlib.pyplot as plt

        #plt.show()
#
        #plt.plot(wlen, I_hat*I_0)
        #plt.show()
        # Vector from source to bottom
        dir_s1 = pos_seabed - pos_s1
        dir_s2 = pos_seabed - pos_s2
        # Length of those
        R_s1 = np.linalg.norm(dir_s1, axis = 1).reshape((-1, 1))
        R_s2 = np.linalg.norm(dir_s2, axis = 1).reshape((-1, 1))
        # Length of camera-bottom vector
        R_c = np.linalg.norm(pos_seabed, axis = 1).reshape((-1, 1))

        # Returns an n by k vector where k are the spectral
        BP1_norm, BP1 = beam_pattern(dir_s0=dir_s10, dir = dir_s1, sigma= sigma_l, I_0 = I_0, I_hat = I_hat)
        BP2_norm, BP2 = beam_pattern(dir_s0=dir_s20, dir = dir_s2, sigma=sigma_l, I_0=I_0, I_hat = I_hat)

        # Returns an n by 1 vector of cosines
        gamma_s1 = compute_gamma(dir = dir_s1, normals = norm_vec)
        gamma_s2 = compute_gamma(dir=dir_s2, normals = norm_vec)

        if id == 1:
            self.cos_gamma_s1_t1 = np.cos(gamma_s1)
            self.cos_gamma_s2_t1 = np.cos(gamma_s2)
            self.BP1_norm_t1 = BP1_norm
            self.BP2_norm_t1 = BP2_norm
            self.R_s1_t1 = R_s1
            self.R_s2_t1 = R_s2

        elif id == 2:
            # Measurements
            self.cos_gamma_s1_t2 = np.cos(gamma_s1)
            self.cos_gamma_s2_t2 = np.cos(gamma_s2)
            self.BP1_norm_t2 = BP1_norm
            self.BP2_norm_t2 = BP2_norm
            self.R_s1_t2 = R_s1
            self.R_s2_t2 = R_s2

        # Simple computation of incident light
        E_i1 = BP1 * np.cos(gamma_s1) * np.exp(-k * R_s1) / (R_s1 ** 2)
        E_i2 = BP2 * np.cos(gamma_s2) * np.exp(-k * R_s2) / (R_s2 ** 2)




        import matplotlib.pyplot as plt


        if is_skogn == False:
            E_i = E_i1 + E_i2
        else:
            E_i = E_i1 # Only this light source that is illuminating the scene

        # Calculate from measurement to seafloor
        theta_w = theta_w_dot - alpha # Intersection angle with respect to glass

        I_f = immersion_factor(theta_w=theta_w, n_w = n_w, n_g = n_g, pixel_nr = pixel_nr)

        L_w = np.einsum('ij,ij -> ij', L_air, I_f)

        #if id == 2:
        #    from scipy.ndimage import gaussian_filter1d
        #    plt.hist(L_air.reshape(-1), 100)
        #    plt.show()

        # Compute backscatter
        include_backscatter = False
        if include_backscatter:
            x_pix = np.tan(theta_w_dot).reshape((-1, 1))
            y_pix = np.zeros(x_pix.shape)
            z_pix = np.ones(x_pix.shape)
            d_pix = np.concatenate((x_pix, y_pix, z_pix), axis = 1)
            k_bands = k.shape[1]

            if is_skogn:
                L_s1 = compute_backscatter(B_p=B_p, b=b.reshape((1, 1, k_bands)), b_w=b_w.reshape((1, 1, k_bands)),
                                           b_p=b_p.reshape((1, 1, k_bands)), d_pix=d_pix, pos_seabed=pos_seabed,
                                           p_si=pos_s1, dir_s0=dir_s10, sigma_l=sigma_l,
                                           I_0=I_0, I_hat=I_hat, k=k.reshape((1, 1, k_bands)), pixel_nr=pixel_nr)
                L_s = L_s1# The total scattered light
            else:

                L_s1 = compute_backscatter(B_p = B_p, b = b.reshape((1, 1, k_bands)), b_w = b_w.reshape((1, 1, k_bands)),
                                           b_p = b_p.reshape((1, 1, k_bands)), d_pix = d_pix, pos_seabed = pos_seabed,
                                           p_si = pos_s1, dir_s0 = dir_s10, sigma_l = sigma_l,
                                           I_0 = I_0, I_hat = I_hat, k= k.reshape((1, 1, k_bands)), pixel_nr = pixel_nr)

                L_s2 = compute_backscatter(B_p=B_p, b=b.reshape((1, 1, k_bands)), b_w=b_w.reshape((1, 1, k_bands)),
                                           b_p=b_p.reshape((1, 1, k_bands)), d_pix=d_pix, pos_seabed=pos_seabed,
                                           p_si=pos_s2, dir_s0=dir_s20, sigma_l=sigma_l,
                                           I_0=I_0, I_hat=I_hat, k=k.reshape((1, 1, k_bands)), pixel_nr=pixel_nr)
                L_s = L_s1 + L_s2  # The total scattered light



        else:
            L_s1 = 0
            L_s2 = 0
            L_s = L_s1 + L_s2

        #poly_dir_1 = p1 * (theta_w_dot - theta_s1) ** 2 + p2
        #poly_dir_2 = p1 * (theta_w_dot + theta_s1) ** 2 + p2

        #L_s1 = I_0 * int_1 * b_b * poly_dir_1
        #L_s2 = I_0 * int_2 * b_b * poly_dir_2



        L_d = L_w - L_s # The direct light

        #import matplotlib.pyplot as plt

        L_r = L_d * np.exp(k * R_c) # Backwards attenuation
        #plt.hist(R_c, 50)
        #plt.show()
        rho = np.pi * L_r / E_i

        #print(np.sum(np.isnan(rho)))
        #print(np.sum(np.isinf(rho)))
        return rho

    def forward_model_simple(self, id, is_skogn):
        """This forward model propagates light for a set of Param. The error is evaluated in digital counts"""
        ## IOP params
        G = self.Param.G
        X = self.Param.X
        Y = self.Param.Y
        # B

        ## Light source params
        I_0 = self.Param.I_0
        sigma_l = self.Param.sigma_l

        ## Immersion factor params
        alpha = self.Param.alpha


        if id == 1:
            # Measurements
            L_air = self.Meas.rad1

            # Variables
            wlen = self.Var.bands1
            pos_seabed = self.Var.points1
            norm_vec = self.Var.normals1
            theta_w_dot = self.Var.angles1
            pixel_nr = self.Var.pixel_nr1
            RF = self.Var.RF1
            t_exp = self.Var.t_exp1
        elif id == 2:
            # Measurements
            L_air = self.Meas.rad2
            # Variables
            wlen = self.Var.bands2
            pos_seabed = self.Var.points2
            norm_vec = self.Var.normals2
            theta_w_dot = self.Var.angles2
            pixel_nr = self.Var.pixel_nr2
            RF = self.Var.RF2
            t_exp = self.Var.t_exp2


        # Constants
        a_ph = 0
        a_w = self.Const.a_w
        b_bw = self.Const.b_w
        S = self.Const.S
        dir_s10 = self.Const.dir_s1
        dir_s20 = self.Const.dir_s2
        pos_s1 = self.Const.pos_s1
        pos_s2 = self.Const.pos_s2

        n_w = self.Const.n_w.reshape((1, -1))
        n_g = self.Const.n_g.reshape((1, -1))
        #P_0 = self.Const.P_0 # Defines total Spectral flux in air
        I_hat = self.Const.I_hat
        rho_gt = self.Const.refl_gt.reshape((1, -1))

        # IOP definitions from Lee
        a_g = G*np.exp(-S*(wlen - 440))

        a = a_w + a_ph + a_g

        b_bp_dot = X*(400/wlen)**Y

        b_b = b_bw + b_bp_dot # eq 18

        k = (a + b_b).reshape((1, -1))


        # Vector from source to bottom
        dir_s1 = pos_seabed - pos_s1
        dir_s2 = pos_seabed - pos_s2
        # Length of those
        R_s1 = np.linalg.norm(dir_s1, axis = 1).reshape((-1, 1))
        R_s2 = np.linalg.norm(dir_s2, axis = 1).reshape((-1, 1))


        import matplotlib.pyplot as plt


        # Length of camera-bottom vector
        R_c = np.linalg.norm(pos_seabed, axis = 1).reshape((-1, 1))

        # Returns an n by k vector where k are the spectral
        BP1_norm, BP1 = beam_pattern(dir_s0=dir_s10, dir = dir_s1, sigma= sigma_l, I_0 = I_0, I_hat = I_hat)
        BP2_norm, BP2 = beam_pattern(dir_s0=dir_s20, dir = dir_s2, sigma=sigma_l, I_0=I_0, I_hat = I_hat)

        # Returns an n by 1 vector of cosines
        gamma_s1 = compute_gamma(dir = dir_s1, normals = norm_vec)
        gamma_s2 = compute_gamma(dir=dir_s2, normals = norm_vec)

        # Simple computation of incident light
        E_i1 = BP1 * np.cos(gamma_s1) * np.exp(-k * R_s1) / (R_s1 ** 2)
        E_i2 = BP2 * np.cos(gamma_s2) * np.exp(-k * R_s2) / (R_s2 ** 2)






        if is_skogn == False:
            E_i = E_i1 + E_i2
        else:
            E_i = E_i2



        # Calculate reflected light:
        L_r = (rho_gt/np.pi)*E_i

        L_d = L_r*np.exp(-k*R_c)

        # Compute backscatter
        include_backscatter = False
        if include_backscatter:
            L_s1 = compute_scattering_integral(k, R_s1)
            L_s2 = compute_scattering_integral(k, R_s2)
        else:
            L_s1 = 0
            L_s2 = 0

        L_s = L_s1 + L_s2

        L_w = L_d + L_s

        # Calculate from measurement to seafloor
        theta_w = theta_w_dot - alpha  # Intersection angle with respect to glass

        I_f = immersion_factor(theta_w=theta_w, n_w=n_w, n_g=n_g, pixel_nr=pixel_nr)

        L_air_est = np.einsum('ij,ij -> ij', L_w, 1/I_f)

        rad_2_DC = radiance_conversion(RF = RF, t_exp = t_exp, pixel_nr = pixel_nr)

        DC_air_est = np.einsum('ij,ij -> ij', L_air_est, rad_2_DC)

        DC_air_meas = np.einsum('ij,ij -> ij', L_air, rad_2_DC)

        DC_diff = DC_air_meas - DC_air_est

        return DC_diff
    #    # The measurement is L_a meaning radiance in air
#
#
    #    L_m = L_plus*I_f
#


