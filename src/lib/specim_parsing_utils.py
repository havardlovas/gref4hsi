"""This script reads Specim data from default captures and reformats data to compatibility with pipeline"""

from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.optimize import least_squares

from scripts.geometry import CalibHSI
# Relative folder paths


class Specim():
    def __init__(self, mission_path, config):
        self.mission_path = mission_path # With slash end
        self.config = config #
    
    @staticmethod
    def fov_2_param(fov):
        """
        param fov: view angle of each pixel in degrees
        returns: A dictionary of camera parameters
        """
        # Script for reading the FOV file and converting it to a camera model
        def optimize_func(param, x_true, n_pix):

            f = param[0]  # The focal length of the lens
            k1 = param[1]  # The 1st Radial distortion
            k2 = param[2]  # The 2nd Radial distortion
            k3 = param[3]
            c_x = param[4]
            n_pix = x_true.size
            #v_c = n_pix / 2 + 1
            u = np.arange(1, n_pix + 1).reshape((-1,1))

            x_norm_lin = (u - c_x) / f
            # Distortion term if to be used
            x_norm_nonlin = -(k1 * ((u - c_x) / 1000) ** 5 + \
                            k2 * ((u - c_x) / 1000) ** 3 + \
                            k3 * ((u - c_x) / 1000) ** 2) / f

            
            x_norm = x_norm_lin + x_norm_nonlin
            diff_x = x_norm-x_true
            return diff_x.reshape(-1)
        
        
        theta_true = fov*np.pi/180

        x_true = np.array(np.tan(theta_true)).reshape((-1, 1))

        n_pix = x_true.size

        # Total angular FOV is:
        AFOV = np.abs(theta_true[0]) + np.abs(theta_true[-1])
        f_0 = (n_pix/2)/(np.tan(AFOV/2))

        param_0 = np.array([f_0, 0, 0, 0, 1 + n_pix/2])

        res = least_squares(optimize_func, param_0, args = (x_true, n_pix), x_scale = 'jac')

        #print(res.x)# Print the found camera model.

        param = res.x

        

        f = param[0]  # The focal length of the lens
        k1 = param[1]  # The 1st Radial distortion
        k2 = param[2]  # The 2nd Radial distortion
        k3 = param[3]  # The tangential distortion
        c_x = param[4] # The
        """rotation_x, rotation_y, rotation_z, translation_x, translation_y, translation_z, c_x, focal_length,
        distortion_coeff_1, distortion_coeff_2, distortion_coeff_3"""

        # initialized to 0
        rotation_x = 0
        rotation_y = 0
        rotation_z = 0
        translation_x = 0
        translation_y = 0
        translation_z = 0

        param_dict = {'rx':rotation_x,
                      'ry':rotation_y,
                      'rz':rotation_z,
                      'tx':translation_x,
                      'ty':translation_y,
                      'tz':translation_z,
                      'f': f,
                      'cx': c_x,
                      'k1': k1,
                      'k2': k2,
                      'k3': k3,
                      'width': n_pix}
        return param_dict

    def read_fov_file(self, fov_file_path):
        

        

        df_fov = pd.read_csv(fov_file_path, sep = ',', header = None)

        fov_arr = df_fov.values[:, 1]

        param_dict = Specim.fov_2_param(fov = fov_arr)
        
        file_name_calib = self.config['Absolute Paths']['hsicalibfile']

        CalibHSI(file_name_cal_xml=file_name_calib, 
                 config = self.config, 
                 mode = 'w', 
                 param_dict = param_dict)

    def read_nav_file(self, nav_file_path, date):
        # Convert date string to datetime object
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        # Open file for reading.
        with open(nav_file_path, 'r') as nav_file:
            count = 0
            imu_data = []
            gnss_data = []
            sync_data = []
            # Set to -1 to an if sentence to initialize to true
            hsi_frame_nr_prev = -1
            # Variable introduced to convert from cyclic HS frame counting to cumulative counts
            restart_counts = 0

            while True:
                count += 1

                # Get next line from file
                line = nav_file.readline()

                line_arr = line.split(sep = ',')

                if line_arr[0] == '$PASHR':
                    Timestamp_str = line_arr[1]
                    sec = float(Timestamp_str[4:])
                    minut = int(Timestamp_str[2:4])
                    hour = int(Timestamp_str[0:2])

                    # Create a timedelta object for the time
                    time_delta = timedelta(hours=hour, minutes=minut, seconds=sec)

                    # Calculate the total timedelta
                    total_timedelta = date_obj + time_delta

                    # Convert to UNIX timestamp (decimal UNIX time)
                    unix_timestamp = (total_timedelta - datetime(1970, 1, 1)) / timedelta(seconds=1)

                    TimestampAbs = unix_timestamp

                    TimestampClock = date + '_' + Timestamp_str

                    Yaw = float(line_arr[2]) # Heading wrt true north
                    Roll = float(line_arr[4])
                    Pitch = float(line_arr[5])# According to https://docs.novatel.com/OEM7/Content/SPAN_Logs/PASHR.htm


                    sigmaRoll = float(line_arr[7])
                    sigmaPitch = float(line_arr[8])  # According to https://docs.novatel.com/OEM7/Content/SPAN_Logs/PASHR.htm
                    sigmaYaw = float(line_arr[9])

                    #dlogr_imu.append_data(np.array([TimestampClock, TimestampAbs, Roll, Pitch, Yaw, sigmaRoll, sigmaPitch, sigmaYaw]))

                    imu_data_row = {'TimestampClock':TimestampClock,
                                     'TimestampAbs': TimestampAbs,
                                     'Roll': Roll,
                                     'Pitch': Pitch,
                                     'Yaw': Yaw,
                                     'sigmaRoll': sigmaRoll,
                                     'sigmaPitch': sigmaPitch,
                                     'sigmaYaw': sigmaYaw}
                    imu_data.append(imu_data_row)

                elif line_arr[0] == '$GNGGA' or line_arr[0] == '$GPGGA':
                    Timestamp_str = line_arr[1]
                    sec = float(Timestamp_str[4:])
                    minut = float(Timestamp_str[2:4])
                    hour = float(Timestamp_str[0:2])

                    # Create a timedelta object for the time
                    time_delta = timedelta(hours=hour, minutes=minut, seconds=sec)

                    # Calculate the total timedelta
                    total_timedelta = date_obj + time_delta

                    # Convert to UNIX timestamp (decimal UNIX time)
                    unix_timestamp = (total_timedelta - datetime(1970, 1, 1)) / timedelta(seconds=1)

                    TimestampAbs = unix_timestamp

                    TimestampClock = date + '_' + Timestamp_str

                    LatDDMM_MM = float(line_arr[2]) # DDMM.mm
                    LatInt = int(LatDDMM_MM/100)
                    Lat = LatInt + (LatDDMM_MM - LatInt*100)/60


                    LonDDDMM_MM = float(line_arr[4]) # DDDMM.mm
                    LonInt = int(LonDDDMM_MM / 100)
                    Lon = LonInt + (LonDDDMM_MM - LonInt * 100) / 60

                    AltMSL = float(line_arr[9])
                    AltGeoid = float(line_arr[11])  # According to https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm

                    #dlogr_gnss.append_data(np.array([TimestampClock, TimestampAbs, Lon, Lat, AltMSL, AltGeoid]))
                    gnss_data_row = {'TimestampClock':TimestampClock,
                                     'TimestampAbs': TimestampAbs,
                                     'Lon': Lon,
                                     'Lat': Lat,
                                     'AltMSL': AltMSL,
                                     'AltGeoid': AltGeoid}
                    gnss_data.append(gnss_data_row)
                elif line_arr[0] == '$SPTSMP':
                    """Recordings of time sync messages, tied to the hyperspectral frames"""
                    hsi_frame_number_new = int(line_arr[2]) - 1 # Seems they start with a 1-base (like MATLAB)

                    if hsi_frame_number_new < hsi_frame_nr_prev: # When frame counter resets
                        restart_counts += (hsi_frame_nr_prev - hsi_frame_number_new) + self.metadata_obj.fps # 9950 + 50


                    hsi_frame_number = hsi_frame_number_new
                    hsi_frame_nr_prev = hsi_frame_number_new
                    prev_time = TimestampAbs

                    # In the file, fram counts go 1, 51, 101,... 9951, 1 ... in a cyclic manner. So to get cumulative counts we add the restart_counts
                    sync_data_row = {'HsiFrameNum': hsi_frame_number + restart_counts,
                                     'TimestampAbs': prev_time,
                                     'TimestampClock': TimestampClock}
                    sync_data.append(sync_data_row)




                # if line is empty
                # end of file is reached
                if not line:
                    break
        
        self.imu_data = imu_data
        self.gnss_data = gnss_data
        self.sync_data = sync_data

            
            









