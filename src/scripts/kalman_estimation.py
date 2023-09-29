## A library for different implementations of Kalman filters:
import visualize
import numpy as np
from scipy.interpolate import interp1d
from geometry import rotation_matrix_ecef2ned
class KalmanFilter():
    def __init__(self, type):

    print('A proper Kalman filter should be designed')








class KalmanPhotogrammetry():
    def __init__(self, time_photo, photo_position, photo_rotation, time_imu, imu_acc_raw, imu_omega_raw, rot_cam_imu, r_imu_cam, dt, filt_dir, ECEF = False):

        if ECEF == False:
            self.R_ned_ecef = np.eye(3)
        else:

            self.R_ned_ecef =
        # Photo_poses are linear position vectors while rotation is scipy rotation object
        self.time_photo = time_photo
        self.photo_position = photo_position
        self.photo_rotation = photo_rotation
        self.imu_rotation = photo_rotation * rot_cam_imu # Composing this rotation

        self.time_imu = time_imu
        self.imu_acc_raw = imu_acc_raw
        self.imu_omega_raw = imu_omega_raw

        # Decinding the start/end frame of the photo
        self.start_time = time_photo[0]
        self.end_time = time_photo[-1]

        self.time_vec = np.arange(start=self.start_time, end = self.end_time + dt, step = dt)

        linearAccInterpolator = interp1d(self.time_imu, np.transpose(self.imu_acc_raw), kind='nearest')
        self.acc = np.transpose(linearAccInterpolator(self.time_vec))

        linearOmegaInterpolator = interp1d(self.time_imu, np.transpose(self.imu_acc_raw), kind='nearest')

        self.omega = np.transpose(linearOmegaInterpolator(self.time_vec))

        # Deciding which photo frames belong to which. That is more about finding the argmin
        self.photo_ind = np.zeros(len(time_photo))
        for i in range(len(time_photo)):
            self.photo_ind[i] = np.argmin(np.abs(self.time_photo[i] - self.time_vec))

        # Simplified measurement matrices
        self.Hp = np.zeros((3, 15))
        self.Hp[1: 3, 1: 3] = np.eye(3)

        self.Hth = np.zeros((3, 15))
        self.Hth[1: 3, 1: 3] = np.eye(3)

        self.Rp = np.eye(3)*(1/1000)**2 # 1 mm
        self.Rth = np.eye(3) * (0.05 * np.pi / 180) ** 2  # 5 degrees

        self.r_imu_cam = r_imu_cam

        if filt_dir == -1:
            self.start_idx = len(self.time_vec) - 1
            self.stop_idx = 0
            self.step_idx = -1
            self.photo_idx = len(self.photo_idx) - 1

            self.quat_0 = self.photo_rotation[self.photo_idx]
            self.p_0 =









