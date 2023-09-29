import h5py

def extract_rgb_images_DBY():
    print('This function should be made')

def create_ref_file_DBY():
    print('This function should also be made')

def write_HDF_to_RGB(filename, config):
    print('This function should be made')
    # Read in an HDF file
    with h5py.File(filename, 'r', libver='latest') as f:
        if eval(config['HDF']['isMarmineUHISpring2017']):
            RGBTimeStamps = f['rgb/parameters'][()][:, 2]
            RGBImgs = f['rgb/pixels'][()]

        else:
            RGBFramesPath = config['HDF.rgb']['rgbFrames']
            timestampRGBPath = config['HDF.rgb']['timestamp']

            RGBTimeStamps = f[timestampRGBPath][()]
            RGBImgs = f[RGBFramesPath][()]

            altitude = f['navigation/altitude/Altitude'][()]
            altitude_time = f['navigation/altitude/TimeStampMeasured'][()]

            heading = f['navigation/imu/Altitude'][()]
            altitude_time = f['navigation/imu/TimeStampMeasured'][()]
def main():
    extract_rgb_images_DBY()
    create_ref_file_DBY()




if __name__ == '__main__':
    main()