import h5py
import numpy as np
import pandas as pd
import os
from PIL import Image
import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Process .h5 files to extract RGB data and IMU data.')
    parser.add_argument('--folder_path', type=str, default="/media/leo/NESP_1/NTNU/UHI_Data/29092023/Transect1", help='Path to the folder containing HDF5 files.')
    parser.add_argument('--save_path', type=str, default="/media/leo/NESP_1/NTNU/UHI_Data/29092023/Transect1/Extracted_RGB/", help='Path to the folder where extracted RGB images and IMU data will be saved.')
    parser.add_argument('--save_name', type=str, default="", help='Base name for saved files. If not given the h5 filename of the first transect is used')
    return parser.parse_args()


def main(folder_path, save_path, save_name):
    # Check for the existence of the Extracted_RGB folder or create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Created the directory:", save_path)
    else:
        print("Directory already exists:", save_path)

    # Initialize a list to hold all the data
    data_entries = []

    # Counter for images across files
    image_counter = 0

    # Find all .h5 files in the folder
    h5_files = glob.glob(os.path.join(folder_path, '*.h5'))

    # Initialize an empty DataFrame to collect IMU data
    df_imu = pd.DataFrame()

    for file_path in h5_files:
        print(f"Extracting files from: {file_path}")
        if save_name == "":
            # Extract the timestamp from the file name for each file
            save_name = os.path.basename(file_path).split('.')[0]

        # Open the HDF5 file
        with h5py.File(file_path, "r") as file:
            # Handle IMU data if available
            rawdata_imu = file.get('/rawdata/navigation/imu')
            if rawdata_imu:
                imu_data = {name: rawdata_imu[name][()].tolist() for name in rawdata_imu}
                df_imu = pd.concat([df_imu, pd.DataFrame(imu_data)], ignore_index=True)

            # Navigate to the '/rawdata/rgb' group
            rawdata_rgb = file['/rawdata/rgb']
            
            # Process non-image data
            non_image_data = {name: rawdata_rgb[name][()].tolist() for name in rawdata_rgb if name != 'rgbFrames'}

            # Now handle RGB data
            rgb_dataset = rawdata_rgb['rgbFrames']
            for i, img_data in enumerate(rgb_dataset):
                img_array = np.array(img_data)
                if img_array.ndim == 3:  # Confirm it's an RGB image
                    # Save the image
                    img = Image.fromarray(img_array.astype('uint8'))
                    img_name = f"{save_name}_image_{image_counter}.png"
                    img.save(os.path.join(save_path, img_name))

                    # Collect the associated non-image data for this particular image
                    non_image_data_for_img = {key: val[i] for key, val in non_image_data.items()}

                    # Add the image information and non-image data to the data_entries list
                    row_data = {
                        'FileName': img_name,
                        'SourceFile': save_name,
                        **non_image_data_for_img  # Unpack the non-image data dictionary here
                    }
                    data_entries.append(row_data)
                    image_counter += 1

    # Convert the list of dictionaries to a DataFrame
    df_rgb = pd.DataFrame(data_entries)

    # Save the DataFrames to CSV files
    df_rgb.to_csv(os.path.join(save_path, 'rgb_data.csv'), index=False)
    if not df_imu.empty:
        df_imu.to_csv(os.path.join(save_path, 'imu_data.csv'), index=False)

    print("All .h5 files have been processed.")

if __name__ == '__main__':
    args = get_args()
    main(args.folder_path, args.save_path, args.save_name)