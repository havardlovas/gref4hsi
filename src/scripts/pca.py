"""
Script Description:
This Python script extracts the principal component information out of hyperspectral datacubes. 
Author: Leonard Günzel
Date: 21.11.2023

Dependencies:
- h5py: For handling HDF5 files.
- numpy and pandas: For data manipulation and analysis.
- os: For interacting with the operating system.
- PIL.Image: For working with images.
- glob: For file path pattern matching.
- argparse: For parsing command-line arguments.

Usage:
- Run the script with appropriate command-line arguments to process HDF5 files and extract PCA information.

Command Line Arguments:
--folder_path: Path to the folder containing HDF5 files.
--save_path: Path to the folder where extracted RGB images and IMU data will be saved.
--save_name: Base name for saved files. If not given, the h5 filename of the first transect is used.

Example:
python script_name.py --folder_path "/path/to/hdf5_files" --save_path "/path/to/save/folder" --save_name "experiment_data"

Note: Ensure that the required dependencies are installed before running the script.
"""

import h5py
import numpy as np
import pandas as pd
import os
from PIL import Image
import glob
import argparse
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle as pk
import numpy.ma as ma

def get_args():
    parser = argparse.ArgumentParser(description='Process .h5 files to extract RGB data and IMU data.')
    parser.add_argument('--folder_path', type=str, default="/media/leo/NESP_1/NTNU/UHI_Data/Ao2022_UHI/UHI_original/ice-station-10-marinal-ice-zone/Transect_3/uhi_20220817_094202_1.h5", help='Path to the folder containing HDF5 files, can also be a single file')
    parser.add_argument('--save_path', type=str, default="/media/leo/NESP_1/NTNU/UHI_Data/Ao2022_UHI/UHI_original/ice-station-10-marinal-ice-zone/Transect_3/", help='Path to the folder where extracted RGB images and IMU data will be saved.')
    parser.add_argument('--save_name', type=str, default="", help='Base name for saved files. If not given the h5 filename of the first transect is used')
    parser.add_argument('--datacube_path', type=str, default="/rawdata/hyperspectral/dataCube", help='Path inside the h5 datacube that points to where the datacube is saved e.g. /rawdata/hyperspectral/dataCube')
    return parser.parse_args()


def main(folder_path, save_path, save_name, datacube_path):
    n_PC_scree = 10
    # Check for the existence of the Extracted_RGB folder or create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Created the directory:", save_path)
    else:
        print("Directory already exists:", save_path)

    if folder_path.endswith("/"):
        # Find all .h5 files in the folder
        h5_files = glob.glob(os.path.join(folder_path, '*.h5'))
    elif folder_path.endswith(".h5"):
        # We assume it is pointing to a file
        h5_files = [folder_path]
    else: 
        print("folder_path format is not recognized.")

    # Initialize an empty DataFrame to collect IMU data
    df_imu = pd.DataFrame()

    # Initialize an empty DataFrame to collect Altimeter data
    df_alti = pd.DataFrame()

    for file_path in h5_files:
        print(f"Extracting files from: {file_path}")
        if save_name == "":
            # Extract the timestamp from the file name for each file
            save_name = os.path.basename(file_path).split('.')[0]

        # Open the HDF5 file
        with h5py.File(file_path, "r") as h5file:
            # Handle datacube data if available
            full_datacube = h5file[datacube_path][:]
            no_wavelengths = full_datacube.shape[2]
            original_wavelengths = np.arange(380, 750, ((750-380)/no_wavelengths))
            
            # For Testing purposes
            pca = pk.load(open(f"{save_path}_{save_name}_pca.pkl",'rb')) 
            pca_result = pk.load(open(f"{save_path}_{save_name}_pca_results.pkl",'rb')) 
            loadings_df = pk.load(open(f"{save_path}_{save_name}_pca_loadings_df.pkl",'rb')) 
            
            #pca, pca_result, loadings_df = principal_component_analysis(full_datacube, save_path, save_name, original_wavelengths)
            
            print_results(pca, pca_result,loadings_df, original_wavelengths, top_n_wavelenths = 10)
            # Display the different PC values in a Scree plot
            print("Displaying Scree Plot")
            #scree_plot(pca, n_PC_scree)

            # Display the wavelengths in a Scatter plot with the first two PC as axis
            print("Displaying Scatter Plot")
            scatter_plot(pca, loadings_df, original_wavelengths)

    print("All .h5 files have been processed.")

def principal_component_analysis(full_datacube, save_path, save_name, original_wavelengths):
    print("Reshaping and Scaling Data")
    # Reshape the 3D datacube into a 2D array (samples, features)
    print(f"Original Datacube | Min: {full_datacube.min()} Max: {full_datacube.max()} Shape: {full_datacube.shape}")
    reshaped_data = full_datacube.reshape((-1, full_datacube.shape[2]))
    print(f"Reshaped Data | Min: {reshaped_data.min()} Max: {reshaped_data.max()} Shape: {reshaped_data.shape}")
    scaled_data = preprocessing.scale(reshaped_data)
    print(f"Scaled Data | Min: {scaled_data.min()} Max: {scaled_data.max()} Shape: {reshaped_data.shape}")

    num_components = min(reshaped_data.shape) - 1
    # Initialize PCA with the number of components you desire
    print("Starting PCA Analysis")
    pca = PCA(n_components=num_components, svd_solver="arpack")
    
    # Fit and transform the data
    pca_result = pca.fit_transform(scaled_data)

    # Create a DataFrame to store loadings for each principal component
    loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC_{i}' for i in range(1, pca.n_components_ + 1)])

    
    print("Dumping all the returns into pickles")
    pk.dump(pca, open(f"{save_path}_{save_name}_pca.pkl","wb"))
    pk.dump(pca_result, open(f"{save_path}_{save_name}_pca_results.pkl","wb"))
    pk.dump(loadings_df, open(f"{save_path}_{save_name}_pca_loadings_df.pkl","wb"))

    return pca, pca_result, loadings_df

def scree_plot(pca, n_PC_scree):
    """
    Creates a Scree-Plot with the n most important Principal Components
    """
        # Plotting the data
    labels = [f'PC_{x}' for x in range(1, (n_PC_scree+1))]
    plt.bar(x=range(1,(n_PC_scree+1)), height=pca.explained_variance_ratio_[0:n_PC_scree], tick_label=labels)
    plt.ylabel('Percentage of explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    

def scatter_plot(pca, loadings_df, original_wavelengths):
    """
    Creates a Scatter-Plot for all entries over PC1 and PC2
    """
    labels_PC = [f'PC_{x}' for x in range(1, len(original_wavelengths) + 1)]  # Adjusted the range
    labels_wavelengths = [f'{y} nm' for y in np.round(original_wavelengths, 1)]
    pca_df = pd.DataFrame(loadings_df, index=[labels_wavelengths], columns=labels_PC)
    
    plt.scatter(loadings_df['PC_1'], loadings_df['PC_2'], c=original_wavelengths, cmap='rainbow')  # Added c and cmap parameters
    plt.title('Scatter Plot PC1 and PC2 Loadings')
    plt.xlabel(f'PC1 - {pca.explained_variance_ratio_[0]}')
    plt.ylabel(f'PC2 - {pca.explained_variance_ratio_[1]}')
    
    # Display every fifth sample label
    for label, x, y in zip(labels_wavelengths[::10], loadings_df['PC_1'][::10], loadings_df['PC_2'][::10]):
        plt.annotate(label, (x, y))

    plt.colorbar(label='Wavelength (nm)')  # Added colorbar for better understanding of color mapping
    plt.show()

def print_results(pca, pca_result,loadings_df, original_wavelengths, top_n_wavelenths):
    # Get the top influential wavelengths for each principal component
    top_indices_first = loadings_df.iloc[:, 0].abs().sort_values(ascending=False).index
    #top_indices_second = loadings_df.iloc[:, 1].abs().sort_values(ascending=False).index

    print("Top indices for the first principal component:", original_wavelengths[top_indices_first[0:10]])
    #print("Top indices for the second principal component:", top_indices_second)

    # Map indices to corresponding wavelengths for each principal component
    #top_wavelengths_first = original_wavelengths[top_indices_first]
    #top_wavelengths_second = original_wavelengths[top_indices_second]

    # Display the loading of each wavelength for the first principal component
    #print("Loading of each wavelength for the first principal component:")
    #print(loadings_df['PC_1'])

    # Display the loading of the PCA in general
    #print("Loading of the PCA in general:")
    #print(pca.components_)

    # Display the explained variance ratio for each principal component
    #print("Explained Variance Ratio:")
    #print(pca.explained_variance_ratio_)

    # Sum the loadings for each wavelength over all principal components
    summed_loadings = loadings_df.sum(axis=1)

    # Map indices to corresponding wavelengths
    summed_wavelengths = pd.DataFrame(original_wavelengths)

    # Display the summed loadings for each wavelength
    #print("Summed Loadings for each wavelength:")
    #print(summed_loadings)

    # Identify the top N important wavelengths
    # Composite feature importance measure in PCA based on the sum of loadings across components.
    top_n_wavelengths_indices = summed_loadings.abs().nlargest(top_n_wavelenths).index
    #top_n_wavelengths = summed_wavelengths.iloc[top_n_wavelengths_indices]
        
    # Display the loading values for the top N wavelengths
    print("Loading values for the top N wavelengths:")
    for index in top_n_wavelengths_indices:
        print(f"Wavelength: {summed_wavelengths.iloc[index][0]}, Loading Value: {summed_loadings.iloc[index]}")

if __name__ == '__main__':
    args = get_args()
    main(args.folder_path, args.save_path, args.save_name, args.datacube_path)
    