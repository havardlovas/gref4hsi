import configparser

import Metashape as MS

import numpy as np

config_file = 'D:/HyperspectralDataAll/UHI/2023-09-29-Hopavaagen-1/configuration.ini'

config = configparser.ConfigParser()
config.read(config_file)

MS.License().activate("EE2Z6-O5ZVF-1JYNV-NKSRY-UXTGR")
print('Licence Activated')

doc = MS.Document()
doc.read_only = False

# Define *.psx file
path_psx = config['General']['pathPsx']

doc.open(path_psx, read_only=False)

# Access the active chunk
chunk = doc.chunk

# Check if there are any chunks in the document
if not chunk:
    print("No chunks found in the project.")
else:
    # Iterate through cameras in the chunk
    for camera in chunk.cameras:
        # Get the calibrated center of the camera
        calibrated_center = camera.calibrated().principal_point

        # Create a marker at the center
        marker = chunk.addMarker()
        marker.label = "CenterMarker"  # Set marker label
        marker.reference.location = calibrated_center  # Set marker location

    # Save the modified project
    doc.save()
    print("Markers added to the project.")

# Close the project
doc.close()



