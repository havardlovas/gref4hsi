# This is a function for running the different processes in a simple fashion with arguments.

# Let us try to start by generating
import subprocess
import spectral as sp

import agisoft_extract
import georeference
import visualize
import configparser

import misc


#misc.remove_index_from_dataframe('C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/pose.csv')



# The ini file contains all the settings
iniPathBarentsSea = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/BarentsSea06052021/configuration.ini'

iniPathTautra = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Tautra07032017/configuration.ini'

iniPathSkogn = 'C:/Users/haavasl/PycharmProjects/newGit/TautraReflectanceTools/Missions/Skogn21012021/configuration.ini'


iniPathNYAA = 'E:/nyaalesund_missions/flight_8/configuration.ini'

missionName = 'Skogn21012021'

if missionName == 'Tautra07032017':
    # Read in the config file.
    config = configparser.ConfigParser()
    config.read(iniPathTautra)
    #agisoft_extract.main(iniPathTautra)
    #misc.fix_naming_convention_error(config['General']['posePath'])

    #georeference.main(iniPathTautra, mode='georeference', is_calibrated=True) # Make a ton of feature points using best calibration

    # Could make some more customization options in the calibration. E.g. adding a translation vector or k1 distortion
    #georeference.main(iniPathTautra, mode='calibrate', is_calibrated=False)

    #georeference.main(iniPathTautra, mode='calibrate', is_calibrated=True)

    #georeference.main(iniPathTautra, mode='georeference',
    #                  is_calibrated=True)  # Make a ton of feature points using best calibration

    georeference.main(iniPathTautra, mode='georeference', is_calibrated=True)

    #georeference.main(iniPathTautra, mode='calibrate')
    #visualize.show_mesh_camera(config)
elif missionName == 'iniPathNYAA':
    config = configparser.ConfigParser()
    config.read(iniPathNYAA)
    import import_tools
    #import_tools.extract_model(config, iniPathNYAA)
    #import_tools.extract_pose_ardupilot(config, iniPathNYAA)
    visualize.show_mesh_camera(config)
    georeference.main(iniPathNYAA, mode='georeference', is_calibrated=True)
elif missionName == 'BarentsSea06052021':
    config = configparser.ConfigParser()
    config.read(iniPathBarentsSea)
    #agisoft_extract.main(iniPathBarentsSea)
    #visualize.show_mesh_camera(config)
    georeference.main(iniPathBarentsSea, mode='georeference', is_calibrated=False)
    georeference.main(iniPathBarentsSea, mode='calibrate', is_calibrated=True)
elif missionName == 'Skogn21012021':
    config = configparser.ConfigParser()
    config.read(iniPathSkogn)
    #agisoft_extract.main(iniPathSkogn)
    visualize.show_mesh_camera(config)
    #georeference.main(iniPathSkogn, mode='georeference', is_calibrated=False)
    #georeference.main(iniPathSkogn, mode='calibrate', is_calibrated=False)
    #georeference.main(iniPathSkogn, mode='georeference', is_calibrated=True)
    georeference.main(iniPathSkogn, mode='georeference', is_calibrated=True) # Makes a nice histogram that we can use


#agisoft_extract.main(iniPath)



# Perform georeferencing