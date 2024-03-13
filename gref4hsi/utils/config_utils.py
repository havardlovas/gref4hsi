import configparser
import os


def prepend_data_dir_to_relative_paths(config_path, DATA_DIR, mkdirs = True):
    """The config file holds a series of default relative paths. By providing a parent directory (one per mission)
    the absolute paths can be established. To automatically generate the folder structure (make the directories), allow mkdirs=True

    :param config_path: path to config file for read and write
    :type config_path: str
    :param DATA_DIR: path to data directory. The modified config is put here.
    :type DATA_DIR: str
    :param mkdirs: Whether to make folders, defaults to True.
    :type mkdirs: bool, optional
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    # Set the value
    config.set('General', 'mission_dir', DATA_DIR)

    if mkdirs:
        if os.path.exists(DATA_DIR):
                pass
        else:
            os.makedirs(DATA_DIR, exist_ok=True)

    # Check if the 'Absolute Paths' section already exists
    if 'Absolute Paths' not in config:
        # Create a new section for 'Absolute Paths'
        config.add_section('Absolute Paths')

    # Iterate through the key-value pairs in 'Relative Paths'
    for key, relative_path in config['Relative Paths'].items():
        # Copy the key-value pair to 'Absolute Paths' section
        absolute_path = os.path.join(DATA_DIR, relative_path)
        config.set('Absolute Paths', key, absolute_path)

        # Only directories end with separators
        is_dir = (len(absolute_path.split('.'))==1)

        # Only create directories if path is a valid directory
        if mkdirs and is_dir:
            # If it exists, do nothing
            if os.path.exists(absolute_path):
                pass
            else:
                os.makedirs(absolute_path, exist_ok=True)

    # Save the updated configuration to the file
    
    config_path_write = os.path.join(DATA_DIR, 'configuration.ini')
    
    with open(config_path_write, 'w') as configfile:
        config.write(configfile)

def customize_config(config_path, dict_custom):
    """Allows changing dictionary-specified entries of the config to custom values

    :param config_path: path to config file for read and write
    :type config_path: str
    :param dict_custom: dictionary of custom options for processing.
    :type dict_custom: nested dictionary. Dictionary of sections which are dictionaries
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    

    # dict_custom like config is a two-level nested dictionary
    for key_section, section_dict in dict_custom.items():
        # Iterate through each section, and assign custom values 
        for key_sub_section, value_custom in section_dict.items():
            try:
                config.set(key_section, key_sub_section, str(value_custom))
            except configparser.NoSectionError:
                config.add_section(key_section)
                config.set(key_section, key_sub_section, str(value_custom))



    # Save the updated configuration to the file
    with open(config_path, 'w') as configfile:
        config.write(configfile)

