import configparser


def prepend_data_dir_to_relative_paths(config_path, DATA_DIR):
    config = configparser.ConfigParser()
    config.read(config_path)
    #mission_dir = config.get('General', 'missiondir')
    
    config.set('General', 'missiondir', DATA_DIR)

    # Check if the 'Absolute Paths' section already exists
    if 'Absolute Paths' not in config:
        # Create a new section for 'Absolute Paths'
        config.add_section('Absolute Paths')

    # Iterate through the key-value pairs in 'Relative Paths'
    for key, relative_path in config['Relative Paths'].items():
        # Copy the key-value pair to 'Absolute Paths' section
        config.set('Absolute Paths', key, DATA_DIR + relative_path)
    # Save the updated configuration to the file
    with open(config_path, 'w') as configfile:
        config.write(configfile)

