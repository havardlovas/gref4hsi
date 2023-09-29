import pandas as pd
import numpy as np

def fix_naming_convention_error(csv_file_name):
    pose = pd.read_csv(
        csv_file_name, sep=',',
        header=0)
    isfirst = 1
    count = 0
    ind = 0
    for index, row in pose.iterrows():
        transect_name = row["CameraLabel"].split("_")[-3] + "_" + row["CameraLabel"].split("_")[-2]
        # This will happen the first time
        if isfirst:
            count = int(row["CameraLabel"].split("_")[-1]) - 1
            transect_name_prev = transect_name
            isfirst = 0




        if transect_name_prev != transect_name:
            count = 0
            transect_name_prev = transect_name
        else:
            count += 1


        new_transect_name = row["CameraLabel"].split("_")[-5] + "_" + row["CameraLabel"].split("_")[-4] + "_" + \
                            row["CameraLabel"].split("_")[-3] + "_" + row["CameraLabel"].split("_")[-2] + "_" + str(
            count)


        pose["CameraLabel"][ind] = new_transect_name
        ind += 1


    pose.to_csv(csv_file_name, index=False)


def remove_index_from_dataframe(csv_file_name):
    pose = pd.read_csv(
        csv_file_name, sep=',',
        header=0, index_col=0)


    pose.to_csv(csv_file_name, index=False)



