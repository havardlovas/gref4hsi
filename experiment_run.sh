#!/bin/bash

python3 gref4hsi/tests/test_main_dbe.py --data_dir="/media/leo/NESP_1/NTNU/UHI_Data/Gref_processed/Svea2_Day1/Transect_2/Gref_10mm/" --resolution=0.01 --interpolation=False ;

python3 gref4hsi/tests/test_main_dbe.py --data_dir="/media/leo/NESP_1/NTNU/UHI_Data/Gref_processed/Svea2_Day1/Transect_6/Gref_10mm/" --resolution=0.01 --interpolation=False ;


