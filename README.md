# gref4hsi

The hyperspectral toolchain is a toolbox for the georeferencing and orthorectification of hyperspectral pushbroom data. It was made with a special emphasis on georeferencing and co-registration of underwater hyperspectral data. However, it is equally simple to use for airborne data.

Installation instructions:

•	Set up a virtual environment with python 3.8. The following has been confirmed to work with anaconda virtual environment with python=3.8 on a windows 10, dell precision

•	Metashape, GDAL and rasterio: These are downloaded as wheel (*.whl) files for python 3.8 to the “/dependencies” folder. Navigate here “cd /dependencies” and run the following to install these:
for %x in (dir *.whl) do python -m pip install %x

•	For remaining dependencies, run:
pip install -r requirements.txt

•	A *.bat script was added for automated installation and testing.

•	The pyvista function multi_ray_trace depends on the pyembree install and should be tested with “/test/test_multi_ray_trace.py”. For my machine the test takes 0.35 s. If it does not work, consider installing pyembree with anaconda (for mac and linux):
conda install -c conda-forge pyembree

