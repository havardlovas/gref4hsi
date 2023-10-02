# hyperspectral_toolchain

The hyperspectral toolchain is a toolbox for the post-processing of hyperspectral pushbroom data. It was made with a special emphasis on georeferencing and co-registration of underwater hyperspectral data. However, it is equally simple to use for airborne data.

Installation instructions:

•	Set up a virtual environment with python 3.8. It is possible to attempt this from the requirements.txt or *.yml file but then I would put gdal to the top of the dependency list.

•	Metashape, GDAL and rasterio: These are downloaded as wheel (*.whl) files to the “/dependencies” folder. Navigate here “cd /dependencies” and run the following to install these:
for %x in (dir *.whl) do python -m pip install %x

•	For remaining dependencies, run:
pip install opencv-python dill geopandas h5py lmfit matplotlib numpy open3d pandas Pillow pymap3d pyproj pyvista pyvistaqt scikit-learn scipy shapely spectral xmltodict pykdtree trimesh rtree pyembree --user

•	The pyvista function multi_ray_trace depends on the Embree library for CPU acceleration and could be tested with “/test/test_multi_ray_trace.py”. If it does not work, check Add pip installation directions for pyembree · Issue #1529 · pyvista/pyvista · GitHub for how to install Embree.

