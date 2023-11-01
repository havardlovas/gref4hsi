@echo on

:: Install local wheel files from the "\dependencies" directory
cd dependencies
for %%x in (*.whl) do (
    python -m pip install %%x
)

cd ..


:: Install remote dependencies from PyPI
pip install -r requirements.txt

::Runs the ray tracing test
python ./src/tests/test_multi_ray_trace.py
