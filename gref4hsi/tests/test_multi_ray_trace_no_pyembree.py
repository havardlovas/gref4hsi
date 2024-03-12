import time

import numpy as np
import pyvista as pv
from pykdtree.kdtree import KDTree
from pyvista import examples
import trimesh

# Record the start time
start_time = time.time()

# Load data
data = examples.load_random_hills()
data.translate((10, 10, 10))




# Create triangular plane (vertices [10, 0, 0], [0, 10, 0], [0, 0, 10])
size = 10
vertices = np.array([[size, 0, 0], [0, size, 0], [0, 0, size]])
face = np.array([3, 0, 1, 2])

planes = pv.PolyData(vertices, face)

# Subdivide plane so we have multiple points to project to
planes = planes.subdivide(8)

# Get origins and normals
origins = planes.cell_centers().points
normals = planes.compute_normals(cell_normals=True, point_normals=False)["Normals"]

# Convert faces to nx3 format
faces = data.faces.reshape(-1, 3)

# Convert PolyData to trimesh.Trimesh
data_mesh = trimesh.Trimesh(vertices=data.points, faces=faces)

# Vectorized Ray trace
"""points, pt_inds, cell_inds = data.multi_ray_trace(
    origins, normals
)"""  # Must have rtree, trimesh, and pyembree installed


ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(geometry=data_mesh)

# Perform ray tracing using trimesh.ray.intersects_id
cell_inds, pt_inds, points = ray_mesh_intersector.intersects_id(ray_origins=origins,  
                                                                ray_directions=normals, 
                                                                multiple_hits=False,
                                                                return_locations=True)


# Filter based on distance threshold, if desired (mimics VTK ray_trace behavior)
# threshold = 10  # Some threshold distance
# distances = np.linalg.norm(origins[inds] - points, ord=2, axis=1)
# inds = inds[distances <= threshold]

tree = KDTree(data.points.astype(np.double))
_, data_inds = tree.query(points)

elevations = data.point_data["Elevation"][data_inds]

# Mask points on planes
planes.cell_data["Elevation"] = np.zeros((planes.n_cells,))
planes.cell_data["Elevation"][pt_inds] = elevations
planes.set_active_scalars("Elevation")  # Probably not necessary, but just in case

# Create axes
axis_length = 20
tip_length = 0.25 / axis_length * 3
tip_radius = 0.1 / axis_length * 3
shaft_radius = 0.05 / axis_length * 3
x_axis = pv.Arrow(
    direction=(axis_length, 0, 0),
    tip_length=tip_length,
    tip_radius=tip_radius,
    shaft_radius=shaft_radius,
    scale="auto",
)
y_axis = pv.Arrow(
    direction=(0, axis_length, 0),
    tip_length=tip_length,
    tip_radius=tip_radius,
    shaft_radius=shaft_radius,
    scale="auto",
)
z_axis = pv.Arrow(
    direction=(0, 0, axis_length),
    tip_length=tip_length,
    tip_radius=tip_radius,
    shaft_radius=shaft_radius,
    scale="auto",
)
x_label = pv.PolyData([axis_length, 0, 0])
y_label = pv.PolyData([0, axis_length, 0])
z_label = pv.PolyData([0, 0, axis_length])
x_label.point_data["label"] = [
    "x",
]
y_label.point_data["label"] = [
    "y",
]
z_label.point_data["label"] = [
    "z",
]


# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the elapsed time in seconds
print(f"Elapsed Time: {elapsed_time:.2f} seconds")

if elapsed_time < 2:
    print('Test run took less than 2 seconds ==> Test passed')
else:
    print('Test run took more than 2 seconds ==> Test failed')
# Plot results
p = pv.Plotter()
p.add_mesh(x_axis, color="r")
p.add_point_labels(x_label, "label", show_points=False, font_size=24)
p.add_mesh(y_axis, color="r")
p.add_point_labels(y_label, "label", show_points=False, font_size=24)
p.add_mesh(z_axis, color="r")
p.add_point_labels(z_label, "label", show_points=False, font_size=24)
p.add_mesh(data)
p.add_mesh(planes)
p.show()