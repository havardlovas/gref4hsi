import pyvista as pv
from pyvistaqt import BackgroundPlotter
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
# A simple visualization of various types of data

def show_mesh_camera(config):
    mesh_path = config['General']['modelPath']
    texture_path = config['General']['texPath']
    pose_path = config['General']['posePath']
    offsetX = float(config['General']['offsetX'])
    offsetY = float(config['General']['offsetY'])
    offsetZ = float(config['General']['offsetZ'])


    pose = pd.read_csv(
        pose_path, sep=',',
        header=0)

    points_cam = np.concatenate( (pose[" X"].values.reshape((-1,1)) - offsetX, pose[" Y"].values.reshape((-1,1)) - offsetY, pose[" Z"].values.reshape((-1,1)) - offsetZ), axis = 1)
    use_local = False
    if use_local == True:
        eul_cam = np.concatenate((pose[" Yaw"].values.reshape((-1, 1)),
                                  pose[" Pitch"].values.reshape((-1, 1)),
                                  pose[" Roll"].values.reshape((-1, 1))), axis=1)
    else:
        eul_cam = np.concatenate( (pose[" RotZ"].values.reshape((-1,1)), pose[" RotY"].values.reshape((-1,1)), pose[" RotX"].values.reshape((-1,1))), axis = 1)


    r_scipy = Rotation.from_euler("ZYX", eul_cam, degrees=True)
    rotMats = r_scipy.as_matrix()

    #cam_rot = Rotation.from_euler("ZYX", np.array([0, 0, 0]), degrees=True).as_matrix()

    # Compose the two
    #rotMats = rotMats*cam_rot
    p = BackgroundPlotter(window_size=(600, 400))
    mesh = pv.read(mesh_path)
    if texture_path != 'NONE':
        tex = pv.read_texture(texture_path)
        p.add_mesh(mesh, texture=tex)
    else:
        p.add_mesh(mesh)
    p.add_points(points_cam, render_points_as_spheres=True,
                      point_size=1)
    directionX = rotMats[:, :, 0]
    directionY = rotMats[:, :, 1]
    directionZ = rotMats[:, :, 2]
    step = 10
    p.add_arrows(points_cam[::step], directionX[::step], mag = 0.25, color= 'red')
    p.add_arrows(points_cam[::step], directionY[::step], mag=0.25, color= 'green')
    p.add_arrows(points_cam[::step], directionZ[::step], mag=0.25, color= 'blue')
    p.show()
    p.app.exec_()

def show_camera_geometry(CameraGeometry, config):
    mesh_path = config['General']['modelPath']
    texture_path = config['General']['texPath']




    rotMats = CameraGeometry.RotationHSI.as_matrix()

    points_cam = CameraGeometry.PositionHSI


    mesh = pv.read(mesh_path)
    tex = pv.read_texture(texture_path)

    p = BackgroundPlotter(window_size=(600, 400))

    p.add_mesh(mesh, texture=tex)
    p.add_points(points_cam, render_points_as_spheres=True,
                      point_size=1)
    directionX = rotMats[:, :, 0]
    directionY = rotMats[:, :, 1]
    directionZ = rotMats[:, :, 2]
    step = 10
    p.add_arrows(points_cam[::step], directionX[::step], mag = 0.25, color= 'red')
    p.add_arrows(points_cam[::step], directionY[::step], mag=0.25, color= 'green')
    p.add_arrows(points_cam[::step], directionZ[::step], mag=0.25, color= 'blue')


    # Add direction vectors in cyan to show FOV
    direction1 = CameraGeometry.rayDirectionsGlobal[:, 0, :]
    direction2 = CameraGeometry.rayDirectionsGlobal[:, -1, :]

    p.add_arrows(points_cam[::step], direction1[::step], mag=1, color='cyan')
    p.add_arrows(points_cam[::step], direction2[::step], mag=1, color='yellow')

    p.show()
    p.app.exec_()

def show_projected_hsi_points(HSICameraGeometry, config, transect_string):
    mesh_path = config['General']['modelPath']
    texture_path = config['General']['texPath']

    point_cloud_path = config['Georeferencing']['rgbPointCloudPath'] + transect_string + '.ply'




    rotMats = HSICameraGeometry.RotationHSI.as_matrix()
    points_cam = HSICameraGeometry.PositionHSI

    mesh = pv.read(mesh_path)
    tex = pv.read_texture(texture_path)

    p = BackgroundPlotter(window_size=(600, 400))

    p.add_mesh(mesh)
    p.add_points(points_cam, render_points_as_spheres=True,
                 point_size=2, color = 'black')
    directionX = rotMats[:, :, 0]
    directionY = rotMats[:, :, 1]
    directionZ = rotMats[:, :, 2]
    step = 100
    p.add_arrows(points_cam[::step], directionX[::step], mag=0.25, color='red')
    p.add_arrows(points_cam[::step], directionY[::step], mag=0.25, color='green')
    p.add_arrows(points_cam[::step], directionZ[::step], mag=0.25, color='blue')

    # Add the points from intersections
    #points_intersection = HSICameraGeometry.projection[::step, ::step, :].reshape((-1,3))

    #p.add_points(points_intersection, render_points_as_spheres=True,
    #             point_size=1)


    pcd = o3d.io.read_point_cloud(point_cloud_path)
    color_arr = np.asarray(pcd.colors)
    hyp_pcl = pv.read(point_cloud_path)
    hyp_pcl['colors'] = color_arr
    p.add_mesh(hyp_pcl, scalars='colors', rgb=True, point_size=2)

    p.show()
    p.app.exec_()

def show_point_clouds(pathPcl1, pathPcl2, ind1 = None, ind2 = None, hyp1 = None, hyp2 = None):

    if ind1 == None:
        p = BackgroundPlotter(window_size=(600, 400))
        pcd = o3d.io.read_point_cloud(pathPcl1)
        color_arr = np.asarray(pcd.colors)
        p.set_background(color='white', top=None)
        #color_arr[:, 0] = (color_arr[:, 0] - color_arr[:, 0].min()) / (color_arr[:, 0].max() - color_arr[:, 0].min())
        #color_arr[:, 1] = (color_arr[:, 1] - color_arr[:, 1].min()) / (color_arr[:, 1].max() - color_arr[:, 1].min())
        #color_arr[:, 2] = (color_arr[:, 2] - color_arr[:, 2].min()) / (color_arr[:, 2].max() - color_arr[:, 2].min())
        hyp_pcl = pv.read(pathPcl1)
        hyp_pcl['colors'] = color_arr
        p.add_mesh(hyp_pcl, scalars='colors', rgb=True, point_size=2)

        pcd = o3d.io.read_point_cloud(pathPcl2)
        color_arr = np.asarray(pcd.colors)


        #color_arr[:, 0] = (color_arr[:, 0] - color_arr[:, 0].min()) / (color_arr[:, 0].max() - color_arr[:, 0].min())
        #color_arr[:, 1] = (color_arr[:, 1] - color_arr[:, 1].min()) / (color_arr[:, 1].max() - color_arr[:, 1].min())
        #color_arr[:, 2] = (color_arr[:, 2] - color_arr[:, 2].min()) / (color_arr[:, 2].max() - color_arr[:, 2].min())

        hyp_pcl2 = pv.read(pathPcl2)
        hyp_pcl2['colors'] = color_arr
        p.add_mesh(hyp_pcl2, scalars='colors', rgb=True, point_size=2)

        p.add_points(hyp1.position_hsi, render_points_as_spheres=True,
                     point_size=1)
        p.add_points(hyp2.position_hsi, render_points_as_spheres=True,
                     point_size=1)

        p.show()
        p.app.exec_()
    else:
        p = BackgroundPlotter(window_size=(600, 400))
        pcd = o3d.io.read_point_cloud(pathPcl1)
        color_arr = np.asarray(pcd.colors)
        hyp_pcl = pv.read(pathPcl1)
        hyp_pcl['colors'] = color_arr
        p.add_mesh(hyp_pcl, scalars='colors', rgb=True, point_size=2)

        pcd = o3d.io.read_point_cloud(pathPcl2)
        color_arr = np.asarray(pcd.colors)
        hyp_pcl2 = pv.read(pathPcl2)
        hyp_pcl2['colors'] = color_arr
        p.add_mesh(hyp_pcl2, scalars='colors', rgb=True, point_size=2)

        p.show()
        p.app.exec_()




def main():
    show_mesh_camera()




if __name__ == '__main__':
    main()


