import pyvista as pv
from pyvistaqt import BackgroundPlotter
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import pymap3d as pm
# A simple visualization of various types of data
from scripts.geometry import rotation_matrix_ecef2ned, rotation_matrix_ecef2enu

def show_mesh_camera(config, show_mesh = True, show_pose = True, ref_frame = 'ECEF'):
    """
    # Reads the mesh file and pose info, and plots the trajectory next to the mesh.

    :param config: Dictionary-like (configuration) object
    :return: Nothing

    """


    # Todo: encode show mesh camera to use h5-embedded data? Or is this a loss of performance?
    mesh_path = config['Absolute Paths']['modelPath']
    try:
        texture_path = config['Absolute Paths']['texPath']
    except KeyError:
        texture_path = None
        
    pose_path = config['Absolute Paths']['posePath']

    # Offsets used for plotting
    offsetX = float(config['General']['offsetX'])
    offsetY = float(config['General']['offsetY'])
    offsetZ = float(config['General']['offsetZ'])


    pose = pd.read_csv(
        pose_path, sep=',',
        header=0)

    points_cam_ecef = np.concatenate( (pose[" X"].values.reshape((-1,1)) - offsetX, pose[" Y"].values.reshape((-1,1)) - offsetY, pose[" Z"].values.reshape((-1,1)) - offsetZ), axis = 1)
    use_local = False
    if use_local == True:
        eul_cam = np.concatenate((pose[" Yaw"].values.reshape((-1, 1)),
                                  pose[" Pitch"].values.reshape((-1, 1)),
                                  pose[" Roll"].values.reshape((-1, 1))), axis=1)
    else:
        eul_cam = np.concatenate( (pose[" RotZ"].values.reshape((-1,1)), pose[" RotY"].values.reshape((-1,1)), pose[" RotX"].values.reshape((-1,1))), axis = 1)


    R_body_to_ecef = Rotation.from_euler("ZYX", eul_cam, degrees=True)
    

    # Read the mesh
    mesh = pv.read(mesh_path)
    if ref_frame == 'NED':
        x = offsetX
        y = offsetY
        z = offsetZ
        lat0, lon0, hei0 = pm.ecef2geodetic(x, y, z, deg=True)
        R_ecef_to_ned = Rotation.from_matrix(rotation_matrix_ecef2ned(lon=lon0, lat=lat0))

        R_body_to_ned = R_ecef_to_ned*R_body_to_ecef
        rotMats = R_body_to_ned.as_matrix()

        points_mesh_ecef = mesh.points

        # Next part is to make a NED
        x_cam, y_cam, z_cam = pm.ecef2ned(x = points_cam_ecef[:,0] + x, y = points_cam_ecef[:,1] + y, z = points_cam_ecef[:,2] + z, lon0=lon0, lat0=lat0, h0=hei0)

        points_cam = np.concatenate((x_cam.reshape((-1,1)), y_cam.reshape((-1,1)), z_cam.reshape((-1,1))), axis = 1)

        x_mesh, y_mesh, z_mesh = pm.ecef2ned(x = points_mesh_ecef[:,0] + x, y = points_mesh_ecef[:,1] + y, z = points_mesh_ecef[:,2] + z, lon0=lon0, lat0=lat0, h0=hei0)

        points_mesh = np.concatenate((x_mesh.reshape((-1,1)), y_mesh.reshape((-1,1)), z_mesh.reshape((-1,1))), axis = 1)
        mesh.points = points_mesh
    elif ref_frame == 'ENU':
        x = offsetX
        y = offsetY
        z = offsetZ
        lat0, lon0, hei0 = pm.ecef2geodetic(x, y, z, deg=True)
        R_ecef_to_enu = Rotation.from_matrix(rotation_matrix_ecef2enu(lon=lon0, lat=lat0))

        R_body_to_enu = R_ecef_to_enu*R_body_to_ecef
        rotMats = R_body_to_enu.as_matrix()

        points_mesh_ecef = mesh.points

        # Next part is to make a NED
        x_cam, y_cam, z_cam = pm.ecef2enu(x = points_cam_ecef[:,0] + x, y = points_cam_ecef[:,1] + y, z = points_cam_ecef[:,2] + z, lon0=lon0, lat0=lat0, h0=hei0)

        points_cam = np.concatenate((x_cam.reshape((-1,1)), y_cam.reshape((-1,1)), z_cam.reshape((-1,1))), axis = 1)

        x_mesh, y_mesh, z_mesh = pm.ecef2enu(x = points_mesh_ecef[:,0] + x, y = points_mesh_ecef[:,1] + y, z = points_mesh_ecef[:,2] + z, lon0=lon0, lat0=lat0, h0=hei0)

        points_mesh = np.concatenate((x_mesh.reshape((-1,1)), y_mesh.reshape((-1,1)), z_mesh.reshape((-1,1))), axis = 1)
        mesh.points = points_mesh
    
    else:
        rotMats = R_body_to_ecef.as_matrix()
        points_cam = points_cam_ecef


    

    #cam_rot = Rotation.from_euler("ZYX", np.array([0, 0, 0]), degrees=True).as_matrix()

    # Compose the two
    #rotMats = rotMats*cam_rot
    p = BackgroundPlotter(window_size=(600, 400))
    
    if show_mesh:
        if texture_path != None:
            tex = pv.read_texture(texture_path)
            p.add_mesh(mesh, texture=tex)
        else:
            p.add_mesh(mesh)

    if show_pose:
        p.add_points(points_cam, render_points_as_spheres=True,
                        point_size=1)
        directionX = rotMats[:, :, 0]
        directionY = rotMats[:, :, 1]
        directionZ = rotMats[:, :, 2]
        step = 10
        scale = np.linalg.norm(np.max(points_cam, axis = 0)-np.min(points_cam, axis = 0), axis=0)
        p.add_arrows(points_cam[::step], directionX[::step], mag = 0.1*scale, color = 'red')
        p.add_arrows(points_cam[::step], directionY[::step], mag=0.1*scale, color= 'green')
        p.add_arrows(points_cam[::step], directionZ[::step], mag=0.1*scale, color= 'blue')
    p.show()
    p.app.exec_()

def show_camera_geometry(CameraGeometry, config):
    mesh_path = config['General']['modelPath']
    texture_path = config['General']['texPath']




    rotMats = CameraGeometry.rotation_hsi.as_matrix()

    points_cam = CameraGeometry.position_ecef

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
    mesh_path = config['Absolute Paths']['modelPath']
    
    point_cloud_path = config['Absolute Paths']['rgbPointCloudPath'] + transect_string + '.ply'

    rotMats = HSICameraGeometry.rotation_hsi.as_matrix()
    points_cam = HSICameraGeometry.position_ecef

    mesh = pv.read(mesh_path)

    try:
        texture_path = config['Absolute Paths']['texPath']
        tex = pv.read_texture(texture_path)
    except:
        pass
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


