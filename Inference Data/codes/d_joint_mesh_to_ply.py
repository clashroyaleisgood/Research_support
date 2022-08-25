from telnetlib import SE
import cv2
import numpy as np
import open3d as o3d
import os
from os import path

# from HandMesh/cmr/datasets/FreiHAND/kinematics.py
class MPIIHandJoints:
    n_joints = 21

    labels = [
        'W', #0
        'T0', 'T1', 'T2', 'T3', #4
        'I0', 'I1', 'I2', 'I3', #8
        'M0', 'M1', 'M2', 'M3', #12
        'R0', 'R1', 'R2', 'R3', #16
        'L0', 'L1', 'L2', 'L3', #20
    ] # 掌根、拇指(thumb)、食指(index)、
      #中指(mid)、無名指(ring)、小指(little)

    parents = [
        None,
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19
    ]

def append_plys(inference_dataset_name: str, method: str, video_folder: str, skip=0):
    '''
    read all joint and mesh file in inference_dataset_name / [joint or mesh] / video_folder

    and visualize the combination

    skip to nth mesh and joint
    '''

    input_joint_folder = os.path.join(
        inference_dataset_name,
        f'4j-{method}_joint',
        video_folder
    )
    input_mesh_folder = os.path.join(
        inference_dataset_name,
        f'3-{method}',
        video_folder
    )

    for filename_j in os.listdir(input_joint_folder)[skip:]:
        # if not filename.endswith('.ply'):
        #     continue

        fullpath_joint = os.path.join(input_joint_folder, filename_j)

        filename_m = filename_j.split('_')[0] + '_mesh.ply'
        fullpath_mesh  = os.path.join(input_mesh_folder, filename_m)

        vis_comb_mesh_joint(fullpath_joint, fullpath_mesh)


def vis_comb_mesh_joint(input_path_joint: str, input_path_mesh: str):
    '''
    visualize combination of hand joints and mesh
    in {input_path_joint} and {input_path_mesh}
    '''
    pcd_joint = o3d.io.read_point_cloud(input_path_joint)
    pcd_mesh = o3d.io.read_point_cloud(input_path_mesh)
    # print(f'mesh get {len(pcd_mesh.points)} points')  # 778
    colors = np.zeros((len(pcd_mesh.points), 3))
    colors[:] = 0.9
    # print(colors)

    pcd_mesh.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd_joint, pcd_mesh])

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(joint_xyz)
    # o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)

def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Visualize results in
    # 3-{method} and 4j-{method}_joint

    inference_dataset_name = 'infer_Viewpoints_2'
    video_folder = [
        'view_1',
        'view_2',
        'view_3',
    ]
    method = 'mobrecon'
    # append_plys(
    #     inference_dataset_name=inference_dataset_name,
    #     method=method,
    #     video_folder=video_folder[0],
    #     skip=72
    # )
    for folder in video_folder:
        append_plys(
            inference_dataset_name=inference_dataset_name,
            method=method,
            video_folder=folder,
            skip=50
        )  # infer_HandMotion / 3-mobrecon        / [0_stone]
           # infer_HandMotion / 4j-mobrecon_joint / [0_stone]
    # show_ply(r'infer_HandMotion\\4j-mobrecon_joint\\0_stone\\0000_xyz.ply')
