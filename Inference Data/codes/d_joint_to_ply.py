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

def create_plys(inference_dataset_name: str, method: str, video_folder: str):
    '''
    transformer all .npy file in: os.path.join(
        inference_dataset_name,
        '3-{method}',
        video_folder,
        'TO_BE_PROCESS.npy'
    )
    to
    os.path.join(
        inference_dataset_name,
        '4j-{method}_joint',
        video_folder,
        'TO_BE_PROCESS.ply'
    )
    '''
    input_folder = os.path.join(
        inference_dataset_name,
        f'3-{method}',
        video_folder
    )
    output_folder = os.path.join(
        inference_dataset_name,
        f'4j-{method}_joint',
        video_folder
    )
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith('.npy'):
            continue

        fullpath = os.path.join(input_folder, filename)

        out_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '.ply')
        joint2ply(fullpath, out_path)


def joint2ply(input_path: str, output_path: str):
    '''
    output data.shape = (joint_counts, 3)
    '''
    joint_xyz = np.load(input_path)
    # joint_xyz[9:] = 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(joint_xyz)
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)

def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # 3-{method} to 4j-{method}_joint
    inference_dataset_name = 'infer_Viewpoints_2'
    video_folder = [
        'view_1',
        'view_2',
        'view_3',
    ]
    method = 'mobrecon'

    # create_plys(
    #     inference_dataset_name=inference_dataset_name,
    #     method=method,
    #     video_folder=video_folder[0],
    #     # skip=135
    # )
    for folder in video_folder:
        create_plys(
            inference_dataset_name=inference_dataset_name,
            method=method,
            video_folder=folder
        )  # infer_HandMotion / 3-mobrecon / [0_stone]
    # show_ply(r'infer_HandMotion\\4j-mobrecon_joint\\0_stone\\0000_xyz.ply')
