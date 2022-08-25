import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

image_row = image_col = 100

def save_ply(Z, filepath):
    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row*image_col, 3), dtype=np.float32)
    # let all point float on a base plane
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)

def joint2ply(input_path, output_path):
    arr = np.load(input_path)
    
    pass

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    inference_dataset_name = 'infer_HandMotion'
    method = '3-mobrecon'
    video_folder = [
        '0_stone',
        '1_good',
        '2_ya'
    ]
    filepath = os.path.join(
        inference_dataset_name,
        method,
        video_folder[0],
        '0000_mesh.ply'
    )
    show_ply(filepath)