from typing import Tuple, List
from cv2 import Mat
import cv2
import os
from os import path
from configs import StageName

def combine(image_folder_path: str, output_path: str, fps: float=24.0, img_type='jpg'):
    '''
    focus on COMBINE images only

    image_folder_path = 'long_path/3-mobrecon/0_stone'
    output_path       = 'long_path/4v-mobrecon_video/0_stone.mp4'
    '''
    print(f'\ncombine images')
    print(f'from: {image_folder_path}')
    print(f'  to: {output_path}')

    image = cv2.imread(path.join(image_folder_path, f'0000_plot.{img_type}'))  # to be edited due to different methods
    h, w = image.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for filename in os.listdir(image_folder_path):
        if not filename.endswith(img_type):  # 3D file or other
            continue
        # img files
        full_path = os.path.join(image_folder_path, filename)
        image = cv2.imread(full_path)
        out.write(image)
    
    out.release()

def combine_video(inference_dataset_name: str, method: str, video_folder: str, skip: int=0):
    '''
    help find corresponding path, fpses
    support of combine()
    '''
    image_folder_path = os.path.join(
        inference_dataset_name,
        f'3-{method}',
        video_folder
    )   # infer_HandMotion / 3-mobrecon / [0_stone] / .jpg(s)

    output_folder = os.path.join(
        inference_dataset_name,
        f'4v-{method}_video'
    )
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(
        output_folder,
        f'{video_folder}.mp4'
    )   # infer_HandMotion / 4-mobrecon_video / [0_stone].mp4

    source_video_path = os.path.join(
        inference_dataset_name,
        StageName[0],
        f'{video_folder}.mp4'
    )
    fps = 24 / (skip+1)
    if not path.isfile(source_video_path):
        print(f'[Warning]: no such source video: {source_video_path}')
        print(f'           use default fps: 24')
    else:
        source_video = cv2.VideoCapture(source_video_path)
        fps = source_video.get(cv2.CAP_PROP_FPS)

    combine(image_folder_path, output_path, fps)

if __name__ == "__main__":
    # 3-{method} to 4v-{method}_video
    inference_dataset_name = 'infer_Hand5'
    video_folder = [
        'hand_1',
        'hand_2',
    ]
    method = 'mobrecon'

    for Select in range(len(video_folder)):
        combine_video(
            inference_dataset_name=inference_dataset_name,
            method=method,
            video_folder=video_folder[Select]
        )  # infer_HandMotion / 3-mobrecon / [0_stone]
