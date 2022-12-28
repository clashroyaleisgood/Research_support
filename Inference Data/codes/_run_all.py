from configs import StageName
from a_split_videos import extract_videos
from b_mediapipe_auto_find import find_bbox, boxing_folder
from d_calculate_rotation import save_csv
from d_combine_to_video import combine_video

import os


def a_step(inference_dataset_name, video_names, skip):
    extract_videos(inference_dataset_name, video_names, skip)


def b_step(inference_dataset_name, video_folders, resume_at=None):
    for select in range(len(video_folders)):
        if resume_at is not None:
            if video_folders[select] != resume_at:
                continue
            else:
                print(f'resume at: {resume_at}')
                resume_at = None

        image_folder = os.path.join(inference_dataset_name, StageName[1], video_folders[select])

        bbox = find_bbox(
            folder=image_folder,
            joint_3d=False,
            palm_detection=True,
            mode='return'
        )

        boxing_folder(image_folder, bbox, img_type='jpg')


def d_step(inference_dataset_name, video_folders, skip, mode):
    ''' mode = {'vr' ,'v', 'r'}
        'v'ideo, 'r'otation
    '''
    import pandas as pd
    if 'v' in mode:
        for Select in range(len(video_folders)):
            combine_video(
                inference_dataset_name=inference_dataset_name,
                method=method,
                video_folder=video_folders[Select],
                skip=skip
            )  # infer_HandMotion / 3-mobrecon / [0_stone]

    if 'r' in mode:
        overall_series = []
        for Select in range(len(video_folders)):
            min_max_series = save_csv(
                inference_dataset_name=inference_dataset_name,
                method=method,
                video_folder=video_folders[Select],
                return_min_max=True
            )
            overall_series += [min_max_series]
        overall_df = pd.concat(overall_series, axis=1).transpose().round(2)
        overall_df.to_csv(os.path.join(inference_dataset_name, 'overall_rotation.csv'))


if __name__ == '__main__':
    inference_dataset_name = 'infer_ROM_data_2'

    video_names = os.listdir(os.path.join(inference_dataset_name, StageName[0]))
    video_folders = [file.rsplit('.', 1)[0] for file in video_names]
    # print(video_folders)

    # step: a, b
    skip = 5
    # a_step(inference_dataset_name, video_names, skip)
    # b_step(inference_dataset_name, video_folders, resume_at='30U')

    # WAIT step: c
    method = 'mobrecon'

    mode = 'vr'  # video and rotation
    # step: dr, dc
    d_step(inference_dataset_name, video_folders, skip, mode)

