import cv2
import os
from os import path
from configs import StageName

def extract(video_path: str, large_folder_path: str, skip, right_hand, img_type: str='jpg') -> None:
    '''
    Extract video from {video_path} to {large_folder_path/video_path/image1, ...}
    large_folder_path/
        basename(video_path)/
            0000.jpg,
            0001.jpg,
            ...,
    '''
    basename = path.basename(video_path).rsplit('.')[0]
    output_full_path = path.join(large_folder_path, basename)
    os.makedirs(output_full_path, exist_ok=True)

    print(f'extract ...')
    print(f'video: {video_path}')
    print(f'   to: {output_full_path}')

    if path.isfile(
        path.join(output_full_path, f'0000.{img_type}')):
        print('file:', path.join(output_full_path, f'0000.{img_type}'))
        ans = input('already exists, cover it? (y/n)')
        if ans != 'y':
            print('[ERROR] user stopped\n')
            return

    vidcap = cv2.VideoCapture(video_path)

    success, image = vidcap.read()
    counter = 0
    while success:
        if counter % 100 == 0:
            print(f'{counter:04d}.{img_type}...>')
        if counter % (skip + 1) == 0:
            if not right_hand:
                image = cv2.flip(image, 1)
            cv2.imwrite(
                path.join(output_full_path, f'{counter:04d}.{img_type}'),
                image
            )
        success, image = vidcap.read()
        counter += 1

    vidcap.release()
    from math import ceil
    print(f'End extracting {int(ceil(counter / (skip+1)))} frames\n')

def extract_one_video(inference_dataset_name, video_name, skip, right_hand):
    '''
    video_path          = path.join(inference_dataset_name, '0-source', video_name)
    output_large_folder = path.join(inference_dataset_name, '1-full_images')
    '''
    video_path = path.join(inference_dataset_name, StageName[0], video_name)
    output_large_folder = path.join(inference_dataset_name, StageName[1])
    extract(video_path, output_large_folder, skip, right_hand)

def extract_videos(inference_dataset_name, video_names, skip, right_hand):
    ''' 
    skip {skip} images, between each result frames
        skip=0, [0, 1, 2, 3, ...]
        skip=1, [0,    2,    4, ...]
        skip=2, [0,       3, ...]
    '''
    for filename in video_names:
        extract_one_video(
            inference_dataset_name,
            filename,
            skip,
            right_hand
        )

if __name__ == '__main__':
    # 0-sourve to 1-full_images
    #   video,      images
    inference_dataset_name = 'test'
    video_names = [
        '2R.mp4',
    ]
    skip = 5 # fps /= (skip+1)

    # video_path = path.join(inference_dataset_name, '0-source', video_names[0])
    # to_folder = path.join(inference_dataset_name, '1-full_images')
    # extract(video_path, to_folder)

    extract_videos(inference_dataset_name, video_names, skip, right_hand=True)
