import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import mediapipe as mp

def print_result(func):
    # print('herehere: ', args)
    def warped(*args):
        if len(args) == 2:
            print(f'{func.__name__}: {_to_str_bbox(args[0])} -> ', end='')
        else:
            print(f'{func.__name__}: {_to_str_bbox(*args)} -> ', end='')

        result = func(*args)
        print(f'{_to_str_bbox(result)}')

        return result

    return warped


def _to_str_bbox(bbox):
    return f'[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]'

def _scale_to_pixel(bbox, shape):
    ''' bbox: [left, top, right down] in (0, 1)
        shape: (height, width) of image

    return:
        bbox: [left, top, right down] in pixel( float )
    '''
    return [
        shape[1] * bbox[0],
        shape[0] * bbox[1],
        shape[1] * bbox[2],
        shape[0] * bbox[3],
    ]

def _box_to_square(bbox):
    ''' bbox: [left, top, right down] in pixel( float )
    return
        bbox: [left, top, W, W]
    '''
    center = [(bbox[0]+bbox[2]) / 2, (bbox[1]+bbox[3]) / 2]
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
    return bbox

def find_bbox(folder,
               show_joint_3d=False, show_joint_2d=False,
               bbox_detection=True, show_bbox=False,
               show_square_bbox=False, ret_square_bbox=False
               ):
    '''
    return bbox[left, up, w, w] in int, if ret_square_bbox
    '''
    assert not show_bbox or bbox_detection, 'show_bbox -> bbox_detection'
    assert not show_square_bbox  or bbox_detection, 'show_square_bbox -> bbox_detection'
    assert not ret_square_bbox  or bbox_detection, 'ret_square_bbox -> bbox_detection'

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For static images:
    IMAGE_FILES = [os.path.join(folder, file) for file in os.listdir(folder)]

    max_bbox = [1.0, 1.0, 0.0, 0.0] # in (0, 1), (left, top ,right, down)

    # mp_hands accept left-hand only
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            # print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            if show_joint_2d:
                for hand_landmarks in results.multi_hand_landmarks:
                    # hand_landmarks: 21 joints xyz
                    # method to get xyz:
                    # print(
                    #     f'Index finger tip coordinates: (',
                    #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    # )
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            if show_joint_3d:
                # 3d visualize
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    mp_drawing.plot_landmarks(
                        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

            if bbox_detection:
                for hand_landmarks in results.multi_hand_landmarks:
                    if len(results.multi_hand_landmarks) > 1:
                        print(f'image: {idx} 2 hands')  # 2 hands... troublesome

                    for joint in hand_landmarks.landmark:
                        # print(joint.x, joint.y)
                        max_bbox[0] = min(max_bbox[0], joint.x)
                        max_bbox[1] = min(max_bbox[1], joint.y)
                        max_bbox[2] = max(max_bbox[2], joint.x)
                        max_bbox[3] = max(max_bbox[3], joint.y)

            if show_bbox:
                # _draw_bbox(annotated_image, 
                bbox = [
                    int(image_width  * max_bbox[0]),
                    int(image_height * max_bbox[1]),
                    int(image_width  * max_bbox[2]),
                    int(image_height * max_bbox[3]),
                ]
                cv2.rectangle(annotated_image, bbox[:2], bbox[2:], (255, 0, 0), 4)

            if show_joint_2d or show_bbox:
                # annot on original image
                cv2.imshow('title', cv2.flip(annotated_image, 1))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # max_bbox may out of (0, 1)
        max_bbox[0] = max(max_bbox[0], 0)
        max_bbox[1] = max(max_bbox[1], 0)
        max_bbox[2] = min(max_bbox[2], 1)
        max_bbox[3] = min(max_bbox[3], 1)
    # End of mp_hands solution

    if bbox_detection:
        bbox = max_bbox[:]
        # bbox flip along y-axis
        bbox[0], bbox[2] = 1 - bbox[2], 1 - bbox[0]
        # bbox to pixel
        bbox = _scale_to_pixel(bbox, (image_height, image_width))
        # bbox to square
        bbox = _box_to_square(bbox)  # bbox = [left, up, W, W]
        # bbox scale up 1.4 times
        append_w = bbox[2] * 0.4
        bbox[1] -= append_w / 2
        bbox[0] -= append_w / 2
        bbox[2] += append_w
        bbox[3] += append_w
        # bbox-> [left, top, width, height], may contain negative

        # output check
        if show_square_bbox:
            print(f'final bbox: {bbox}, [left, up, w, w]')
            print(f'in (H, W) == {(image_height, image_width)}')

            bbox = [int(e) for e in bbox]
            for file in IMAGE_FILES:
                image = cv2.imread(file)
                cv2.rectangle(image, bbox[:2], (bbox[0]+bbox[2], bbox[1]+bbox[2]), (255, 0, 0), 4)
                cv2.imshow('title', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        if ret_square_bbox:
            return [int(e) for e in bbox]

''' Copied from b_boxing_images.py '''
from typing import Tuple
from cv2 import Mat
import numpy as np
from configs import StageName

# bboxing the images, stage1 to stage2
def boxing_folder(image_folder: str, bbox: Tuple[int, int, int, int], img_type: str='jpg'):
    '''
    output_folder = image_folder.replace(stage1, stage2)
    bbox = [p1x, p1y, p2x, p2y]
    '''
    output_folder = image_folder.replace(StageName[1], StageName[2])
    os.makedirs(output_folder, exist_ok=True)
    print(f'boxing image with bbox: {bbox}')
    print(f'from: {image_folder}')
    print(f'  to: {output_folder}')

    for filename in os.listdir(image_folder):
        full_path = os.path.join(image_folder, filename)
        output_path = os.path.join(output_folder, filename)
        output_path = output_path.rsplit('.', 1)[0] + f'.{img_type}'
        # print(full_path, output_path, sep='\n')

        image = cv2.imread(full_path)
        image = boxing(image, bbox)
        cv2.imwrite(output_path, image)

''' edited from b_boxing_images, change to affine transform '''
def boxing(image: Mat, bbox: Tuple[int, int], size: Tuple[int, int]=(256, 256)) -> Mat:
    '''
    boxing the image
    and resize to {size}

    bbox: [left, up, w, w] in int
    return the image
    ---
    Affine transformer
    p1 ---- p2
    |       |
    |  Pic  |
    |       |
    p3 ---- .
    '''
    # Affine form of transformation
    p1x, p1y, w, _ = bbox
    # [[x, y]]
    tgt = np.array([
        [0      , 0],
        [size[0], 0],
        [0      , size[0]],
    ])
    src = np.array([
        [p1x    , p1y],
        [p1x + w, p1y],
        [p1x    , p1y + w]
    ])
    trans = cv2.getAffineTransform(np.float32(src), np.float32(tgt)).astype(np.float32)
    img_patch = cv2.warpAffine(image, trans, (256, 256), flags=cv2.INTER_LINEAR)
    # cv2.imshow('Result', img_patch)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return img_patch
''' end of Copied from b_boxing_images.py '''

if __name__ == '__main__':
    # folder = os.path.join('mediapipe_test', '1-full_images', '01M')
    # exp([os.path.join(folder, file) for file in os.listdir(folder)])
    inference_dataset_name = 'test'
    video_folder = [
        '2R',
    ]
    for select in range(len(video_folder)):
        # select = 2
        image_folder = os.path.join(inference_dataset_name, StageName[1], video_folder[select])

        # bbox = [left, up, w, w]
        bbox = find_bbox(
            folder=image_folder,
            show_joint_3d=False,
            show_joint_2d=False,
            bbox_detection=True,
            show_bbox=False,
            show_square_bbox=False,
            ret_square_bbox=True,
        )

        boxing_folder(image_folder, bbox, img_type='jpg')

