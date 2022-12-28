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


def _draw_bbox(image, bbox):
    ''' bbox: [left, top, right down] pixels
    '''
    cv2.rectangle(image, bbox[:2], bbox[2:], (255, 0, 0), 4)
    cv2.imshow('My Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# @print_result
def _square_bbox(bbox, shape):
    ''' bbox: [left, top, right down] in (0, 1)
        return bbox, width == height
        region may be {<0} or {>1}
    '''
    img_h, img_w = shape

    w = (bbox[2] - bbox[0]) * img_w  # 0.3
    h = (bbox[3] - bbox[1]) * img_h  # 0.3

    if w > h:
        dist = (w-h) / 2  # in pixel
        bbox[1] -= dist / img_h
        bbox[3] += dist / img_h
    elif h > w:
        dist = (h-w) / 2  # in pixel
        bbox[0] -= dist / img_w
        bbox[2] += dist / img_w

    return bbox

# @print_result
def _expand_bbox(bbox, percentage):
    ''' bbox: [left, top, right down] in (-1, 2)
        return bbox[left -?, top-?, ...]
    '''
    ret_bbox = [0, 0, 0, 0]
    ret_bbox[0] = bbox[0] - percentage
    ret_bbox[1] = bbox[1] - percentage
    ret_bbox[2] = bbox[2] + percentage
    ret_bbox[3] = bbox[3] + percentage

    return ret_bbox

# @print_result
def _validate_bbox(bbox):
    ''' bbox: [left, top, right down] in (-1, 2)
        平移
        return in (0, 1)
    '''
    if bbox[0] < 0:
        bbox[2] -= bbox[0]
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[3] -= bbox[1]
        bbox[1] = 0
    if bbox[2] > 1:
        move = bbox[2] - 1
        bbox[2] -= move
        bbox[0] -= move
    if bbox[3] > 1:
        move = bbox[3] - 1
        bbox[1] -= move
        bbox[3] -= move
    if bbox[0] < 0 or bbox[1] < 0:
        raise ValueError(f'Invalid bbox: {bbox}')

    return bbox


def find_bbox(folder, joint_3d=False, palm_detection=True, mode='show'):
    ''' mode = {'show', 'return'} '''

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For static images:
    IMAGE_FILES = [os.path.join(folder, file) for file in os.listdir(folder)]

    max_bbox = [1.0, 1.0, 0.0, 0.0] # in (0, 1), (left, top ,right, down)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
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
            if joint_3d:
                for hand_landmarks in results.multi_hand_landmarks:
                    print('hand_landmarks:', hand_landmarks)
                    print(
                        f'Index finger tip coordinates: (',
                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    )
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                # annot on original image
                # cv2.imshow('title', cv2.flip(annotated_image, 1))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # 3d visualize
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    mp_drawing.plot_landmarks(
                        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

            if palm_detection:
                for hand_landmarks in results.multi_hand_landmarks:
                    if len(results.multi_hand_landmarks) > 1:
                        print(f'image: {idx} 2 hands')  # 2 hands... troublesome

                    for joint in hand_landmarks.landmark:
                        # print(joint.x, joint.y)
                        max_bbox[0] = min(max_bbox[0], joint.x)
                        max_bbox[1] = min(max_bbox[1], joint.y)
                        max_bbox[2] = max(max_bbox[2], joint.x)
                        max_bbox[3] = max(max_bbox[3], joint.y)
        # max_bbox may out of (0, 1)
        max_bbox[0] = max(max_bbox[0], 0)
        max_bbox[1] = max(max_bbox[1], 0)
        max_bbox[2] = min(max_bbox[2], 1)
        max_bbox[3] = min(max_bbox[3], 1)

    if palm_detection:
        # print(max_bbox)
        max_bbox = _square_bbox(max_bbox, image.shape[:2])
        try:
            max_bbox_exp = _expand_bbox(max_bbox, 0.05)
            max_bbox_exp = _validate_bbox(max_bbox_exp)
        except:
            print('fail at 0,05, re-excecute 0')
            max_bbox_exp = _expand_bbox(max_bbox, 0)
            max_bbox_exp = _validate_bbox(max_bbox_exp)

        image_height, image_width, _ = image.shape
        max_bbox[0] = int(image_width  * max_bbox[0])
        max_bbox[2] = int(image_width  * max_bbox[2])
        max_bbox[1] = int(image_height * max_bbox[1])
        max_bbox[3] = int(image_height * max_bbox[3])
        
        if mode == 'show':
            print('final: ', max_bbox)
            print('width, height:', image_width, image_height)

            for file in IMAGE_FILES:
                image = cv2.imread(file)
                image = cv2.flip(cv2.imread(file), 1)
                _draw_bbox(image, max_bbox)
        elif mode == 'return':
            max_bbox[0], max_bbox[2] = \
                image_width - max_bbox[2], image_width - max_bbox[0]
            return max_bbox

''' Copied from b_boxing_images.py '''
from typing import Tuple
from cv2 import Mat
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

def boxing(image: Mat, bbox: Tuple[int, int], size: Tuple[int, int]=(256, 256)) -> Mat:
    '''
    boxing the image
    and resize to {resize}

    return the image
    '''
    p1x, p1y, p2x, p2y = bbox
    image = image[p1y:p2y, p1x:p2x]
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    # cv2.imshow('Result', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return image
''' end of Copied from b_boxing_images.py '''

if __name__ == '__main__':
    # folder = os.path.join('mediapipe_test', '1-full_images', '01M')
    # exp([os.path.join(folder, file) for file in os.listdir(folder)])
    inference_dataset_name = 'infer_ROM_data'
    video_folder = [
        '29M',
    ]
    for select in range(len(video_folder)):
        # select = 2
        image_folder = os.path.join(inference_dataset_name, StageName[1], video_folder[select])

        bbox = find_bbox(
            folder=image_folder,
            joint_3d=False,
            palm_detection=True,
            mode='return'
        )

        boxing_folder(image_folder, bbox, img_type='jpg')

