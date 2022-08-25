from typing import Tuple
from cv2 import Mat
import cv2
import os
from os import path
from configs import StageName

# Find good bbox
def find_video_fitting_bbox(inference_dataset_name, video_folder, image_name):
    '''
        cv2.imread(path.join('infer_HandMotion', StageName[1], '0_stone', '0000.jpg'))
    '''
    image = cv2.imread(path.join(inference_dataset_name, StageName[1], video_folder, image_name))
    cv2.imshow('Result', image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    bbox = [50, 50, 100, 100]  # init
    bbox = find_fitting_bbox(image, bbox)

    return bbox

def find_fitting_bbox(image: Mat, bbox: Tuple[int, int, int, int]):
    print('input instructions: [ inc | dec ] [ u | d | l | r ] [ pixel counts ]')
    print('                                   up down left right')
    print('              type ok, to finish')

    img_size = image.shape  # [height, width]

    instruction = input()
    while instruction != 'ok':
        bbox = update_bbox(img_size, bbox, instruction)
        print(f'update bbox to: {bbox}')
        test_image = image.copy()
        test_show_bbox(test_image, bbox)
        instruction = input()

    return bbox

def test_show_bbox(image: Mat, bbox: Tuple[int, int, int, int]):
    '''
    bbox: [p1x, p1y, p2x, p2y]

         p1x    p2x
        *------*
    p1y |p1    |
        |      |
        *------*
    p2y         p2
    '''
    Red = (0, 0, 255)

    p1 = bbox[:2]  # p1x, p1y
    p2 = bbox[2:]  # p2x, p2y

    cv2.rectangle(image, p1, p2, Red, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'W: {bbox[2]-bbox[0]}, H: {bbox[3]-bbox[1]}'
    cv2.putText(image, text, p1, font, 1, (0,0,0), 2)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def update_bbox(img_size: Tuple[int, int], bbox: Tuple[int, int, int, int], instruction: str) -> Tuple[int, int, int, int]:
    '''
    bbox:
        [p1x, p1y, p2x, p2y]
    img_size:
        [ height, width ]
    instruction:
      - [ inc | dec ] [ u | d | l | r ] [ pixel counts ]
                      up down left right

    return bbox
    '''
    inst = instruction.split()
    pixels = int(inst[2])

    symbol = 1
    if inst[0] == 'inc':
        symbol = 1
    elif inst[0] == 'dec':
        symbol = -1

    if inst[1] == 'u':  # upper, not 'p1y += pixels'
        bbox[1] -= symbol * pixels
        bbox[1] = max(bbox[1], 0)
    elif inst[1] == 'd':  # lower
        bbox[3] += symbol * pixels
        bbox[3] = min(bbox[3], img_size[0])
    elif inst[1] == 'l':
        bbox[0] -= symbol * pixels
        bbox[0] = max(bbox[0], 0)
    elif inst[1] == 'r':
        bbox[2] += symbol * pixels
        bbox[2] = min(bbox[2], img_size[1])

    return bbox

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
        full_path = path.join(image_folder, filename)
        output_path = path.join(output_folder, filename)
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


if __name__ == '__main__':
    # image = cv2.imread(path.join('infer_HandMotion', StageName[1], '0_stone', '0000.jpg'))
    # bbox = [50, 50, 100, 100]  # init
    # find_fitting_bbox(image, bbox)

    # 1-full_images to 2-boxed_images
    inference_dataset_name = 'infer_Hand5'
    video_folder = [
        'hand_1',
        'hand_2',
    ]
    for select in range(len(video_folder)):
        # select = 2

        bbox = find_video_fitting_bbox(
            inference_dataset_name = inference_dataset_name,
            video_folder = video_folder[select],
            image_name = '0000.jpg'
        )

        image_folder = path.join(inference_dataset_name, StageName[1], video_folder[select])
        boxing_folder(image_folder, bbox, img_type='jpg')


