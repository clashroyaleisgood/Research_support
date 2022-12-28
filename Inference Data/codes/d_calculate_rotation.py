import cv2
import numpy as np
import os
from os import path
import pandas as pd
from typing import OrderedDict

# from HandMesh/cmr/datasets/FreiHAND/kinematics.py
class MPIIHandJoints:
    n_joints = 21

    labels = [
        'W', #0
        'T0', 'T1', 'T2', 'T3', #4, T0 是假的，T1 才是真正的 joint
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

class MPIIHandRotation:
    labels = [
        'W', #0
        'T0', 'T1', 'T2', 'T3', #4, T0 是假的，T1 才是真正的 joint
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

    rotations = [
        'T0', 'T1', 'T2',         # rot(W, T0, T1), rot(T0, T1, T2), rot(T1, T2, T3)
        'I0', 'I1', 'I2',   # rot(W, I0, I1), rot(I1, I2, I3), ...
        'M0', 'M1', 'M2',
        'R0', 'R1', 'R2',
        'L0', 'L1', 'L2'
    ]

    # map from medical name to mobrecon joint name
    mapping = OrderedDict({
        '1 IP' : 'T2',                '1 MCP': 'T1',  # thumb
        '2 DIP': 'I2', '2 PIP': 'I1', '2 MCP': 'I0',  # index  finger
        '3 DIP': 'M2', '3 PIP': 'M1', '3 MCP': 'M0',  # middle finger
        '4 DIP': 'R2', '4 PIP': 'R1', '4 MCP': 'R0',  # ring   figner
        '5 DIP': 'L2', '5 PIP': 'L1', '5 MCP': 'L0',  # little finger
    })

    def get_rotation_dict():
        '''
        return {
            'T1': [0, 2, 3],
            ...,
            'I2': [6, 7, 8],
            ...
        }
        '''
        d = {}
        d.update(MPIIHandRotation.add_finger(
            ['W', 'T0', 'T1', 'T2', 'T3']))       # Thumb  finger
        d.update(MPIIHandRotation.add_finger(
            ['W', 'I0', 'I1', 'I2', 'I3'])) # Index  finger
        d.update(MPIIHandRotation.add_finger(
            ['W', 'M0', 'M1', 'M2', 'M3'])) # middle finger
        d.update(MPIIHandRotation.add_finger(
            ['W', 'R0', 'R1', 'R2', 'R3'])) # ring   finger
        d.update(MPIIHandRotation.add_finger(
            ['W', 'L0', 'L1', 'L2', 'L3'])) # little finger
        return d
    
    def add_finger(finger_joints):
        '''
        finger_joints = ['W', 'T1', 'T2', 'T3']
        finger_ind = get_index(finger_joints)
                   = [0, 2, 3, 4]
        return {
            'T1': [0, 2, 3],
            'T2': [2, 3, 4]
        }
        '''
        d = {}
        finger_joints_index = MPIIHandRotation.get_index(finger_joints)
        for mid in range(1, len(finger_joints)-1):
            d[finger_joints[mid]] = [
                finger_joints_index[mid-1], # joint1
                finger_joints_index[mid],   # joint2
                finger_joints_index[mid+1]  # joint3
            ]
        return d
    
    def get_index(joints):
        '''
        joints: ['W', 'T1', 'T2', 'T3']

        return: [0  , 2   , 3   , 4   ]
        return: [labels.index('W'), ...]
        '''
        # index_joints = []
        # for name in joints:
        #     index_joints += [self.labels.index(name)]
        return [MPIIHandRotation.labels.index(name) for name in joints]

def save_csv(inference_dataset_name: str, method: str, video_folder: str, return_min_max=False):
    '''
    prepare path strings for calc_save_csv()
    '''
    input_folder = os.path.join(
        inference_dataset_name,
        f'3-{method}',
        video_folder
    )   # infer_HandMotion / 3-mobrecon / [0_stone] / .npy(s)
    output_folder = os.path.join(
        inference_dataset_name,
        f'4r-{method}_rotation'
    )
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(
        output_folder,
        f'{video_folder}.csv'
    )

    out = calc_save_csv(input_folder, output_path, return_min_max=return_min_max)
    if return_min_max:
        out.name = video_folder
        return out

def calc_save_csv(input_folder: str, output_path: str, return_min_max=False):
    '''
    compute all npy files in {input_folder}

    calculate and
    output rotation information at {output_path}
    '''
    rotations = {e: [] for e in MPIIHandRotation.rotations}  # {'T1': [], ...}
    RotationMap = MPIIHandRotation.get_rotation_dict()

    for filename in os.listdir(input_folder):
        if not filename.endswith('npy'):
            continue

        fullpath = os.path.join(input_folder, filename)
        joints = np.load(fullpath)  # (21, 3)
        update_angle(rotations, joints, RotationMap)

    # end calculate all rotation angles
    df = pd.DataFrame(rotations)

    # normal rotation map in each angle
    df.to_csv(output_path)

    print(f'\ncompute rotation')
    print(f'from: {input_folder}')
    print(f'  to: {output_path}')

    if return_min_max:
        return prepare_overall_minmax(df)


def prepare_overall_minmax(df: pd.DataFrame) -> pd.Series:
    '''
    rotations: {T I M R L} + {0   1   2  }    min max
    new data : {1 2 3 4 5} + {MCP PIP DIP} + {ext flex}
    Note that: thumb have one IP instead of {DIP PIP}

    return pd.Series({
        '1 IP ext': [int], '1 IP flex': [int],
        '1 MCP ext': [int], ...
    })
    '''
    mapping = MPIIHandRotation.mapping
    # {..., '2 DIP': 'I2', '2 PIP': 'I1', '2 MCP': 'I0', ...}  # index  finger

    df_min, df_max = df.min(), df.max()

    row_data = {}
    for medical_name, mobrecon_name in mapping.items():
        row_data[medical_name + ' ext' ] = df_min[mobrecon_name] # min
        row_data[medical_name + ' flex'] = df_max[mobrecon_name] # max

    series = pd.Series(data = row_data)
    return series

# -----------------------------------------------

def update_angle(rotations, joints, RotationMap):
    for rot_joint in rotations:
        # rot_joint: 'T1'
        i1 = RotationMap[rot_joint][0]
        i2 = RotationMap[rot_joint][1]
        i3 = RotationMap[rot_joint][2]
        rotations[rot_joint] += [calc_angle(
            joints[i1], joints[i2], joints[i3]
        )]

def calc_angle(j1, j2, j3):
    '''
    joint 1 ~ 3
    [x, y, z]

    : Θ /(j3)
    :  /
    : /
    :/
    |(j2)
    |
    |(j1)

    '''
    v1 = j2 - j1
    v2 = j3 - j2
    return angle_between(v1, v2) / np.pi * 180

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

if __name__ == "__main__":
    # 3-{method} to 4r-{method}_rotation
    inference_dataset_name = 'infer_ROM_data_2'
    video_folder = [
        '8M',
        '8R',
        '8U',
        # 'hand_2',
    ]
    method = 'mobrecon'

    overall_series = []
    for i in range(len(video_folder)):
        min_max_series = save_csv(
            inference_dataset_name=inference_dataset_name,
            method=method,
            video_folder=video_folder[i],
            return_min_max=True
        )
        overall_series += [min_max_series]

    overall_df = pd.concat(overall_series, axis=1).transpose().round(2)

    overall_df.to_csv(os.path.join(inference_dataset_name, 'overall_rotation.csv'))
