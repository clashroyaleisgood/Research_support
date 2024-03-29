'''
my code to transform all handmesh from jsons, cam calibrations, ... to .ply, .npy files
Options:
- mode: {'cam', 'world'}
- centered: {'cam', 'wrist'}
  'wrist': verts -= root.xyz
- transform_to: {'ply', 'numpy', 'numpy_seq'}
- np_store: {'verts', 'v+j+t'}
  'verts'(Default): store vertices only
  'v+j+t': store vertices + joint + global_t(x, y, z)
           global_t = root_xyz in camera coord
- store_intrinsic: {True, False}
  True (Default): also store intrinsic matrix( camera coord to image )

------

hanco/
    rgb/
    calib/
        0123/
            00000000.json -> cam0~cam8 calib, global_t
    shape/
        0123/
            00000000.json       # world coord -> cam coord  , mode='world'
            cam0/00000000.json  # cam coord                 , mode='cam'
    xyz/
        00000000.json

to

hanco/
    1st out_form: transform_to == 'ply'
    mesh/
        0123/
            cam0/
                00000000.ply    <- verts[778, 3] + faces: 3D model
    mesh_intrinsic/
        0123/
            cam0/
                00000000.npz    <- intrinsic matrix: [3, 3]

    2nd out_form: transform_to == 'numpy'
    verts/
        0123/
            cam0/
                00000000.npz    <- verts[778, 3],
                                   (+ joint[21, 3], global_t[1, 3]) if np_store == 'vert_joint_root'
                                   (+ intrinsic[3, 3])              if store_intrinsic == True

    3rd out_form: transform_to == 'numpy_seq'
    verts/
        0123/
            cam0.npz            <- frames of verts[frames, 778, 3]
                                   (+ joint[frame, 21, 3], global_t[frame, 1, 3]) if np_store == 'vert_joint_root'
                                   (+ intrinsic[frame, 3, 3])                     if store_intrinsic == True
'''

import os, argparse, json, torch
from turtle import shape
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.plot_util import draw_hand

import openmesh as om
from manopth.manolayer import ManoLayer
from utils.mano_utils import pred_to_mano, trafoPoints #, project

# from HandMesh.utils.read import save_mesh
def save_mesh(fp: str, x, f):
    # fp: file_path, x: verts, f: faces
    if not fp.endswith('.ply'):
        fp = fp + '.ply'
    om.write_mesh(fp, om.TriMesh(x, f))

def save_numpy(fp, x):
    # fp: file_path, x: verts
    np.save(fp, x)

# ----------------------------------------------------------------

class Transform():
    def __init__(self,
            hanco_root,
            transform_to='ply', face_path=None,
            np_store='verts', store_intrinsic=False,
            mode='cam',
            centered='camera',
            center_idx=9
        ):
        '''
        input:
        - hanco_root: path to hanco_root directory
        - transform_to: {'ply', 'numpy', 'numpy_seq'}
          transform to which kind of type
          ply       -> store mesh(verts + face)     | for each seq, cam, frame
          numpy     -> store verts( +xyz,...)       | for each seq, cam, frame
          numpy_seq -> store list[verts( +xyz,...)] | for each seq, cam

        - face_path: path to hand mesh faces file
          (optional) when tranform_type=='ply'
        - mode: {'cam', 'world'}
        - centered: {'camera', 'wrist'}
        - center_idx: camera mano needs
            turn mano-{verts, xyz} to wrist-relative

        Different types:
        - transform_to: {'ply', 'numpy', 'numpy_seq', ...}
        - np_store: {'verts', 'v+j+t'}
        - store_intrinsic: {True, False}
        - mode: {'cam', 'world'}
        - centered: {'camera', 'wrist'}
        '''
        self.hanco_root = hanco_root
        self.transform_to = transform_to

        if face_path:
            self.face = np.load(face_path)
            self.face = torch.from_numpy(self.face).long()
            # self.face = np.load(os.path.join('mano_models', 'right_faces.npy'))
        if transform_to == 'ply':
            assert self.face.shape == (1538, 3), 'face file sould be provided while creating ply files'
        elif transform_to == 'numpy':
            pass
        elif transform_to == 'numpy_seq':
            pass
        else:
            raise NotImplementedError(f'transform_to: {transform_to} is not implemented yet')

        self.np_store = np_store
        assert np_store in ('verts', 'v+j+t'), f'np_store: ["verts", "v+j+t"], got: {np_store}'
        self.store_intrinsic = store_intrinsic
        assert type(store_intrinsic) == bool, f'store_intrinsic: bool parameter, got: {store_intrinsic}'

        self.mode = mode
        assert mode in ('cam', 'world'), f'only 2 ways to produce mesh, "cam" or "world", got: {mode}'
        self.centered = centered
        assert centered in ('camera', 'wrist'), f'accept only camera-centered or wrist-centered, got: {centered}'

        self.mano_cam = ManoLayer(use_pca=False, ncomps=45, flat_hand_mean=False, center_idx=center_idx)
        self.mano_world=ManoLayer(use_pca=False, ncomps=45, flat_hand_mean=False)
        self.mano_cam.eval()
        self.mano_world.eval()

        print(f'Transform(transform_to={transform_to}, np_store={np_store}, store_intrinsic={store_intrinsic}, '
              f'mode={mode}, centered={centered})')

    def render(self, poses, shapes, global_t, mode='cam', M=None):
        '''
        create mano verts, xyz by
        input:
        - poses:    tensor with shape [1, 48]
        - shapes:   tensor with shape [1, 10]
        - global_t: tensor with shape [1, 1, 3]
        - mode: {'cam', 'world'}
        - M:        tensor with shape [1, 4, 4]
          (optional) when mode=='world'

        output:
        - verts:    tensor with shape [1, 778, 3], unit: meter
        - xyz:      tensor with shape [1, 21, 3],  unit: meter

        details:
        mode: {'cam', 'world'}
        camera case -> mode == 'cam'
            mano(poses, shapes) -= wrist_xyz += global_t

        world  case -> mode == 'world'
            mano(poses, shapes) += global_t
            transform by M( world coord to cam coord )
        '''
        assert type(poses) == torch.Tensor, f'"poses" in Transform.render() should be torch.Tensor, got: {type(poses)}'
        assert type(shapes) == torch.Tensor, f'"shapes" in Transform.render() should be torch.Tensor, got: {type(shapes)}'
        assert type(global_t) == torch.Tensor, f'"global_t" in Transform.render() should be torch.Tensor, got: {type(global_t)}'

        if mode == 'cam':
            verts_n_xyz = self._render_cam(poses, shapes, global_t)
        elif mode == 'world':
            assert M != None, 'M matrix is needed, to transform from world coord to cam coord'
            verts_n_xyz = self._render_world(poses, shapes, global_t, M)
        else:
            raise TypeError('render() input {mode} should be "cam" or "world" '
                            f'mode = {mode}')
        
        return verts_n_xyz

    def _render_cam(self, poses, shapes, global_t):
        ''' only called by render() '''
        with torch.no_grad():
            verts, xyz = self.mano_cam(poses, shapes, global_t)
        return verts, xyz

    def _render_world(self, poses, shapes, global_t, M):
        ''' only called by render() '''
        with torch.no_grad():
            verts, xyz = self.mano_world(poses, shapes, global_t)
        verts = trafoPoints(verts, M)
        xyz   = trafoPoints(xyz,   M)
        return verts, xyz

    def TransformSequence(self, seq_id):
        '''
        TRANSFORM verts, (joint, global_t), (intrinsic matrix)
        transform
            hanco_root/shape/{seq_id}/ cam_id/frame_id.json -> cam coord
         or hanco_root/shape/{seq_id}/        frame_id.json -> world coord
            hanco_root/calib/{seq_id}/        frame_id.json -> calib matrices
        to
            hanco_root/mesh     /{seq_id}/cam_id/frame_id.ply
            hanco_root/numpy    /{seq_id}/cam_id/frame_id.npz
            hanco_root/numpy_seq/{seq_id}/cam_id.npz

            hanco_root/mesh_intrinsic/{seq_id}/cam_id/frame_id.npz if self.store_intrinsic

        Different type:
        > self.mode: {'cam', 'world'}, which method to preduce verts
        > self.transform_to
        > self.np_store
        > self.store_intrinsic

        Details:
        data: {
            'verts':    ndarray[#, 778, 3],
            'joint':    ndarray[#, 21,  3], if self.np_store == 'v+j+t'
            'global_t': ndarray[#, 1,   3], if self.np_store == 'v+j+t'
            'intrinsic':ndarray[#, 3,   3], if self.store_intrinsic
        }
        '''
        calib_root = os.path.join(self.hanco_root, 'calib', f'{seq_id:04d}')
        frame_counts = len(os.listdir(calib_root))
        # if self.mode == 'cam':
        #     for cam_id in range(8):
        #         self._trans_seq_cam_with_mode_cam(seq_id, cam_id, frame_counts)
        # elif self.mode == 'world':
        #     for cam_id in range(8):
        #         self._trans_seq_cam_with_mode_world(seq_id, cam_id, frame_counts)
        # else:
        #     raise TypeError('render() input {mode} should be "cam" or "world" '
        #                     f'mode = {self.mode}')

        for cam_id in range(8):
            data = self._get_verts(seq_id, cam_id, frame_counts)
            # data = {'verts': (#, 778, 3), 'joint': (#, 21, 3), 'global_t': (#, 1, 3), 'intrinsic': (#, 3, 3)}
            self._store_data(seq_id, cam_id, frame_counts, data)

    def _trans_seq_cam_with_mode_cam(self, seq_id, cam_id, frame_counts):
        '''
        ~ Replaced by _get_verts & _store_data ~

        only called by TransformSequence()
        Process data[seq_id, cam_id]
        stored to mesh/ or numpy/ or numpy_seq/

        Type:
        - self.centered: {'camera', 'wrist'}
        - self.transform_to: {'ply', 'numpy', 'numpy_seq'}
        '''
        calib_folder = os.path.join(self.hanco_root, 'calib', f'{seq_id:04d}')
        shape_folder = os.path.join(self.hanco_root, 'shape', f'{seq_id:04d}', f'cam{cam_id}')

        if self.transform_to == 'ply':
            out_folder = os.path.join(self.hanco_root, 'mesh', f'{seq_id:04d}', f'cam{cam_id}')
        elif self.transform_to == 'numpy':
            out_folder = os.path.join(self.hanco_root, 'numpy', f'{seq_id:04d}', f'cam{cam_id}')
        elif self.transform_to == 'numpy_seq':
            out_folder = os.path.join(self.hanco_root, 'numpy_seq', f'{seq_id:04d}') # , f'cam{cam_id}'
        os.makedirs(out_folder, exist_ok=True)

        verts_list = []

        for frame_id in range(frame_counts):
            calib_file = os.path.join(calib_folder, f'{frame_id:08d}.json')
            shape_file = os.path.join(shape_folder, f'{frame_id:08d}.json')
            out_file = os.path.join(out_folder, f'{frame_id:08d}')  # 忽略副檔名, used for 'ply', 'numpy'

            with open(calib_file, 'r') as file: 
                calib = json.load(file)
            with open(shape_file, 'r') as file:
                shape = np.array(json.load(file))[None]  # (1, 16*3 +10 +3), (1, rot & shape & trans_uv + scale)

            # Use intrinsic matrix to compute global_t position
            pose_cam, shape_cam, global_t_cam = pred_to_mano(shape, np.array(calib['K'])[cam_id][None], fw=np)

            # get camera relative position
            verts, xyz = self.render(
                poses=torch.Tensor(pose_cam),
                shapes=torch.Tensor(shape_cam),
                global_t=torch.Tensor(global_t_cam),
                mode='cam',
            )  # (1, 778, 3), (1, 21, 3)

            # wrist centered or camera centered(default)
            if self.centered == 'wrist':
                root = xyz[0:1, 0:1, :]  # keep dimention (1, 1, 3)
                verts -= root
                xyz = xyz - root  # RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single memory location. 

            # store ply or numpy file
            if self.transform_to == 'ply':
                save_mesh(out_file, verts[0], self.face)
            elif self.transform_to == 'numpy':
                save_numpy(out_file, verts[0])
            elif self.transform_to == 'numpy_seq':
                verts_list += [verts[0]]

        if self.transform_to == 'numpy_seq':
            verts_np = np.stack(verts_list)
            out_file = os.path.join(out_folder, f'cam{cam_id}.npy')
            save_numpy(out_file, verts_np)

    def _trans_seq_cam_with_mode_world(self, seq_id, cam_id, frame_counts):
        '''
        ~ Replaced by _get_verts & _store_data ~

        only called by TransformSequence()
        Process data[seq_id, cam_id]
        stored to mesh/ or numpy/ or numpy_seq/

        Type:
        - self.centered: {'camera', 'wrist'}
        - self.transform_to: {'ply', 'numpy', 'numpy_seq'}
        '''
        calib_folder = os.path.join(self.hanco_root, 'calib', f'{seq_id:04d}')
        shape_folder = os.path.join(self.hanco_root, 'shape', f'{seq_id:04d}') # f'cam{cam_id}'

        if self.transform_to == 'ply':
            out_folder = os.path.join(self.hanco_root, 'mesh', f'{seq_id:04d}', f'cam{cam_id}')
        elif self.transform_to == 'numpy':
            out_folder = os.path.join(self.hanco_root, 'numpy', f'{seq_id:04d}', f'cam{cam_id}')
        elif self.transform_to == 'numpy_seq':
            out_folder = os.path.join(self.hanco_root, 'numpy_seq', f'{seq_id:04d}') # , f'cam{cam_id}'
        os.makedirs(out_folder, exist_ok=True)

        verts_list = []

        for frame_id in range(frame_counts):
            calib_file = os.path.join(calib_folder, f'{frame_id:08d}.json')
            shape_file = os.path.join(shape_folder, f'{frame_id:08d}.json')  # woorld coord
            out_file = os.path.join(out_folder, f'{frame_id:08d}')  # 忽略副檔名, used for 'ply', 'numpy'

            with open(calib_file, 'r') as file: 
                calib = json.load(file)
            with open(shape_file, 'r') as file:
                shape = json.load(file)  # {'poses': (16*3), 'shapes': (10), 'global_t': (3)}

            # get camera relative position
            verts, xyz = self.render(
                poses=torch.Tensor(shape['poses']),         # (1, 16*3)
                shapes=torch.Tensor(shape['shapes']),       # (1, 10)
                global_t=torch.Tensor(shape['global_t']),   # (1, 1, 3)
                mode='world',
                M=torch.Tensor(np.array(calib['M'][cam_id])[None]),  # (4, 4) -> (1, 4, 4) -> Tensor(1, 4, 4)
            )  # (1, 778, 3), (1, 21, 3)

            if self.centered == 'wrist':
                root = xyz[0:1, 0:1, :]  # keep dimention (1, 1, 3)
                verts -= root
                xyz = xyz - root  # RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single memory location. 

            if self.transform_to == 'ply':
                save_mesh(out_file, verts[0], self.face)
            elif self.transform_to == 'numpy':
                save_numpy(out_file, verts[0])
            elif self.transform_to == 'numpy_seq':
                verts_list += [verts[0]]

        if self.transform_to == 'numpy_seq':
            verts_np = np.stack(verts_list)
            out_file = os.path.join(out_folder, f'cam{cam_id}.npy')
            save_numpy(out_file, verts_np)

    def _get_verts(self, seq_id, cam_id, frame_counts):
        '''
        called by TransformSequence()
        produce: verts, (joint, global_t), (intrinsic matrix)
            of data[seq_id, cam_id]

        input: seq_id, cam_id

        Type:
        > self.mode: {'cam', 'world'}
          transform method
        > self.centered: {'cam', 'wrist'}
          (0, 0, 0) at camera or wrist
        > self.np_store, self.store_intrinsic

        return: {
            'verts':    ndarray[#, 778, 3],
            'joint':    ndarray[#, 21,  3], if self.np_store == 'v+j+t'
            'global_t': ndarray[#, 1,   3], if self.np_store == 'v+j+t'
            'intrinsic':ndarray[#, 3,   3], if self.store_intrinsic
        }
        '''
        calib_folder = os.path.join(self.hanco_root, 'calib', f'{seq_id:04d}')
        shape_folder = os.path.join(self.hanco_root, 'shape', f'{seq_id:04d}')
        if self.mode == 'cam':
            shape_folder = os.path.join(shape_folder, f'cam{cam_id}')

        verts_list = []
        xyz_list = []
        global_t_list = []
        intrinsic_list = []
        for frame_id in range(frame_counts):
            calib_file = os.path.join(calib_folder, f'{frame_id:08d}.json')
            shape_file = os.path.join(shape_folder, f'{frame_id:08d}.json')

            with open(calib_file, 'r') as file: 
                calib = json.load(file)
            with open(shape_file, 'r') as file:
                if self.mode == 'cam':
                    shape = np.array(json.load(file))[None]  # (1, 16*3 +10 +3), (1, rot & shape & trans_uv + scale)
                elif self.mode == 'world':
                    shape = json.load(file)  # {'poses': (16*3), 'shapes': (10), 'global_t': (3)}
            
            # rendering
            K = np.array(calib['K'])[cam_id][None]
            if self.mode == 'cam':
                # Use intrinsic matrix to compute global_t position
                pose_cam, shape_cam, global_t_cam = pred_to_mano(shape, K, fw=np)

                # get camera relative position
                verts, xyz = self.render(
                    poses=torch.Tensor(pose_cam),
                    shapes=torch.Tensor(shape_cam),
                    global_t=torch.Tensor(global_t_cam),
                    mode='cam',
                )  # (1, 778, 3), (1, 21, 3)
            elif self.mode == 'world':
                # get camera relative position
                verts, xyz = self.render(
                    poses=torch.Tensor(shape['poses']),         # (1, 16*3)
                    shapes=torch.Tensor(shape['shapes']),       # (1, 10)
                    global_t=torch.Tensor(shape['global_t']),   # (1, 1, 3)
                    mode='world',
                    M=torch.Tensor(np.array(calib['M'][cam_id])[None]),  # (4, 4) -> (1, 4, 4) -> Tensor(1, 4, 4)
                )  # (1, 778, 3), (1, 21, 3)

            # centered
            root = xyz[0:1, 0:1, :]  # keep dimention (1, 1, 3)
            if self.centered == 'wrist':
                verts -= root
                xyz = xyz - root  # RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single memory location. 

            # store type
            verts_list.append(verts[0])
            if self.np_store == 'v+j+t':
                xyz_list.append(xyz[0])
                global_t_list.append(root[0])
            if self.store_intrinsic == True:
                intrinsic_list.append(K[0])

        # return Dict(frames of data)
        data = {}
        data['verts'] = np.stack(verts_list)
        if self.np_store == 'v+j+t':
            data['joint'] = np.stack(xyz_list)
            data['global_t'] = np.stack(global_t_list)
        if self.store_intrinsic == True:
            data['intrinsic'] = np.stack(intrinsic_list)

        return data

    def _store_data(self, seq_id, cam_id, frame_counts, data):
        '''
        called by TransformSequence()
        store data in different format
            according to self.transform_to, and self.{np_store, store_intrinsic}

        input: seq_id, cam_id, data
        data = {
            'verts':    ndarray[#, 778, 3],
            'joint':    ndarray[#, 21,  3], if self.np_store == 'v+j+t'
            'global_t': ndarray[#, 1,   3], if self.np_store == 'v+j+t'
            'intrinsic':ndarray[#, 3,   3], if self.store_intrinsic
        }

        Type:
        > self.transform_to: {'ply', 'numpy', 'numpy_seq'}
          place / file format to store
        > self.np_store, self.store_intrinsic
          store more information
        '''
        if self.transform_to == 'ply':
            out_folder = os.path.join(self.hanco_root, 'mesh', f'{seq_id:04d}', f'cam{cam_id}')
            if self.store_intrinsic == True:
                out_intrinsic = os.path.join(self.hanco_root, 'mesh_intrinsic', f'{seq_id:04d}', f'cam{cam_id}')
                os.makedirs(out_intrinsic, exist_ok=True)
        elif self.transform_to == 'numpy':
            out_folder = os.path.join(self.hanco_root, 'numpy', f'{seq_id:04d}', f'cam{cam_id}')
        elif self.transform_to == 'numpy_seq':
            out_folder = os.path.join(self.hanco_root, 'numpy_seq', f'{seq_id:04d}') # , f'cam{cam_id}'
        os.makedirs(out_folder, exist_ok=True)

        if self.transform_to == 'ply':
            for frame_id in range(frame_counts):
                out_file = os.path.join(out_folder, f'{frame_id:08d}.ply')
                verts = data['verts'][frame_id]
                save_mesh(out_file, verts, self.face)

                if self.store_intrinsic == True:
                    intrinsic = data['intrinsic'][frame_id]
                    out_file = os.path.join(out_intrinsic, f'{frame_id:08d}.npz')
                    np.savez(out_file, intrinsic=intrinsic)

        elif self.transform_to == 'numpy':
            for frame_id in range(frame_counts):
                out_file = os.path.join(out_folder, f'{frame_id:08d}.npz')
                frame_data = {}
                for k in data:
                    frame_data[k] = data[k][frame_id]

                np.savez(out_file, **frame_data)

        elif self.transform_to == 'numpy_seq':
            out_file = os.path.join(out_folder, f'cam{cam_id}.npz')
            np.savez(out_file, **data)

    def TransformFrom(self, start=0):
        '''
        TRANSFORM verts, (joint, global_t), (intrinsic matrix)
        Use TransformSequence() to transform valid sequences
        '''
        last = 1517  # (0, last)
        meta_file = os.path.join(self.hanco_root, 'meta.json')
        with open(meta_file, 'r') as file: 
            meta_data = json.load(file)

        self.valid_seq = []  # True, False sequence, of len=1518
        for seq in meta_data['has_fit']:
            self.valid_seq.append(sum(seq) == len(seq))  # all the frames are available
        assert len(self.valid_seq) == 1518, '# of seq must be 1518'

        # Start Transform from input: {start}
        print(f'Start transform sequences from: {start}')
        for i in tqdm(range(start, last +1)):
            if self.valid_seq[i] == False:
                print(f'invalid seq: {i}')
                continue
            # else
            try:
                self.TransformSequence(i)
            except Exception as e:
                print(f'[Error] Happend while transforming seq: {i}')
                raise e




def exp_wrist_centered_vibration(index=100):
    ''' check vibration on verts[index] '''
    import pandas as pd
    hanco_root='C:\\Users\\oscar\\Desktop\\HanCo_tester'
    face_path = os.path.join('mano_models', 'right_faces.npy')

    seq_id = 110
    cam_id = 2
    pool = {
        'w_x': [],
        'w_y': [],
        'w_z': [],
        'c_x': [],
        'c_y': [],
        'c_z': [],
    }

    calib_root = os.path.join(hanco_root, 'calib', f'{seq_id:04d}')
    frame_counts = len(os.listdir(calib_root))

    def _test(seq_id, cam_id, frame_counts):
        calib_folder = os.path.join(hanco_root, 'calib', f'{seq_id:04d}')
        shape_folder = os.path.join(hanco_root, 'shape', f'{seq_id:04d}') # f'cam{cam_id}'
        mesh_folder = os.path.join(hanco_root, 'mesh', f'{seq_id:04d}', f'cam{cam_id}')
        os.makedirs(mesh_folder, exist_ok=True)

        for frame_id in range(frame_counts):
            calib_file = os.path.join(calib_folder, f'{frame_id:08d}.json')
            shape_file = os.path.join(shape_folder, f'{frame_id:08d}.json')  # woorld coord
            mesh_file = os.path.join(mesh_folder, f'{frame_id:08d}.ply')

            with open(calib_file, 'r') as file: 
                calib = json.load(file)
            with open(shape_file, 'r') as file:
                shape = json.load(file)  # {'poses': (16*3), 'shapes': (10), 'global_t': (3)}

            # get camera relative position
            verts, xyz = trans.render(
                poses=torch.Tensor(shape['poses']),         # (1, 16*3)
                shapes=torch.Tensor(shape['shapes']),       # (1, 10)
                global_t=torch.Tensor(shape['global_t']),   # (1, 1, 3)
                mode='world',
                M=torch.Tensor(np.array(calib['M'][cam_id])[None]),  # (4, 4) -> (1, 4, 4) -> Tensor(1, 4, 4)
            )  # (1, 778, 3), (1, 21, 3)

            if trans.centered == 'wrist':
                root = xyz[0:1, 0:1, :]  # keep dimention (1, 1, 3)
                verts -= root
                xyz = xyz - root  # RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single memory location. 

            center = trans.centered[0]  # 'c' or 'w'
            pool[f'{center}_x'] += [ verts[0][index][0].item() ]
            pool[f'{center}_y'] += [ verts[0][index][1].item() ]
            pool[f'{center}_z'] += [ verts[0][index][2].item() ]

    trans = Transform(hanco_root, transform_to='ply', face_path=face_path, mode='world', centered='wrist')
    _test(seq_id, cam_id, frame_counts)

    trans = Transform(hanco_root, transform_to='ply', face_path=face_path, mode='world', centered='camera')
    _test(seq_id, cam_id, frame_counts)

    df = pd.DataFrame(pool)
    csv_path = os.path.join(trans.hanco_root, '')
    df.to_csv('_exp_vibration\\vibration.csv')

# edited from show_dataset
def render_mesh(poses, shapes, global_t, K, M=None, center_idx=None):
    '''
    create mano verts, xyz by
    input:
    @ poses: [1, 48]
    @ shapes: [1, 10]
    @ global_t: [1, 1, 3]

    out:
    verts: [1, 778, 3]
    xyz: [1, 21, 3]
    '''
    # print(f'K: {K}')
    if M is None:
        M = np.eye(4)

    mano = ManoLayer(use_pca=False, ncomps=45, flat_hand_mean=False, center_idx=center_idx)

    verts, xyz = mano(poses, shapes, global_t)
    # uv = project(trafoPoints(xyz, torch.Tensor(M)[None]), torch.Tensor(K)[None])
    # mask, _  = render_verts_faces(verts,
    #                                 mano.th_faces[None],
    #                                 K[None], M[None], img_shape[None], device='cpu')


    # mask = mask[0].detach().cpu().numpy()[0]
    # uv = uv.detach().cpu().numpy()[0]
    return verts, xyz

def exp_cam(hanco_root: str, seq_id=110, cam_id=0, frame_id=0):
    calib_file = os.path.join(hanco_root, 'calib', f'{seq_id:04d}',                 f'{frame_id:08d}.json')
    with open(calib_file, 'r') as file: 
        calib = json.load(file)

    shape_file = os.path.join(hanco_root, 'shape', f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}.json')
    with open(shape_file, 'r') as file:
        shape = np.array(json.load(file))[None]

    pose_cam, shape_cam, global_t_cam = pred_to_mano(shape, np.array(calib['K'])[cam_id][None], fw=np)
    
    verts, xyz = render_mesh(torch.Tensor(pose_cam), 
                             torch.Tensor(shape_cam), 
                             torch.Tensor(global_t_cam),
                             np.array(calib['K'][cam_id]),
                             center_idx=9)  # verts, xyz -= xyz[center_idx]
    # print(xyz - global_t_cam)
    
    face = np.load('mano_models\\right_faces.npy')
    save_mesh(os.path.join('_exp', 'cam.ply'), verts[0], face)
    # print(xyz)
    return verts[0], xyz[0]

def exp_world(hanco_root: str, seq_id=110, cam_id=0, frame_id=0):
    calib_file = os.path.join(hanco_root, 'calib', f'{seq_id:04d}',                 f'{frame_id:08d}.json')
    with open(calib_file, 'r') as file: 
        calib = json.load(file)

    shape_file = os.path.join(hanco_root, 'shape', f'{seq_id:04d}',                 f'{frame_id:08d}.json')
    with open(shape_file, 'r') as file:
        shape = json.load(file)

    verts, xyz = render_mesh(torch.Tensor(shape['poses']), 
                             torch.Tensor(shape['shapes']), 
                             torch.Tensor(shape['global_t']),
                             np.array(calib['K'][cam_id]),
                             np.array(calib['M'][cam_id]))
    M = np.array(calib['M'][cam_id])
    # print(xyz - np.array(shape['global_t']))

    # World to Cam coord
    verts = trafoPoints(verts, torch.Tensor(M)[None])

    face = np.load('mano_models\\right_faces.npy')
    save_mesh(os.path.join('_exp', 'world.ply'), verts[0], face)
    return verts[0], xyz[0]

def example_meta_data(hanco_root):
    meta_file = os.path.join(hanco_root, 'meta.json')
    with open(meta_file, 'r') as fi: 
        meta_data = json.load(fi)
    # keys are: 'is_train': 綠幕, 'subject_id': 人物, 'is_valid': MANO 有人工檢驗過,
    #           'object_id': 手握, 'has_fit': 有 MANO 參數
    shape_filter = {
        'is_valid': [],
        'is_valid_ALL': [],
        'has_fit': [],
        'has_fit_ALL': [],
    }
    for seq in meta_data['is_valid']:
        shape_filter['is_valid'] += [sum(seq)]
        shape_filter['is_valid_ALL'] += [len(seq)]
    for seq in meta_data['has_fit']:
        shape_filter['has_fit'] += [sum(seq)]
        shape_filter['has_fit_ALL'] += [len(seq)]
    import pandas as pd
    df = pd.DataFrame(shape_filter)
    df.to_csv(os.path.join(hanco_root, 'meta_info.csv'))

if __name__ == '__main__':
    hanco_root = '/home/oscar/Desktop/HanCo'
    # hanco_root = 'C:\\Users\\oscar\\Desktop\\HanCo_tester'
    face_path = os.path.join('mano_models', 'right_faces.npy')

    hyper_parameters = {
        'hanco_root': hanco_root,
        'face_path': face_path,

        'transform_to': 'numpy_seq',
        'np_store': 'v+j+t',
        'store_intrinsic': True,

        'mode': 'world',
        'centered': 'wrist',
    }

    trans = Transform(**hyper_parameters)
    # trans.TransformSequence(0)

    trans.TransformFrom(0)

    # mask_dir = os.path.join(hanco_root, 'mask_hand')
    # for i in range(1518):
    #     if not os.path.exists(os.path.join(mask_dir, f'{i:04d}')):
    #         print(f'mask not exist: {i:04d}')

    # exp_wrist_centered_vibration(424) # 食指前端

    # example_meta_data(hanco_root)

    # m1, x1 = exp_cam(hanco_root, cam_id=1)
    # m2, x2 = exp_world(hanco_root, cam_id=1)
    # print((m1-m2).max())
    # print(m2)
