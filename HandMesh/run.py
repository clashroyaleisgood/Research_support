import os
import torch
import cv2
import numpy as np

cnt_area = 'utils.vis'
registration, map2uv, base_transform = 'utils.vis'
save_a_image_with_mesh_joints = 'utils.draw3d'
save_mesh = 'utils.read'
mano_to_mpii = 'datasets.FreiHAND.kinematics'
Bar = 'utils.progress.bar'
from termcolor import colored, cprint


##### recover it to 
class Runner(object):
    def demo(self):
        INFER_FOLDER = '2_ya'
        args = self.args
        self.model.eval()
        # image_fp = os.path.join(args.work_dir, 'images')
        image_fp = os.path.join(args.work_dir, 'images', INFER_FOLDER)
        output_fp = os.path.join(args.out_dir, 'demo', INFER_FOLDER)
        os.makedirs(output_fp, exist_ok=True)
        ''' paths
        input : ~/HandMesh/images/{INFER_FOLDER}/
        output: ~/HandMesh/out/FreiHAND/mobrecon/demo/{INFER_FOLDER} /
        '''

        # image_files = [os.path.join(image_fp, i) for i in os.listdir(image_fp) if '_img.jpg' in i]
        image_files = [os.path.join(image_fp, e) for e in os.listdir(image_fp) if e.endswith('.jpg')]  # or jpg...
        bar = Bar(colored("DEMO", color='blue'), max=len(image_files))
        with torch.no_grad():
            for step, image_path in enumerate(image_files):
                # image_name = image_path.split('/')[-1].split('_')[0]
                image_name = os.path.basename(image_path).split('.')[0]  # '0000'
                image = cv2.imread(image_path)[..., ::-1]
                image = cv2.resize(image, (args.size, args.size))
                input = torch.from_numpy(base_transform(image, size=args.size)).unsqueeze(0).to(self.device)

                # print(f'processing file: {image_path}')
                _Knpy_file_path = image_path.replace('_img.jpg', '_K.npy')
                if os.path.isfile(_Knpy_file_path) and _Knpy_file_path.endswith('_K.npy'):  # example images' K
                    K = np.load(_Knpy_file_path)
                elif os.path.isfile(os.path.join(args.work_dir, 'images', 'default.npy')):  # my images' K
                    K = np.load(os.path.join(args.work_dir, 'images', 'default.npy'))

                K[0, 0] = K[0, 0] / 224 * args.size
                K[1, 1] = K[1, 1] / 224 * args.size
                K[0, 2] = args.size // 2
                K[1, 2] = args.size // 2

                out = self.model(input)
                # silhouette
                mask_pred = out.get('mask_pred')
                if mask_pred is not None:
                    mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    mask_pred = cv2.resize(mask_pred, (input.size(3), input.size(2)))
                    try:
                        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours.sort(key=cnt_area, reverse=True)
                        poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                    except:
                        poly = None
                else:
                    mask_pred = np.zeros([input.size(3), input.size(2)])
                    poly = None
                # vertex
                pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
                vertex = (pred[0].cpu() * self.std.cpu()).numpy()
                uv_pred = out['uv_pred']
                if uv_pred.ndim == 4:
                    uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (input.size(2), input.size(3)))
                else:
                    uv_point_pred, uv_pred_conf = (uv_pred * args.size).cpu().numpy(), [None,]
                vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, K, args.size, uv_conf=uv_pred_conf[0], poly=poly)

                vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
                # np.savetxt(os.path.join(args.out_dir, 'demotext', image_name + '_xyz.txt'), vertex2xyz)
                np.savetxt(os.path.join(output_fp, image_name + '_xyz.txt'), vertex2xyz, fmt='%f')

                save_a_image_with_mesh_joints(image[..., ::-1], mask_pred, poly, K, vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                                              os.path.join(output_fp, image_name + '_plot.jpg'))
                save_mesh(os.path.join(output_fp, image_name + '_mesh.ply'), vertex, self.faces[0])

                bar.suffix = '({batch}/{size})' .format(batch=step+1, size=len(image_files))
                bar.next()
        bar.finish()
