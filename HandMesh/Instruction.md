# Install

follow the instruction in README

## Python & PyTorch
```
conda create -n handmesh_new python=3.9
conda activate handmesh_new

# PyTorch official( previous verrsion ): https://pytorch.org/get-started/previous-versions/
# PyTorch 1.11.0 | Pip | Python | CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

## Other Packages( pip )
```
pip install -r requirements.txt  # fail, need some edits
```
comment (註解掉)
- `torch==1.11.0`
- `torchaudio==0.11.0`
- `torchvision==0.12.0`
- `torch-cluster==1.6.0`, fail
- `torch-scatter==2.0.9`, fail
- `torch-sparse==0.6.13`, fail
- `torch-spline-conv==1.2.1`, fail

re-execute
```
pip install -r requirements.txt
```

install the packages that fails
install at: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```
# PyTorch 1.11.* | Linux | Pip | CUDA 11.3
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

## Specail Package( MPI-IS )
Install MPI-IS Mesh from the source: https://github.com/MPI-IS/mesh
```
# require
sudo apt-get install libboost-dev
    # check if exist
    sudo apt list --installed | grep boost

# install psbody-mesh
~/Desktop/mesh $
    BOOST_INCLUDE_DIRS=/path/to/boost/include make all
```
test with: `$ make tests`
done!

## download MANO model
download it!
```
Downloads/MANO/
    mano_v1_2/models/MANO_RIGHT.pkl
    ...
```

create soft link
```
Desktop/HandMesh $
    ln -s ~/Downloads/MANO/mano_v1_2/models/MANO_RIGHT.pkl template/MANO_RIGHT.pkl
```

## Download other model weight or other files
https://drive.google.com/drive/folders/1MIE0Jo01blG6RWo2trQbXlQ92tMOaLx_

- place `cmr/out/FreiHAND/mobrecon/checkpoints/mobrecon_densestack_dsconv.pt`
  for demo mobrecon
- place `HandMesh/cmr/out/Human36M/cmr_g/checkpoints/cmr_pg_res18_human36m.pt`
  for demo cmr...


# Demo
```
./cmr/scripts/demo_mobrecon.sh
```
## Original Version
inputs at
```
HandMesh/cmr/images/...
```
outputs at
```
HandMesh/cmr/out/FreiHAND/mobrecon/demo/...
```

## Edited Version
replace `HandMesh/cmr/runer.py`
with `Research_support/HandMesh/run.py`

inputs at
```
HandMesh/cmr/images/{some folders}/...
```
outputs at
```
HandMesh/cmr/out/FreiHAND/mobrecon/demo/{some folders}/...
```

# Train
```
./mobrecon/scripts/train_mobrecon.sh
```
## Dataset - FreiHAND
Download FreiHAND dataset from official website: https://lmb.informatik.uni-freiburg.de/projects/freihand/  
`FreiHAND_pub_v2.zip`

follow the instructions in HandMesh README to accomplish the dataset  
and download `freihand_train_mesh.zip` from https://drive.google.com/drive/folders/1MIE0Jo01blG6RWo2trQbXlQ92tMOaLx_

also prepare its evaluation set

create soft link to `HandMesh/data/FreiHAND
```
HandMesh/data $
    ln -s /path/to/FreiHAND_pub_v2 FreiHAND
```

## Dataset - GE
```
~/Desktop $
    git clone https://github.com/3d-hand-shape/hand-graph-cnn/
```
create soft link to `HandMesh/data/Ge
```
HandMesh/data $
    ln -s ../../hand-graph-cnn/data/real_world_testset Ge
```

## Dataset - Compdata
Download Complement dataset following https://github.com/SeanChenxy/HandMesh/blob/main/complement_data.md
```
create folders like
CompHand/           -- different to official README
    base_pose/
    trans_pose_batch1/
    trans_pose_batch2/
    trans_pose_batch3/
```
create soft link to `HandMesh/data/CompHand
```
HandMesh/data $
    ln -s ~/Desktop/CompHand CompHand
```

## Pretrained Weight
Download `densestack.pth` from https://drive.google.com/drive/folders/1MIE0Jo01blG6RWo2trQbXlQ92tMOaLx_  
and placed at `HandMesh/mobrecon/out/densestack.pth`

## code edition
- show message on cmd interface
  utils/writer.py -> Writer.print_step_ft(self, info) -> add `print('  > ' + message)`
- ... as my forked version

## code detail
`mobrecon/main.py`:
- build_model()
  `build.py` - `mobrecon/models/mobrecon_ds.py`
- build_dataset()
  `build.py` - `mobrecon/datasets/multipledatasets.py` - `mobrecon/datasets/[freihand.py, comphand.py]`
- train_loader = DataLoader(dataset)
- runner.run(), run the training, testing...
  `runner.py`


`mobrecon/build.py`:
```
def build_model(cfg):
    #                    .get('MobRecon_DS')
    return MODEL_REGISTRY.get(cfg['MODEL']['NAME'])(cfg)

def build_dataset(cfg):
    #                   .get('MultipleDatasets')
    return DATA_REGISTRY.get(cfg[phase.upper()]['DATASET'])(cfg, phase, **kwargs)
```

`mobrecon/models/mobrecon_ds.py`:
```
from mobrecon.build import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class MobRecon_DS(nn.Module):
    pass

decorater 將 MobRecon_DS class 紀錄到 build.py - MODEL_REGISTRY 裡面
MODEL_REGISTRY.get('MobRecon_DS') -> 得到那個 class
```

`mobrecon/datasets/multipledatasets.py`
```
from mobrecon.build import DATA_REGISTRY
@DATA_REGISTRY.register()
class MultipleDatasets(Dataset):
    pass

decorater 將 MultipleDatasets class 紀錄到 build.py - DATA_REGISTRY 裡面
DATA_REGISTRY.get('MultipleDatasets') -> 得到那個 class

MultipleDatasets(Dataset)
將兩個 datasets 合併，FreiHAND(), CompHand()
```

`mobrecon/runner.py`:
- Runner() class
```
self.run(phase='train')
-> for epoch in range(MAX_EPOCH): self.train()
-> for data in self.train_loader: out = self.model(data['img'])
```
```
self.run(phase='pred') # predict on FreiHAND eval dataset
-> verts_pred, align_state = registration(verts_pred, joint_img_pred, self.j_reg, data['calib'][0].cpu().numpy(), self.cfg.DATA.SIZE, poly=poly)
    update verts_pred from wrist centered to camera centered
    by utilizing verts_pred, and joint_img_pred
-> j_reg @ (verts + j)
```

# New Code
## Train... same

## Evaluation - Ge
Evaluate on Ge dataset

**Steps**
```
exp_name='mrc_ds'
CUDA_VISIBLE_DEVICES=0

python -m mobrecon.main \
    --exp_name $exp_name \
    --config_file mobrecon/configs/mobrecon_ds_eval.yml
```

**evaluate** result show on terminal

## Test - FreiHAND
Test on FreiHAND test set

**Steps**
```
exp_name='mrc_ds'
CUDA_VISIBLE_DEVICES=0

python -m mobrecon.main \
    --exp_name $exp_name \
    --config_file mobrecon/configs/mobrecon_ds_pred.yml
```

**predict** result at:
- json: `HandMesh/mobrecon/out/MultipleDatasets/mrc_ds/mrc_ds.json`
- image: `HandMesh/mobrecon/out/MultipleDatasets/mrc_ds/test/`

## Inference
inference the images in `HandMesh/mobrecon/images/{ fold }/*.jpg`

**Steps**
```
exp_name='mrc_ds'
CUDA_VISIBLE_DEVICES=0

python -m mobrecon.main \
    --exp_name $exp_name \
    --config_file mobrecon/configs/mobrecon_ds_demo.yml
```

**results** are stored in `HandMesh/mobrecon/out/MultipleDatasets/mrc_ds/demo/{ fold }/`  
or shortcut(soft link) in `HandMesh/mobrecon/images/demo/{ fold }/`

## (New) Dataset - HanCo
hand motion **sequential** images shot from **8 viewpoints**
data descriptions
- RGB images in `rgb` folder  
  `rgb      /{clip name}/cam[0-7]/[0000 - 0123].jpg`
- Hand Mask in `mask_hand` folder  
  `mask_hand/{clip name}/cam[0-7]/[0000 - 0123].jpg`
- Hand Joint notation in `xyz` folder  
  `xyz      /{clip name}/[0000 - 0123].json`
- Hand Mesh notation in `shape` folder  
  `shape    /{clip name}/[0000 - 0123].json`
  `shape    /{clip name}/cam[0-7]/[0000 - 0123].json`
- Camera Parameter in `calib` folder  
  `calib    /{clip name}/[0000 - 0123].json`

```
- Hand Joint notation in `xyz` folder
[21 * [xyz coord]]
== 21 * 3 matrix

- Hand Mesh notation in `shape` folder
shapes: 1 * 10 matrix
poses: 1 * 48 matrix
global_t: 1 * 1 * 3 matrix

- Camera Parameter in `calib` folder
K: 8 * [3x3 matrix]
M: 8 * [4x4 matrix]
```

### Transform
use [transform_json_to_meshes.py](https://github.com/clashroyaleisgood/Research_support/blob/main/contra-hand/transform_json_to_meshes.py) to transform original `MANO shapes, poses` into `ply or numpy`  

in HanCo meta_data  
keys are:
- 'is_train': 綠幕
- 'subject_id': 人物
- 'object_id': 手握
- 'has_fit': 是否有 MANO 參數
- 'is_valid': 該 MANO 參數 是否有人工檢驗過

for each sequence, has_fit 的數量比 is_valid 更多，  
此 Tansform 只會將，所有 frames 都有 MANO 參數的 SEQUENCE 進行轉換  
filter: `len(has_fit) == sum(has_fit)`  
另一方面，某些 sequence 沒有 mask，這些 sequence 在上一步的 filter 剛好都被過濾掉了，所以不用擔心

## Mobrecon + Transformer(with previous 8 frames)
