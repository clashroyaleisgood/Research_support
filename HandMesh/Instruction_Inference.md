# Overall Pipeline
1. do video pre-processing( crop images )
2. do `MobRecon` handmesh predction( cropped images to mesh coord )
3. do prediction post-processing( mesh to angle )

# Environment
## Install Miniconda
to manage python package environments: 
[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

## Environment for pre/post-processing
### Clone my repository
```
~/Desktop $
    git clone https://github.com/clashroyaleisgood/Research_support.git
```

### Install python Packages
```
conda create -n processing python=3.9
conda activate processing
pip install numpy==1.23.1 opencv-python==4.6.0.66 mediapipe==0.8.9.1 pandas==1.4.2 protobuf==3.19.4
```

## Environment for MobRecon
### Clone my repository
```
~/Desktop $
    git clone https://github.com/clashroyaleisgood/HandMesh.git
~/Desktop $
    cd HandMesh
```

### Install Python & PyTorch
```
conda create -n handmesh python=3.9.12 pip=23.0.1
conda activate handmesh

# PyTorch official( previous version ): https://pytorch.org/get-started/previous-versions/
# PyTorch 1.11.0 | Conda | Python | CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

### Install other Packages
```
~/Desktop/HandMesh $
    pip install -r requirements.txt
```

### Install pytorch-geometric
install at: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```
# PyTorch 1.11.* | Linux | Pip | CUDA 11.3
pip install torch-scatter==2.0.9 torch-sparse==0.6.14 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

### Install MPI-IS
Install MPI-IS Mesh from the source: https://github.com/MPI-IS/mesh
```
~/Desktop/HandMesh $
    cd ..
~/Desktop $
    git clone https://github.com/MPI-IS/mesh.git
~/Desktop $
    cd mesh
```

pre-requirement
```
sudo apt-get install libboost-dev
sudo apt list --installed | grep boost  # check if exist
```

delete text in `Desktop/mesh/requirements.txt`
> reason: the procedure of installing MPI-IS will upgrade python packages in resuirements.txt automatically,
> and that will DISTROY the package version dependencies.
> In the other hand, we have already install these packages( pip ) in [Install other Packages](#install-other-packages) step, so we don't need to install it again.

run the installation
```
~/Desktop/mesh $
    BOOST_INCLUDE_DIRS=/path/to/boost/include make all
```

## End of Environment Installation

# Before `MobRecon`: Prepare MANO model and Pretrained Weights
Place `MANO_RIGHT.pkl` and `checkpoint_best.pt` as below
```
Desktop/HandMesh/
    template/
        MANO_RIGHT.pkl
    my_research/out/FreiHAND_Angle/mrc_ds_angle_1_head_pretrained_correct/checkpoints/
        checkpoint_best.pt
```
## Mano model
Accept [MANO LICENCE](https://mano.is.tue.mpg.de/license.html).  
Download MANO model from official website, then place it to
```
~/Desktop/HandMesh/
    template/MANO_RIGHT.pkl
```

## Pretrained Weights
Download pretrained weights (`checkpoint_last.pth`) at: [link](https://drive.google.com/drive/folders/1Nai7gcDmep39QGN3ToDaXwsauQUvutHh?usp=drive_link)  
place it to
```
~/Desktop/HandMesh/
    my_research/out/FreiHAND_Angle/mrc_ds_angle_1_head_pretrained_correct/checkpoints/checkpoint_best.pt
```

# Inference Videos( Demo )
Folder Config
```
Research_support/Inference Data/
    codes/
    {Experiment_name}/
        0-source/               --- your source videos
            inference_video1.mp4
            inference_video2.mp4
            ...

        1-full_images/
            inference_video1/
                0000.jpg
                0006.jpg
                ...
            inference_video2/
                0000.jpg
                ...
        2-boxed_images/
        3-{model_name}/
            video_clip1/
                0000_mesh.ply   # the output mesh file
                0000_plot.jpg   # simple visualization result
                0000_xyz.npy    # the xyz coordinates of 21 joints
                0006_mesh.ply
                ...
            video_clip2/
                ...
        4r-{model_name}_rotation/
        4v-{model_name}_video/
```

## Pre-process
`Videos -> pre-process -> Cropped square images with fps: ~5`  
> switch to env: pre/post-processing  
> `conda activate processing`  
> `cd Research_support/Inference\ Data`

Follow folder config above  
Create a new {Experiment_name} and place videos inside `0-source/`

Use `Inference Data/codes/_run_all.py` for pre-processing
1. **edit** `inference_dataset_name` in `_run_all.py` to your {Experiment_name}
2. uncommented `a_step()` and `b_step` for spliting frames and cropping
3. commented `d_step()` because we don't have prediction result yet

then, run
```
Research_support/Inference Data $
    codes/_run_all.py
```

the cropped output images are placed in: `Experiment_name/2-boxed_images/`  
move it to MobRecon folder as described below

## MobRecon Predict
`Cropped images -> MobRecon -> Hand mesh, joint prediction`  
> switch to env: MobRecon  
> `conda activate handmesh`
> `cd HandMesh`

place cropped images in `Experiment_name/2-boxed_images/` to
```
~/Desktop/HandMesh/my_research/images/
    video_clip1/
        0000.jpg
        0006.jpg
        ...
    video_clip2/
        000.jpg
        ...
```

and run
```
~/Desktop/HandMesh $
    ./my_research/scripts/train_mobrecon_angle.sh
```

prediction results are stored in `HandMesh/my_research/out/FreiHAND_Angle/mrc_ds_angle_1_head_pretrained_correct/demo/{ video_clip1 | video_clip2 }/`  
move it back to post-processing folder `3-{model_name}/` as described above

## Post-process
`Hand joint prediction -> post-process -> angles in each frame + min-max angles`  
> switch to env: pre/post-processing  
> `conda activate processing`
> `cd Research_support/Inference\ Data`

place prediction results to `3-{model_name}/` as folder configuration above
```
Inference Data/
    Experiment_name/
        0-source/
        ...
        3-{model_name}/
            video_clip1/
                0000_mesh.ply   # the output mesh file
                0000_plot.jpg   # simple visualization result
                0000_xyz.npy    # the xyz coordinates of 21 joints
                0006_mesh.ply
                ...
            video_clip2/
```
Use `Desktop/Research_support/Inference Data/codes/_run_all.py` for post-processing
1. **edit** `method` in `_run_all.py` to your {model_name}
2. uncommented `d_step()` to generate joint angles in each frame and the video
3. set `mode` to `'vr'`, to do `video` and `rotation` step on prediction results
4. commented `a_step()` and `b_step`

then, run
```
Research_support/Inference Data $
    codes/_run_all.py
```
 
the joint angle information is at: `Experiment_name/4r-mobrecon_rotation/`
the video is at: `Experiment_name/4v-mobrecon_video/`
