# Instructions
```
Inference Data/
    codes/
    Experiment_name/
        0-source/             --- if you have source videos
            inference_video1.mp4
            inference_video2.mp4
            ...

        1-full_images/
            inference video1/
                0000.jpg
                0001.jpg
                ...
            inference video2/
                0000.jpg
                ...
        2-boxed_images/
        3-{model_name}/
        4v-{model_name}_video/
        4r-{model_name}_rotation/
        4j-{model_name}_joint/
```
0. prepare `{videos}.mp4` in `Experiment_name/0-source`  
1. extract videos to images  
   `a_split_video.py`  
   > this will generate `Experiment_name/1-full_images`
2. find bbox, cut and resize  
   `b_boxing_images.py`  
   you'll need to manually find good bbox position  
   **Note that:** bbox must be square  
   > this will generate `Experiment_name/2-boxed_images`

   **new**  
   use `b_mediapipe_auto_find.py` to crop images automatically  
   by joint 2D prediction from mediapipe

3. use some other ML model to inference  
   > Put results to `Experiment_name/3-{model_name}/`  
   > `video_name/0000_mesh.ply`  
   > `video_name/0000_plot.jpg`  
   > `video_name/0000_xyz.npy`
4. Manipulate result
   1. combine  
      `d_combine_to_video.py`  
      > this will generate `Experiment_name/4v-{model_name}_video`
   2. calculate rotation angles  
      `d_calculate_rotation.py`  
      > this will generate `Experiment_name/4r-{model_name}_rotation`
   3. create joint `.ply` file  
      `d_joint_to_ply.py`  
      > this will generate `Experiment_name/4j-{model_name}_joint`
   4. visualize joint + mesh result  
      `d_joint_mesh_to_ply.py`  
      > this will **NOT** generate anything

**Update**:
- Use `_run_all.py` to run steps above faster and more convinient
