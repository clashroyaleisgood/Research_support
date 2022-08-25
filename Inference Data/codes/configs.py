
'''
inference_dataset_name = 'infer_HandMotion'
video_folder = [
    '0_stone',
    '1_good',
    '2_ya'
]
method = 'mobrecon'

data structure:
infer_HandMotion/
    0-source/
        0_stone.mp4
        1_good.mp4
        ...
    1-full_images/
        0_stone/
            0000.jpg
        1_good/
            0000.jpg
            ...
    2-boxed_images/

    3-{method}/
        0_stone/
            0000_mesh.ply # hand mesh result
            0000_plot.jpg # visualization result
            0000_xyz.npy  # 21 joints
    4v-{method}_video/
        0_stone.mp4
    4j-{method}_joint/
        0_stone/
            0000_joint.ply
            0001_joint.ply
            ...
    4r-{method}_rotation/
        0_stone.csv
'''

StageName = [
    '0-source',
    '1-full_images',
    '2-boxed_images'
    # '3-{method}'
    # '4v-{method}_video'
    # '4j-{method}_joint'
    # '4r-{method}_rotation'
]
