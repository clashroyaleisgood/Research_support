# Evaluate Results

```
FH_dataset/
    freihand/
        eval.py             <- replace file
        view_samples.py     <- replace file
        ...
    eval_GT/                <- ground-truth data
    mrc_ds_50.json          <- prediction result
    {output_folder}/        <- evaluate result, set name in eval.py
```

1. Prepare code:
   1. Create folder to contain eval code, datasets, and prediction result: `FH_dataset`
   2. git clone https://github.com/lmb-freiburg/freihand inside `FH_dataset`
   3. 將 `FH_dataset/freihand/eval.py, FH_dataset/freihand/view_samples.py` 替換為資料夾內的兩份程式檔案
2. Prepare ground-truth
   1. downloads ground-truth from lab-NAS or FreiHAND Offitial: `FreiHAND_pub_v2_eval.zip` and unzip
   2. palce it inside `FH_dataset` and rename with `'eval_GT'`,  
      or using `ln` in ubuntu: `ln -s path/to/FreiHAND_pub_v2_eval FH_dataset/eval_GT`  
      就像是捷徑資料夾一樣，注意資料夾尾端不能任意添加 `/` !
3. Prepare environment:  
   使用與 HandMesh 相同的環境  
   如果要另外創一個: `pip install numpy==1.23.1 scipy==1.8.1 open3d==0.17.0 matplotlib==3.5.2 tqdm==4.64.0 scikit-learn==1.1.1 scikit-image==0.19.2`
4. Start to evaluate:
   1. 將 HandMesh 輸出之預測結果 `mrc_ds_50.json` 放在 `FH_dataset` 內
   2. 根據檔名調整 `eval.py> if __name__=='__main__':` 部分
