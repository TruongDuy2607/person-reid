# Person Re-ID: Datasets, Benchmarks and Models



This repo provides details about how to use [SOLIDER](https://github.com/tinyvision/SOLIDER) pretrained representation on person re-identification task.
We modify the code from [TransReID](https://github.com/damo-cv/TransReID), and you can refer to the original repo for more details.

## Installation and Datasets

We use python version 3.7, PyTorch version 1.7.1, CUDA 10.1 and torchvision 0.8.2. More details of installation and dataset preparation can be found in [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL).

## Prepare Pre-trained Models 
You can download models from [SOLIDER](https://github.com/tinyvision/SOLIDER), or use [SOLIDER](https://github.com/tinyvision/SOLIDER) to train your own models.
Before training, you should convert the models first.

```bash
python convert_model.py path/to/SOLIDER/log/lup/swin_tiny/checkpoint.pth path/to/SOLIDER/log/lup/swin_tiny/checkpoint_tea.pth
```

## Training

We utilize 1 GPU for training. Please modify the `MODEL.PRETRAIN_PATH`, `DATASETS.ROOT_DIR` and `OUTPUT_DIR` in the config file.

```bash
sh run.sh
```

## Benchmark evaluations

```bash
bash run_benchmark.sh
```

## Datasets

| Index | Dataset | Directory | Train | Test | Query |
| :----:| --- | --- | :---: | :---: | :---: |
| (1) | CUHK03 | ReID_Embedding/cuhk03.zip | 12514 | 1142 | 440 |
| (2) | DukeMEMC-ReID | ReID_Embedding/Unified-ReID-Dataset.zip | 16522| 17661 | 2228 |
| (3) | IUST | ReID_Embedding/Unified-ReID-Dataset.zip | 72393 | 45062 | 1428 |
| (4) | Market1501 | ReID_Embedding/Unified-ReID-Dataset.zip | 12937 | 19733 | 3369 |
| (5) | VNPT | - | 2856 | 633 | 41 |
| (6) | MSMT17 | - | - | - | - | 
| (7) | Entireid | ReID_Embedding/Unified-ReID-Dataset.zip |  | 10415 | 3000 |




## Performance

| Method | Backbone | Entireid<br>(w/o RK) | Entireid<br>(w RK) | VNPT<br>(w/o RK) | VNPT<br>(w RK) | Checkpoint/logs | Datasets |
| ------ | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SOLIDER| Swin Base | 49.8/50.65 | 54.34/51.28 | - | - | [solider_b]()/[logs]() | (1,2,3,4,5) |
| SOLIDER| Swin Small | 41.05/41.69 | 45.28/42.25 | - | - | [solider_s]()/[logs]() | (1,2,3,4,5) |
| OSNet | Resnet | 27.92/28.34 | 26.88/25.33 | - | - | [osnet]()/[logs]() | (1,2,4) |
| PersonViT | ViT | 23.15/22.88 | 21.54/19.80 | - | - | [osnet]()/[logs]() | (1,2,4) |

- `mAP/Rank1` are used as evaluation metric, `RK` indicates whether re-ranking is involved.
- `RK` shares the same models with `w/o RK`.
- We use the pretrained models from [SOLIDER](https://github.com/tinyvision/SOLIDER).
- The semantic weight is set to 0.2 in these experiments.

## Citation

If you find this code useful for your research, please cite SOLIDER paper

```
@inproceedings{chen2023beyond,
  title={Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks},
  author={Weihua Chen and Xianzhe Xu and Jian Jia and Hao Luo and Yaohua Wang and Fan Wang and Rong Jin and Xiuyu Sun},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```
