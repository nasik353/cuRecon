
## Introduction

This is a modified PARENet code with two main added functionalities:

- Support for XYZ+RGB data instead of only coordinates
- Creating a custom 3D dataset synthesized from 3D models to simulate an object placed on a rotating table with a fixed camera recording from a certain pose.

We are only interested here in backbone part. As we will be using MAC for Feature Matching (KD tree) and Correspondence Selection.



<div align="center">
    <img src="assets/framework.png" alt="framework" width="700" >
</div>




## 🔧  Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n pareconv python==3.8
conda activate pareconv


pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop

cd pareconv/extentions/pointops/
python setup.py install
```

Code has been tested with Ubuntu 18.04, GCC 7.5.0, Python 3.9, PyTorch 1.13.0, CUDA 11.6 and cuDNN 8.0.5.

## 💾 Dataset and Pre-trained models
we provide pre-trained models in [Google Drive](https://drive.google.com/file/d/1jrPejtxnhMwtlr6LxXNJznm0BxSXP1eY/view?usp=drive_link). Please download the latest weights and put them in `pretain` directory.


Moreover, [3DMatch](https://drive.google.com/file/d/1_6tW-DREQdpGi4idLin8yITHuS_qXXnw/view?usp=drive_link) and [KITTI](https://drive.google.com/file/d/11OtJHWtX5y5dko2SiI3jCmw9zX305rJz/view?usp=drive_link) datasets can be downloaded from Google Drive or [Baidu Disk](https://pan.baidu.com/s/1mgqaelsAaV5Jx8o8nAseWQ) (extraction code: qyhn). 

##### 3DMatch should be organized as follows:
```text
--your_3DMatch_path--3DMatch
              |--train--7-scenes-chess--cloud_bin_0.pth
                    |--     |--...         |--...
              |--test--7-scenes-redkitchen--cloud_bin_0.pth
                    |--     |--...         |--...
              |--train_pair_overlap_masks--7-scenes-chess--masks_1_0.npz
                    |--     |--...         |--...       
```

Modify the dataset path in `experiments/3DMatch/config.py` to
```python
_C.data.dataset_root = '/your_3DMatch_path/3DMatch'
```

##### KITTI should be organized as follows:
```text
--your_KITTI_path--KITTI
            |--downsampled--00--000000.npy
                    |--...   |--... |--...
            |--train_pair_overlap_masks--0--masks_11_0.npz
                    |--...   |--... |--...
```                   

Modify the dataset path in `experiments/KITTI/config.py` to
```python
_C.data.dataset_root = '/your_KITTI_path/KITTI' 
```

## ⚽ Demo
After installation, you can run the demo script in `experiments/3DMatch` by:
```bash
cd experiments/3DMatch
python demo.py
```

To test your own data, you can downsample the point clouds with 2.5cm and specify the data path:
```bash
python demo.py --src_file=your_data_path/src.npy --ref_file=your_data_path/ref.npy --gt_file=your_data_path/gt.npy --weights=../../pretrain/3dmatch.pth.tar
```

## 🚅 Training
You can train a model on 3DMatch (or KITTI) by the following commands:

```bash
cd experiments/3DMatch (or KITTI)
CUDA_VISIBLE_DEVICES=0 python trainval.py
```
You can also use multiple GPUs by:
```bash
CUDA_VISIBLE_DEVICES=GPUS python -m torch.distributed.launch --nproc_per_node=NGPUS trainval.py
```
For example,
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trainval.py
```

## ⛳ Testing
To test a pre-trained models on 3DMatch, use the following commands:
```bash
# 3DMatch
python test.py --benchmark 3DMatch --snapshot ../../pretrain/3dmatch.pth.tar
python eval.py --benchmark 3DMatch
```
To test the model on 3DLoMatch, just change the argument `--benchmark 3DLoMatch`.

To test a pre-trained models on KITTI, use the similar commands:
```bash
# KITTI
python test.py --snapshot ../../pretrain/kitti.pth.tar
python eval.py
```

## Citation

```bibtex
@inproceedings{yao2024parenet,
    title={PARE-Net: Position-Aware Rotation-Equivariant Networks for Robust Point Cloud Registration},
    author={Runzhao Yao and Shaoyi Du and Wenting Cui and Canhui Tang and Chengwu Yang},
    journal={arXiv preprint arXiv:2407.10142},
    year={2024}
}
```

## Acknowledgements
Our code is heavily brought from
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [VectorNeurons](https://github.com/FlyingGiraffe/vnn)
- [PAConv](https://github.com/CVMI-Lab/PAConv)



