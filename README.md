# # CAT: LoCalization and IdentificAtion Cascade Detection Transformer for Open-World Object Detection


# Installation

### Requirements

We have trained and tested our models on `Ubuntu 16.0`, `CUDA 10.2`, `GCC 5.4`, `Python 3.7`

```bash
conda create -n cat python=3.7 pip
conda activate cat
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Backbone features

Download the self-supervised backbone from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) and add in `models` folder.

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```


# Dataset & Results

### OWOD proposed splits
<br>
<p align="center" ><img width='500' src = "https://imgur.com/9bzf3DV.png"></p> 
<br>

The splits are present inside `data/VOC2007/OWOD/ImageSets/` folder. The remaining dataset can be downloaded using this [link](https://drive.google.com/drive/folders/11bJRdZqdtzIxBDkxrx2Jc3AhirqkO0YV)

The files should be organized in the following structure:
```
OW-DETR/
└── data/
    └── VOC2007/
        └── OWOD/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```

### Results

![image](https://user-images.githubusercontent.com/104605826/210916496-e63bc151-bc1e-4608-8713-1f0c1bc54e6f.png)


### OWDETR proposed splits

<br>
<p align="center" ><img width='500' src = "https://imgur.com/RlqbheH.png"></p> 
<br>

#### Dataset Preparation

The splits are present inside `data/VOC2007/OWDETR/ImageSets/` folder.
1. Make empty `JPEGImages` and `Annotations` directory.
```
mkdir data/VOC2007/OWDETR/JPEGImages/
mkdir data/VOC2007/OWDETR/Annotations/
```
2. Download the COCO Images and Annotations from [coco dataset](https://cocodataset.org/#download).
3. Unzip train2017 and val2017 folder. The current directory structure should look like:
```
OW-DETR/
└── data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```
4. Move all images from `train2017/` and `val2017/` to `JPEGImages` folder.
```
cd OW-DETR/data
mv data/coco/train2017/*.jpg data/VOC2007/OWDETR/JPEGImages/.
mv data/coco/val2017/*.jpg data/VOC2007/OWDETR/JPEGImages/.
```
5. Use the code `coco2voc.py` for converting json annotations to xml files.

The files should be organized in the following structure:
```
OW-DETR/
└── data/
    └── VOC2007/
        └── OWDETR/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```


Currently, Dataloader and Evaluator followed for OW-DETR is in VOC format.

### Results

![image](https://user-images.githubusercontent.com/104605826/210916691-fb04f87a-496e-408b-aba5-d5300cb6b677.png)

    
# Training

#### Training on single node

To train OW-DETR on a single node with 8 GPUS, run
```bash
./run.sh
```

#### Training on slurm cluster

To train OW-DETR on a slurm cluster having 2 nodes with 8 GPUS each, run
```bash
sbatch run_slurm.sh
```

# Evaluation

For reproducing any of the above mentioned results please run the `run_eval.sh` file and add pretrained weights accordingly.


**Note:**
For more training and evaluation details please check the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) reposistory.

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


# Citation

If you use CAT, please consider citing:


# Contact

Should you have any question, please contact :e-mail: xiaomabufei@gmail.com

**Acknowledgments:**

CAT builds on previous works code base such as [OWDETR](https://github.com/akshitac8/ow-detr),[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [Detreg](https://github.com/amirbar/DETReg), and [OWOD](https://github.com/JosephKJ/OWOD). If you found OW-DETR useful please consider citing these works as well.

