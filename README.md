# CAT: LoCalization and IdentificAtion Cascade Detection Transformer for Open-World Object Detection


# Abstract
Open-world object detection (OWOD), as a more general and challenging goal, requires the model trained from data on known objects to detect both known and unknown objects and incrementally learn to identify these unknown objects. For existing works which employ standard detection framework and fixed pseudo-labelling mechanism
(PLM), we observe the hindering problems. (𝑖) The inclusion of detecting unknown objects substantially reduces the model’s ability to detect known ones. (𝑖𝑖) The PLM does not adequately utilize the priori knowledge of inputs. (𝑖𝑖𝑖) The fixed manner of PLM cannot guarantee that the model is trained in the right direction. We observe that humans subconsciously prefer to focus on all foreground objects and then identify each one in detail, rather than localize and identify a single object simultaneously, for alleviating the confusion. This motivates us to propose a novel solution called CAT: LoCalization and IdentificAtion Cascade Detection Transformer which decouples the detection process via two cascade transformer decoders. In the meanwhile, we propose the self-adaptive pseudo-labelling mechanism which combines the model-driven with input-driven PLM and self-adaptively generates robust pseudo-labels for unknown objects, significantly improving the ability of CAT to retrieve unknown objects. Comprehensive experiments on two benchmark datasets, 𝑖.𝑒., MS-COCO and PASCAL VOC, show that our model outperforms the state-of-the-art in terms of all metrics in the task of OWOD, incremental object detection (IOD) and open-set detection.
![image](https://github.com/NEUfan/photo/blob/main/photo/429e05d98586eace2f25c4039531bbf.jpg)

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

### Results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=1>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center">4.9</td>
        <td align="center">56.0</td>
        <td align="center">2.9</td>
        <td align="center">39.4</td>
        <td align="center">3.9</td>
        <td align="center">29.7</td>
        <td align="center">25.3</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center">7.5</td>
        <td align="center">59.2</td>
        <td align="center">6.2</td>
        <td align="center">42.9</td>
        <td align="center">5.7</td>
        <td align="center">30.8</td>
        <td align="center">27.8</td>
    </tr>
    <tr>
        <td align="left">CAT</td>
        <td align="center">21.8</td>
        <td align="center">59.9</td>
        <td align="center">19.2</td>
        <td align="center">43.6</td>
        <td align="center">24.4</td>
        <td align="center">34.6</td>
        <td align="center">30.4</td>
    </tr>
    <tr>
        <td align="left">Weight</td>
        <td align="center">[T1] (https://drive.google.com/file/d/1wm7NwBBc1cY18wHQ_uCqzQrDkX43Svv4/view?usp=sharing)</td>
        <td align="center"> </td>
        <td align="center">[T2] (https://drive.google.com/file/d/1wm7NwBBc1cY18wHQ_uCqzQrDkX43Svv4/view?usp=sharing)</td>
        <td align="center"> </td>
        <td align="center">[T3] (https://drive.google.com/file/d/1wm7NwBBc1cY18wHQ_uCqzQrDkX43Svv4/view?usp=sharing)</td>
        <td align="center"> </td>
        <td align="center">[T4](https://drive.google.com/file/d/1wm7NwBBc1cY18wHQ_uCqzQrDkX43Svv4/view?usp=sharing)</td>
    </tr>
</table>


### OWDETR proposed splits

<br>
<p align="center" ><img width='500' src = "https://imgur.com/RlqbheH.png"></p> 
<br>

### Results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=1>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center">1.5</td>
        <td align="center">61.4</td>
        <td align="center">3.9</td>
        <td align="center">40.6</td>
        <td align="center">3.6</td>
        <td align="center">33.7</td>
        <td align="center">31.8</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center">5.7</td>
        <td align="center">71.5</td>
        <td align="center">6.2</td>
        <td align="center">43.8</td>
        <td align="center">6.9</td>
        <td align="center">38.5</td>
        <td align="center">33.1</td>
    </tr>
    <tr>
        <td align="left">CAT</td>
        <td align="center">24.0</td>
        <td align="center">74.2</td>
        <td align="center">23.0</td>
        <td align="center">50.7</td>
        <td align="center">24.6</td>
        <td align="center">45.0</td>
        <td align="center">42.8</td>
    </tr>
</table>

#### Dataset Preparation

The splits are present inside `data/VOC2007/CAT/ImageSets/` folder.
1. Make empty `JPEGImages` and `Annotations` directory.
```
mkdir data/VOC2007/CAT/JPEGImages/
mkdir data/VOC2007/CAT/Annotations_selective/
```
2. Download the COCO Images and Annotations from [coco dataset](https://cocodataset.org/#download).
3. Unzip train2017 and val2017 folder. The current directory structure should look like:
```
CAT/
└── data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```
4. Move all images from `train2017/` and `val2017/` to `JPEGImages` folder.
```
cd CAT/data
mv data/coco/train2017/*.jpg data/VOC2007/CAT/JPEGImages/.
mv data/coco/val2017/*.jpg data/VOC2007/CAT/JPEGImages/.
```
5. **Annotations_selective** link: [Annotations_selective](https://drive.google.com/drive/folders/1dsuwZyM0I-c2yl8IIR7gu7frkGjLnn0h?usp=share_link)

The files should be organized in the following structure:
```
CAT/
└── data/
    └── VOC2007/
        └── OWOD/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations_selective
```


Currently, Dataloader and Evaluator followed for CAT is in VOC format.



    
# Training

#### Training on single node

To train CAT on a single node with 8 GPUS, run
```bash
./run.sh
```

#### Training on slurm cluster

To train CAT on a slurm cluster having 2 nodes with 8 GPUS each, run
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
~~~
@inproceedings{Ma2023CATLA,
  title={CAT: LoCalization and IdentificAtion Cascade Detection Transformer for Open-World Object Detection},
  author={Shuailei Ma and Yuefeng Wang and Jiaqi Fan and Ying-yu Wei and Thomas H. Li and Hongli Liu and Fanbing Lv},
  year={2023}
}
~~~

# Contact

Should you have any question, please contact :e-mail: xiaomabufei@gmail.com

**Acknowledgments:**

CAT builds on previous works code base such as [OWDETR](https://github.com/akshitac8/ow-detr),[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [Detreg](https://github.com/amirbar/DETReg), and [OWOD](https://github.com/JosephKJ/OWOD). If you found CAT useful please consider citing these works as well.

