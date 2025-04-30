# [TMM2025] Tackling Ambiguity from Perspective of Uncertainty Inference and Affinity Diversification for Weakly Supervised Semantic Segmentation [![arXiv](https://img.shields.io/badge/arXiv-2303.02506-b31b1b.svg)](https://arxiv.org/pdf/2404.08195.pdf)
## News

* **Our UniA is accepted by TMM 2025.** 
* All Code, logs, and checkpoints are available now🔥🔥🔥
* If you have any questions, please feel free to leave issues or contact us by zwyang21@m.fudan.edu.cn.

## Overview

<p align="middle">
<img src="/sources/Fig_Uni.png" alt="UniA pipeline" width="1200px">
</p>
Weakly supervised semantic segmentation (WSSS) with image-level labels aims to achieve dense predictions without laborious annotations. However, due to the ambiguous contexts and fuzzy regions, the performance of WSSS, particularly during the stages of generating Class Activation Maps (CAMs) and refining pseudo masks, is widely hindered by ambiguity. Despite this, this issue has received little attention in previous literature. In this work, we propose UniA, a unified single-staged WSSS framework, to efficiently tackle this issue from the perspectives of uncertainty inference and affinity diversification. When activating class objects, we argue that the false activation stems from the bias to ambiguous regions during the feature extraction. Therefore, we formulate a robust feature representation with a Gaussian distribution and introduce the uncertainty estimation to avoid the bias. A distribution loss is proposed to supervise the process, which effectively captures the ambiguity and models the complex dependencies among features. When refining pseudo labels, we observe that the affinity from the prevailing refinement methods intends to be overly similar among ambiguities. To this end, we design an affinity diversification module to promote diversity among semantics. A mutual complementing refinement is first proposed to statically rectify the ambiguous affinity with multiple inferred pseudo labels. Then a contrastive affinity loss is further designed to dynamically diversify the relations among unrelated semantics. It stably propagates the diversity into the feature representation and helps generate better pseudo masks. Extensive experiments are conducted on PASCAL VOC, MS COCO, and medical ACDC datasets, which validate the efficiency of UniA tackling ambiguity and its superiority over recent single- staged or even most multi-staged competitors.

## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). The download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012/`. 

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### MSCOCO 2014

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

#### 2. Segmentation Labels

To generate VOC style segmentation labels for COCO, you could use the scripts provided at this [repo](https://github.com/alicranck/coco2voc), or just download the generated masks from [Google Drive](https://drive.google.com/file/d/147kbmwiXUnd2dW9_j8L5L0qwFYHUcP9I/view?usp=share_link).

``` bash
COCO/
├── JPEGImages
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```

## Requirement

Please refer to the requirements.txt. 

We incorporate a regularization loss for segmentation. Please refer to the instruction for this [python extension](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module).

## Train UniA
``` bash
### train voc
bash run_train.sh scripts/train_voc.py [gpu_number] [master_port] [gpu_device] train_voc

### train coco
bash run_train.sh scripts/train_coco.py [gpu_numbers] [master_port] [gpu_devices] train_coco
```

## Evaluate UniA
``` bash
### eval voc
bash run_evaluate_seg_voc.sh tools/infer_seg_voc.py [gpu_device] [checkpoint_path]

### eval coco
bash run_evaluate_seg_coco.sh tools/infer_seg_coco.py [gpu_number] [master_port] [gpu_device] [checkpoint_path]
```


## Main Results

* **Quantitative Results**
  
Semantic performance on VOC and COCO. Logs are available now.
| Dataset | Backbone |  Val  | Test | Log | Weight |
|:-------:|:--------:|:-----:|:----:|:---:|:------:|
|   PASCAL VOC   |   ViT-B  | 74.1  | [73.6](http://host.robots.ox.ac.uk:8080/anonymous/YZGTOM.html) | [log](logs/voc_train.log) | [checkpoints](https://drive.google.com/file/d/1q0YorSB2gTDwQ1d5VkvMledlXfWe0mqe/view?usp=drive_link)      |
|   MS COCO  |   ViT-B  |  43.2 |   -  | [log](logs/coco_train.log) | [checkpoints](https://drive.google.com/file/d/1S_M2lPQcXsomGqfyoljEcUDhX5lT77ZR/view?usp=drive_link)       |

* **Qualitative Results**
  
<p align="middle">
<img src="/sources/results1.png" alt="UniA results" width="1200px">
</p>

## Citation 
Please cite our work if you find it helpful to your reseach. :two_hearts:
```bash
@article{yang2024tackling,
  title={Tackling Ambiguity from Perspective of Uncertainty Inference and Affinity Diversification for Weakly Supervised Semantic Segmentation},
  author={Yang, Zhiwei and Meng, Yucong and Fu, Kexue and Wang, Shuo and Song, Zhijian},
  journal={arXiv preprint arXiv:2404.08195},
  year={2024}
}
```
