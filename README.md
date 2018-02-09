# pytorch-mask-rcnn
A pytorch implementation of Mask RCNN detection framework based on 
* [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), developed based on TensorFlow + Numpy
* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), developed based on Pytorch + Numpy

This project supports single-GPU training of ResNet101-based Mask R-CNN (without FPN support). 
The purpose is to support the experiments in [Referring Expression Comprehension](http://gpuvision.cs.unc.edu/refer/), whose [REFER](https://github.com/lichengunc/refer) dataset is a subset of COCO training portion.
Thus our pre-trained model takes COCO_2014_train_minus_refer_valtest + COCO_2014_valminusminival images for training.


## Prerequisites
* Python 2.7
* Pytorch 0.2 or higher
* CUDA 8.0 or higher
* requirements.txt

## Preparation

1. First of all, clone the code with [refer API](https://github.com/lichengunc/refer):
```
git clone --recursive https://github.com/lichengunc/mask-faster-rcnn
```

2. Prepare data:

* **COCO**: We use `coco` to name COCO's API as inheritance. Download the [annotations and images](http://cocodataset.org/#download) into `data/coco`. Note the valminusminival and minival can be downloaded [here](https://github.com/rbgirshick/py-faster-rcnn/blob/77b773655505599b94fd8f3f9928dbf1a9a776c7/data/README.md). 
```shell
git clone https://github.com/cocodataset/cocoapi data/coco
```

* **REFER**: Follow the instructions in [REFER](https://github.com/lichengunc/refer) to prepare the annotations for RefCOCO, RefCOCO+ and RefCOCOg.

* **ImageNet Weights**: Find the resnet101-caffe download link from this [repository](https://github.com/ruotianluo/pytorch-resnet), and download it as `data/imagenet_weights/res101.pth`.

* **coco_minus_refer**: Make the `coco_minus_refer` annotation, which is to be saved as `data/coco/annotations/instances_train_minus_refer_valtest2014.json`
```shell
python tools/make_coco_minus_refer_instances.py
```


## Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

Compile the CUDA-based `nms` and `roi_pooling` using following simple commands:
```
cd lib
make
```

### Training 

Run by (`notime` as extra/tag)
```bash
./experiments/scripts/train_mask_rcnn_notime.sh 0 refcoco res101 notime
```
- Train on COCO 2014 [trainval35k](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) minus refer_valtest, and test on [minival](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) (800k/1250k), **35.8** on detection and **30.7** on segmentation (nms).


### Evaluation
Run by (`notime` as extra/tag)
```bash
./experiments/scripts/test_mask_rcnn_notime.sh 0 refcoco res101 notime
```

Detection Comparison:
- Compared with [Faster R-CNN](https://github.com/ruotianluo/pytorch-faster-rcnn) trained/evaluated on the same images. 
| Backbone |     AP^bb^ |    AP^bb^~50~  |    AP^bb^~75~  |
|----------|------------|----------------|----------------|
| res101   |  34.1      |    53.7        |    36.8        |
| res101   |  35.8      |    55.3        |    38.6        |


Segmentation Comparison:
- We compare with [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation. Note this comparison is slightly unfair to ours, due to
* We have fewer (~25,000) training images.
* Our training is single GPU.
* The shorter border length in our model is 600px instead of 800px.

| Backbone |     AP     |    AP~50~  |    AP~75~  |
|----------|------------|------------|------------|
| res101   |  30.7      |    52.3    |    32.4    |
| res101   |  32.7      |    54.2    |    34.0    |

### Demo













