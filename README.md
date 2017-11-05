# pytorch-mask-rcnn
A pytorch implementation of Mask RCNN detection framework based on Xinlei Chen's [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn). Xinlei Chen's repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).

### Detection Performance (master branch)

With ResNet101 (last ``conv4``):
- Train on COCO 2014 trainval35k and test on minival (350k/450k), **30.7** (nms)
- Train on RefCOCO_trainval and test on RefCOCO_test (300K/400K), **25.7** (nms)
- Train on RefCOCOg_trainval and test on RefCOCOg_test (300K/400K), **27.6** (nms)

### Detection Performance (tf-compatible)
Run by (tf as extra/tag)
```bash
./experiments scripts/train_faster_rcnn.sh 0 refcoco res101 tf
```
- Train on RefCOCO_trainval and test on RefCOCO_test (300K/400K), **26.3** (nms)
- Train on RefCOCOg_trainval and test on RefCOCOg_test (300K/400K), **28.5** (nms)

### TODO
- mask evaluation
- script for mask prediction given boxes (from "stronger" Faster R-CNN results)