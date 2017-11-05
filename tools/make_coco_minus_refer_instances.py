"""
Make refcoco instances.
We will use all coco_2014_train images, but excluding
- refcoco_unc  (val/testA/testB)
- refcocog_umd (val/test)
sets. 

Specifically, we will make the follows:
coco_2014_train_minus_refer_valtest.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import os
import os.path as osp
import _init_paths
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='refcoco', help='refcoco or refcocog')
args = parser.parse_args()
params = vars(args)

# set path
refer_data_path = 'data/refer/data'
coco_data_path = 'data/coco/annotations'
refer_coco_data_path = 'data/refer_coco/annotations'

#######################################################################
# accumulate refer_trainval_ids and refer_test_ids
#######################################################################
from refer import REFER

refcoco = REFER(refer_data_path, 'refcoco', 'unc')
refcoco_excluded_image_ids = []
for _, ref in refcoco.Refs.items():
  if ref['split'] in ['testA', 'testB', 'val']:
    refcoco_excluded_image_ids += [ref['image_id']]
refcoco_excluded_image_ids = list(set(refcoco_excluded_image_ids))
print('In refcoco_unc, %s valtest image_ids will be excluded.' % len(refcoco_excluded_image_ids))

refcocog = REFER(refer_data_path, 'refcocog', 'umd')
refcocog_excluded_image_ids = []
for _, ref in refcocog.Refs.items():
  if ref['split'] in ['test', 'val']:
    refcocog_excluded_image_ids += [ref['image_id']]
refcocog_excluded_image_ids = list(set(refcocog_excluded_image_ids))
print('In refcocog_umd, %s valtest image_ids will be excluded.' % len(refcocog_excluded_image_ids))

excluded_image_ids = list(set(refcoco_excluded_image_ids + refcocog_excluded_image_ids))
print('In total, %s refcoco_unc+refcocog_umd images will be excluded.' % len(excluded_image_ids))

#######################################################################
# make coco_2014_train_minus_refcoco_test.json, which is to save
# instances_train_minus_refcoco_test2014.json into data/coco/annotations
#######################################################################
data = json.load(open(osp.join(coco_data_path, 'instances_train2014.json')))

print('Before pruning [%s] test, COCO has %s images.' % (params['dataset'], len(data['images'])))
data['images'] = [image for image in data['images'] if image['id'] not in excluded_image_ids]
print('After pruning [%s] test, COCO has %s images.' % (params['dataset'], len(data['images'])))

with open(osp.join(coco_data_path, 'instances_train_minus_refer_valtest2014.json'), 'w') as io:
  json.dump(data, io)













