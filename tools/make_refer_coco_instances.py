"""
Make refer instances.
We will use refcoco(+)_unc and refcocog_umd. The reason we do not 
use refcocog_google is its overlap between train and val on image
set. Specifically, we will make the follows:
1) refcoco_train.json, refcoco_trainval.json, refcoco_val.json, refcoco_test.json
2) refcocog_train.json, refcocog_trainval.json, refcocog_val.json, refcocog_test.json
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

train_image_ids = []
val_image_ids = []
test_image_ids = []

if params['dataset'] == 'refcoco':
  refer = REFER(refer_data_path, params['dataset'], 'unc')
elif params['dataset'] == 'refcocog':
  refer = REFER(refer_data_path, params['dataset'], 'umd')
else:
  raise ValueError('[%s] not considered yet.' % params['dataset'])

for _, ref in refer.Refs.items():
  if 'test' in ref['split']:
    test_image_ids.append(ref['image_id'])
  elif 'val' in ref['split']:
    val_image_ids.append(ref['image_id'])
  elif 'train' in ref['split']:
    train_image_ids.append(ref['image_id'])
  else:
    raise ValueError('No such split')
train_image_ids = list(set(train_image_ids))
val_image_ids = list(set(val_image_ids))
test_image_ids = list(set(test_image_ids))
trainval_image_ids = list(set(train_image_ids + val_image_ids))
assert len(set(trainval_image_ids).intersection(set(test_image_ids))) == 0, 'trainval and test have overlaps'
print('[%s] has %s train images, %s val images, and %s test images.' % \
      (params['dataset'], len(train_image_ids), len(val_image_ids), len(test_image_ids)))

#######################################################################
# make refcoco_trainval.json and refcoco_test.json
# save into data/refer_coco/annotations
#######################################################################
def make_instances(my_image_ids, data):
  my_data = {}
  for k in ['info', 'licenses', 'categories']:
    my_data[k] = data[k]
  my_data['images'] = [image for image in data['images'] if image['id'] in my_image_ids]
  my_data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in my_image_ids]
  return my_data

# make dir
if not osp.isdir(refer_coco_data_path):
  os.makedirs(refer_coco_data_path)

# load coco data
coco_data = json.load(open(osp.join(coco_data_path, 'instances_train2014.json')))
print('raw COCO instances_train2014.json loaded.')

# make train
refer_train = make_instances(train_image_ids, coco_data)
print('[%s]\'s [train] has %s images and %s anns.' % (params['dataset'], len(refer_train['images']), len(refer_train['annotations'])))
with open(osp.join(refer_coco_data_path, params['dataset']+'_train.json'), 'w') as io:
  json.dump(refer_train, io)

# make val
refer_val = make_instances(val_image_ids, coco_data)
print('[%s]\'s [val] has %s images and %s anns.' % (params['dataset'], len(refer_val['images']), len(refer_val['annotations'])))
with open(osp.join(refer_coco_data_path, params['dataset']+'_val.json'), 'w') as io:
  json.dump(refer_val, io)

# make test
refer_test = make_instances(test_image_ids, coco_data)
print('[%s]\'s [test] has %s images and %s anns.' % (params['dataset'], len(refer_test['images']), len(refer_test['annotations'])))
with open(osp.join(refer_coco_data_path, params['dataset']+'_test.json'), 'w') as io:
  json.dump(refer_test, io)

