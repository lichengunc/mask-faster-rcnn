"""
Make coco_2014_suerminival.json used for testing.
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

K = 20
coco_data_path = 'data/coco/annotations'

data = json.load(open(osp.join(coco_data_path, 'instances_minival2014.json')))
data['images'] = data['images'][:K]

with open(osp.join(coco_data_path, 'instances_superminival2014.json'), 'w') as io:
  json.dump(data, io)