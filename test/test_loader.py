from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../tools'))
import _init_paths
from datasets.coco import coco
from pprint import pprint
import time

imdb = coco('minival', '2014')
imdb.append_flipped_images()
print(len(imdb.roidb))
print(imdb.roidb[0].keys())

# # check image_set and roidb
# image_set = loader.image_set  # split -> image_ids
# roidb = loader.roidb
# for split in ['train', 'val', 'testA', 'testB']:
#   print('[%s] has %s roidb instances and %s images.' % (split, len(roidb), len(image_set[split])))

# # check appending
# print('After appending flipped...')
# loader.append_flipped_images('train')
# for split in ['train', 'val', 'testA', 'testB']:
#   print('[%s] has %s roidb instances and %s images.' % (split, len(roidb), len(image_set[split])))

# # check if image_id is consistent
# for split in ['train', 'val', 'testA', 'testB']:
#   num_images = len(loader.image_set[split])
#   for i in range(num_images):
#     assert loader.image_set[split][i] == loader.roidb[split][i]['image_id']

# # check one roidb instance
# roi = roidb['train'][0]
# ann_ids = roi['ann_ids']
# for ix, ann_id in enumerate(ann_ids):
#   print(ix, ann_id)
#   print(loader._classes[int(roi['gt_classes'][ix])])
#   for encode_type in ['NN', 'NP']:
#     h5_ids = loader.Anns[ann_id][encode_type+'_h5_ids']
#     labels = loader.fetch_labels(h5_ids, encode_type)
#     decoded = loader.decode_labels(labels)
#     print('encode_type[%s]: %s' % (encode_type, decoded))
#   print('\n')
# print(roi['boxes'])

# # check image_path
# image_path0 = loader.image_path_at(roidb['train'][0]['image_id'])
# image_path1 = loader.image_path_at(loader.image_set['train'][0])
# assert image_path0 == image_path1

# # check _class_to_synonyms
# tic = time.time()
# _class_to_synonyms = loader.collect_class_to_synonyms('NN')
# print('_class_to_synonyms made in %.2f seconds.' % (time.time() - tic))
# pprint(_class_to_synonyms)

