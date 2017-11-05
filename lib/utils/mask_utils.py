from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imresize
from pycocotools import mask as COCOmask

def segmToMask(segm, h, w):
  """
  segm  : coco annotated segmentation
  output: mask ndarray uint8 (im_height, im_width), range {0,1}
  """
  if type(segm) == list:
    # polygon -- a single object might consist of multiple parts
    # we merge all parts into one mask rle code
    rles = COCOmask.frPyObjects(segm, h, w)
    rle = COCOmask.merge(rles)
  elif type(segm['counts']) == list:
    # uncompressed RLE
    rle = COCOmask.frPyObjects(segm, h, w)
  else:
    raise NotImplementedError

  m = COCOmask.decode(rle)  # binary mask (numpy 2D array)
  return m

def clip_np_boxes(boxes, im_shape):
  """
  Clip boxes to image boundaries.
  boxes: ndarray float32 (n, 4) [xyxy]
  """
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
  return boxes

def recover_masks(masks, rois, ih, iw, interp='bilinear'):
  """Decode 14x14 masks into final masks
  Params
  - masks : of shape (N, 14, 14) float32, ranging [0, 1]
  - rois  : of shape (N, 4) [x1, y1, x2, y2] float32. Note there is no batch_ids in rois!
  - ih    : image height
  - iw    : image width
  - interp: bilinear or nearest 
  Returns
  - recovered_masks : of shape (N, ih, iw) uint8, range [0, 255]
  """
  assert rois.shape[0] == masks.shape[0], '%s rois vs %d masks'%(rois.shape[0], masks.shape[0])

  num_rois = rois.shape[0]
  recovered_masks = np.zeros((num_rois, ih, iw), dtype=np.uint8) # (num_rois, ih, iw)
  rois = clip_np_boxes(rois, (ih, iw))
  for i in np.arange(num_rois):
    # read mask of (14, 14) float32
    mask = masks[i, :, :]
    # range [0, 255] float32
    mask *= 255.
    # resize will convert it to uint8 [0, 255]
    h, w = int(rois[i, 3] - rois[i, 1] + 1), int(rois[i, 2] - rois[i, 0] + 1)
    x, y = int(rois[i, 0]), int(rois[i, 1])
    mask = imresize(mask, (h, w), interp=interp) # (roi_h, roi_w) uint8
    # paint
    recovered_masks[i, y:y+h, x:x+w] = mask

  return recovered_masks


def recover_cls_masks(masks, rois, ih, iw, interp='bilinear'):
  """Decode 14x14 masks into final masks
  Arguments
  - masks : (N, C, 14, 14) float32, ranging [0,1]
  - rois  : (N, 4) [xyxy] float32
  - ih    : image height
  - iw    : image width
  - interp: bilinear or nearest
  Returns
  - recovered_masks : (N, ih, iw) uint8, range [0, 255]
  """
  assert rois.shape[0] == masks.shape[0], '%s rois vs %d masks'%(rois.shape[0], masks.shape[0])

  num_rois = rois.shape[0]
  num_classes = masks.shape[1]
  recovered_masks = np.zeros((num_rois, num_classes, ih, iw), dtype=np.uint8) # (num_rois, ih, iw)
  rois = clip_np_boxes(rois, (ih, iw))
  for i in np.arange(num_rois):
    # read mask of (C, 14, 14) float32
    mask = masks[i, :, :, :]
    # range [0, 255] float32
    mask *= 255.
    # resize
    h, w = int(rois[i, 3] - rois[i, 1] + 1), int(rois[i, 2] - rois[i, 0] + 1)
    x, y = int(rois[i, 0]), int(rois[i, 1])
    for c in range(num_classes):
      m = mask[c] # (14, 14)
      m = imresize(m, (h, w), interp=interp) # (roi_h, roi_w) uint8
      recovered_masks[i, c, y:y+h, x:x+w] = m

  return recovered_masks


