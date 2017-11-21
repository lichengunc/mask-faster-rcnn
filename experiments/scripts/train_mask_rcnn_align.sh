#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  coco_minus_refer)
    TRAIN_IMDB="coco_2014_train_minus_refer_valtest+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[800000]"
    ITERS=1250000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_mask_rcnn_iter_${ITERS}.pth
else
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_mask_rcnn_iter_${ITERS}.pth
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
    if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
            --weight data/imagenet_weights/${NET}.pth \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/${NET}_align.yml \
            --tag ${EXTRA_ARGS_SLUG} \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
            TRAIN.STEPSIZE ${STEPSIZE} # ${EXTRA_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
            --weight data/imagenet_weights/${NET}.pth \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/${NET}_align.yml \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
            TRAIN.STEPSIZE ${STEPSIZE} # ${EXTRA_ARGS}
    fi
fi

./experiments/scripts/test_mask_rcnn_align.sh $@
