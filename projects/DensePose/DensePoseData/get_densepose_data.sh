#!/bin/bash
# get denspose uv
mkdir UV_data
cd UV_data
wget https://dl.fbaipublicfiles.com/densepose/densepose_uv_data.tar.gz
tar xvf densepose_uv_data.tar.gz
rm densepose_uv_data.tar.gz
cd ..


# get denspose COCO data
mkdir DensePose_COCO
cd DensePose_COCO
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_valminusminival.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_minival.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_test.json
cd ..

#get denspose eval data
mkdir eval_data
cd eval_data
wget https://dl.fbaipublicfiles.com/densepose/densepose_eval_data.tar.gz
tar xvf densepose_eval_data.tar.gz
rm densepose_eval_data.tar.gz
cd ..