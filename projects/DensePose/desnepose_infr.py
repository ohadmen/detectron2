import argparse
from typing import List

import torch
import base64
from io import BytesIO

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import argparse
import glob
import logging
import os
import pickle
import sys
from typing import Any, ClassVar, Dict, List

import cv2
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config, add_hrnet_config


class DensePoseResultExtractor(object):
    """
    Extracts DensePose result from instances
    """

    @staticmethod
    def _extract_boxes_xywh_from_instances(instances, select=None):
        if instances.has("pred_boxes"):
            boxes_xywh = instances.pred_boxes.tensor.clone()
            boxes_xywh[:, 2] -= boxes_xywh[:, 0]
            boxes_xywh[:, 3] -= boxes_xywh[:, 1]
            return boxes_xywh if select is None else boxes_xywh[select]
        return None

    def __call__(self, instances, select=None):
        boxes_xywh = self._extract_boxes_xywh_from_instances(instances)
        if instances.has("pred_densepose") and (boxes_xywh is not None):
            dpout = instances.pred_densepose
            if select is not None:
                dpout = dpout[select]
                boxes_xywh = boxes_xywh[select]
            return dpout.to_result(boxes_xywh)
        else:
            return None


class DensePoseInference:

    @staticmethod
    def setup_config(
            config_fpath: str, model_fpath: str, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        add_hrnet_config(cfg)
        cfg.merge_from_file(config_fpath)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @staticmethod
    def _decode_png_data(shape, s):
        """
        Decode array data from a string that contains PNG-compressed data
        @param Base64-encoded string containing PNG-compressed data
        @return Data stored in an array of size (3, M, N) of type uint8
        """
        fstream = BytesIO(base64.decodebytes(s.encode()))
        im = Image.open(fstream)
        data = np.moveaxis(np.array(im.getdata(), dtype=np.uint8), -1, 0)
        return data.reshape(shape)

    def __init__(self, config_fpath: str, model_fpath: str, opts: List = []):
        cfg = self.setup_config(config_fpath, model_fpath, opts)
        self.predictor = DefaultPredictor(cfg)
        self.extractor = DensePoseResultExtractor()
        pass

    def apply(self, rgbimg):
        h, w, d = rgbimg.shape
        if d != 3:
            raise RuntimeError("expected RGB image")
        out_iuv = np.zeros((h, w, 3))
        with torch.no_grad():
            outputs = self.predictor(rgbimg)["instances"]
            data = self.extractor(outputs)

            for j in range(len(data.results)):
                iuv_arr = np.moveaxis(self._decode_png_data(*data.results[j]), 0, -1)
                bbox_xywh = data.boxes_xywh[j]
                bbox_xywh = np.array(bbox_xywh).astype(int)
                out_iuv[bbox_xywh[1]:bbox_xywh[1] + bbox_xywh[3], bbox_xywh[0]:bbox_xywh[0] + bbox_xywh[2],
                :] = iuv_arr

        out_iuv = out_iuv / [1, 255, 255]
        return out_iuv
