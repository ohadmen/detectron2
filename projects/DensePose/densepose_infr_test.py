


from tempfile import NamedTemporaryFile

import numpy as np
from pyk4a import PyK4A
import cv2

class Depth2xyz:
    def __init__(self,intrinsics):
        pass


def depth2gray(depth, min_depth=0.1, max_depth=3):
    out = (depth - min_depth) / (max_depth - min_depth)
    out = out.reshape(out.shape[0], out.shape[1], 1)
    out = np.repeat(out, 3, axis=2)
    return out


from projects.DensePose.desnepose_infr import DensePoseInference

if __name__ == "__main__":
    config_fpath = 'projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml'
    model_fpath = 'projects/DensePose/configs/weights/densepose_rcnn_R_50_FPN_s1x.pkl'
    opts = ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', '0.8']
    dpinfr = DensePoseInference(config_fpath=config_fpath, model_fpath=model_fpath, opts=opts)
    k4a = PyK4A()
    k4a.connect()
    calb_fn = NamedTemporaryFile()
    k4a.save_calibration_json(calb_fn.name)


    Depth2xyz()
    i = 0

    while True:
        img = k4a.get_capture()
        out = dpinfr.apply(img[0][:, :, :3])

        img_disp = np.concatenate([img[0][:, :, :3] / 255, depth2gray(img[1])], axis=1)
        img_disp = np.r_[img_disp, np.concatenate([out / [24, 1, 1], out * 0], axis=1)]
        img_disp = cv2.resize(img_disp, (img_disp.shape[1] // 2, img_disp.shape[0] // 2))
        cv2.imshow("results", img_disp)
        cv2.waitKey(1)
        print(i)
        i += 1
