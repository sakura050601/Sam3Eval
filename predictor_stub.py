import numpy as np
import cv2


class NoModelPredictor:
    """
    先用它跑通：保证“可运行+可视化+指标输出”。
    mode="bbox": 用GT bbox 生成一个矩形mask（粗baseline）
    mode="empty": 全背景
    """

    def __init__(self, mode="bbox"):
        self.mode = mode

    def predict(self, image_bgr, gt_bbox_xyxy=None):
        h, w = image_bgr.shape[:2]
        pred = np.zeros((h, w), dtype=np.uint8)
        if self.mode == "empty" or gt_bbox_xyxy is None:
            return pred
        x0, y0, x1, y1 = [int(v) for v in gt_bbox_xyxy]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        pred[y0 : y1 + 1, x0 : x1 + 1] = 1
        return pred


class Sam3Predictor:
    """
    你以后把SAM3接进来，只需要实现这个类：
    - __init__(ckpt, device)
    - predict(image_bgr, gt_bbox_xyxy=None) -> (H,W) 0/1 mask
    """

    def __init__(self, ckpt, device="cuda"):
        raise NotImplementedError("把你的SAM3推理代码写在这里即可")

    def predict(self, image_bgr, gt_bbox_xyxy=None):
        raise NotImplementedError
