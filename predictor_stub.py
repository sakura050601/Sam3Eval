import numpy as np
import cv2
import torch

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
        pred[y0:y1+1, x0:x1+1] = 1
        return pred

class Sam3Predictor:
    """
    固定接口：
    - __init__(ckpt, device)
    - predict(image_bgr, gt_bbox_xyxy=None) -> (H,W) 0/1 mask
    """
    def __init__(self, ckpt, device="cuda"):
        self.device = device

        # 1) 用你已经能 import 的 builder
        from sam3.model_builder import build_sam3_image_model

        # 2) 不同repo这里参数名可能不同：ckpt_path / ckpt / checkpoint
        try:
            model = build_sam3_image_model(ckpt_path=ckpt)
        except TypeError:
            try:
                model = build_sam3_image_model(ckpt=ckpt)
            except TypeError:
                model = build_sam3_image_model(checkpoint=ckpt)

        model = model.to(device)
        model.eval()
        self.model = model

        # 3) 你们 repo 里如果有 Predictor 类，就用它（最像官方用法）
        # 如果没有 predictor 类，我们就直接调用 model 的推理函数（需要你改一行）。
        self.predictor = None
        try:
            # 这行很可能需要你改成真实路径/类名（例如 sam3.predictor / sam3.sam3_predictor）
            from sam3.predictor import Sam3Predictor as _Pred
            self.predictor = _Pred(self.model)
        except Exception:
            self.predictor = None

    @torch.no_grad()
    def predict(self, image_bgr, gt_bbox_xyxy=None):
        """
        gt_bbox_xyxy: [x0,y0,x1,y1] float/int
        """
        if gt_bbox_xyxy is None:
            # 没bbox就返回空（你也可以改成整图prompt）
            h, w = image_bgr.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # BGR -> RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        box = np.asarray(gt_bbox_xyxy, dtype=np.float32)

        # ===== 路线A：你们有 predictor（推荐）=====
        if self.predictor is not None:
            # 常见接口：set_image + predict(box=...)
            if hasattr(self.predictor, "set_image"):
                self.predictor.set_image(image_rgb)
            elif hasattr(self.predictor, "set_images"):
                self.predictor.set_images([image_rgb])
            else:
                raise RuntimeError("predictor没有 set_image/set_images，请打印dir(self.predictor)看实际接口")

            # 常见：predict(box=xyxy) / predict(boxes=[xyxy])
            if hasattr(self.predictor, "predict"):
                try:
                    masks = self.predictor.predict(box=box)
                except TypeError:
                    masks = self.predictor.predict(boxes=np.array([box], dtype=np.float32))
            else:
                raise RuntimeError("predictor没有 predict()，请打印dir(self.predictor)看实际接口")

            masks = np.asarray(masks)
            if masks.ndim == 3:
                masks = masks[0]
            pred = (masks > 0).astype(np.uint8)
            return pred

        # ===== 路线B：没有 predictor 类，就直接调 model（你需要按实际接口改这一段）=====
        # 你需要把下面这行替换成你们 model 的“给定图像+box输出mask”的真实函数。
        # 例如：masks = self.model.infer(image_rgb, box) 或 self.model.predict(...)
        raise RuntimeError(
            "没有找到 sam3.predictor.Sam3Predictor。你需要用你们model的真实推理接口替换路线B那一段。"
        )