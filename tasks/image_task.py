"""
Image (2D) task: VOC2007.

Detection contract produced for every image detector:
    {"boxes": [N, 4] xyxy (original image pixels), "scores": [N], "labels": [N]}
"""

import time

import torch
from PIL import Image

from common import config as C
from common.datasets.voc import VOCDetection, letterbox_params
from common.geometry import batched_nms_2d, box_iou_2d
from common.progress import Progress
from common.metrics import evaluate_detections, print_results
from common.trainer import make_loader


class ImageTask:
    name = "image"

    def __init__(self):
        self.spec = C.TASKS["image"]
        self.class_names = list(self.spec["classes"])

    # ------------------------------------------------------------ data
    def train_set(self, args):
        return VOCDetection(C.VOC_ROOT, self.spec["train_split"], args.img_size, self.class_names,
                            train=True, hflip_prob=C.PROFILE["hflip_prob"], max_samples=args.max_train,
                            aug=C.PROFILE["image_aug"])

    def test_set(self, args):
        return VOCDetection(C.VOC_ROOT, self.spec["test_split"], args.img_size, self.class_names,
                            train=False, max_samples=args.max_test)

    # ------------------------------------------------------------ post-process
    def postprocess(self, raw, orig_sizes, img_size, conf_thr):
        """raw [B, N, 5 + C]: cx, cy, w, h (input px), objectness, class probs."""
        P = C.PROFILE
        out = []
        for pred, (w, h) in zip(raw, orig_sizes):
            cls_conf, cls_id = pred[:, 5:].max(1)
            score = pred[:, 4] * cls_conf
            keep = score >= conf_thr
            pred, score, cls_id = pred[keep], score[keep], cls_id[keep]
            if score.numel() > P["pre_nms_topk"]:
                top = score.topk(P["pre_nms_topk"]).indices
                pred, score, cls_id = pred[top], score[top], cls_id[top]
            boxes = torch.stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2,
                                 pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2], 1)
            k = batched_nms_2d(boxes, score, cls_id, P["nms_iou"])[: P["max_det"]]
            boxes, score, cls_id = boxes[k], score[k], cls_id[k]
            scale, px, py, _, _ = letterbox_params(w, h, img_size)
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / scale).clamp(0, w - 1)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / scale).clamp(0, h - 1)
            out.append({"boxes": boxes.cpu().numpy(), "scores": score.cpu().numpy(),
                        "labels": cls_id.cpu().numpy()})
        return out

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, model, dataset, args, device):
        model.eval()
        loader = make_loader(dataset, args.batch_size, args.workers, False, args.seed)
        dets, gts = [], []
        t0 = time.time()
        bar = Progress(len(loader), f"{args.run_name} | test inference")
        for bi, (idx, images, _) in enumerate(loader, 1):
            raw = model(images.to(device, dtype=torch.float32))
            sizes = [Image.open(dataset.image_path(i)).size for i in idx]
            dets += self.postprocess(raw, sizes, args.img_size, C.PROFILE["eval_conf"])
            gts += [dataset.annotation(i) for i in idx]
            bar.update()
        bar.close()
        fps = len(dataset) / max(time.time() - t0, 1e-9)
        res = evaluate_detections(dets, gts, self.class_names, box_iou_2d, C.PROFILE["eval_iou"])
        print_results(f"{self.spec['dataset'].upper()} {self.spec['test_split']} ({len(dataset)} images)", res)
        print(f"Inference speed incl. post-process: {fps:.1f} img/s")
        return {"metric": "VOC AP50 (difficult ignored)", "images": len(dataset), "fps": fps, "ap50": res}
