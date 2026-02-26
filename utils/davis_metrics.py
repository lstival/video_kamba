import numpy as np
import torch
import cv2
from torchmetrics import Metric

def db_eval_iou(annotation, segmentation):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
    """
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1.0
    elif np.isclose(np.sum(annotation), 0) or np.isclose(np.sum(segmentation), 0):
        return 0.0

    inter = np.logical_and(annotation, segmentation)
    union = np.logical_or(annotation, segmentation)

    return np.sum(inter) / np.sum(union)

def db_eval_boundary(annotation, segmentation, bound_th=0.008):
    """ Compute boundary F-measure.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        bound_th     (float):   bounding threshold (normalized by image diagonal).
    Return:
        f_measure (float): boundary F-measure
    """
    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1.0
    elif np.isclose(np.sum(annotation), 0) or np.isclose(np.sum(segmentation), 0):
        return 0.0

    annotation = annotation.astype(np.uint8)
    segmentation = segmentation.astype(np.uint8)

    # get diagonal
    h, w = annotation.shape
    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm([h, w]))

    # get contours
    segmentation_bdry = cv2.Canny(segmentation * 255, 0, 1)
    segmentation_bdry_dilated = cv2.dilate(segmentation_bdry, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=int(bound_pix))
    
    annotation_bdry = cv2.Canny(annotation * 255, 0, 1)
    annotation_bdry_dilated = cv2.dilate(annotation_bdry, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=int(bound_pix))

    # compute precision & recall
    seg_bdry_pts = np.sum(segmentation_bdry) / 255.0
    ann_bdry_pts = np.sum(annotation_bdry) / 255.0

    if seg_bdry_pts == 0 or ann_bdry_pts == 0:
        return 0.0
        
    match_seg = np.sum(np.logical_and(segmentation_bdry > 0, annotation_bdry_dilated > 0))
    match_ann = np.sum(np.logical_and(annotation_bdry > 0, segmentation_bdry_dilated > 0))

    precision = match_seg / seg_bdry_pts
    recall = match_ann / ann_bdry_pts

    if precision + recall == 0:
        return 0.0

    f_measure = 2.0 * precision * recall / (precision + recall)
    return f_measure

def evaluate_j_f(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_objects: int):
    """
    Evaluates J and F scores for multiple objects.
    pred_mask: [H, W] or [T, H, W] tensor of class indices
    gt_mask: [H, W] or [T, H, W] tensor of class indices
    """
    pred_np = pred_mask.cpu().numpy()
    gt_np = gt_mask.cpu().numpy()
    
    # Ignore void index (255) by making prediction match background there
    pred_np[gt_np == 255] = 0
    
    j_scores = []
    f_scores = []
    
    # Evaluate per object ID (excluding background 0)
    for obj_id in range(1, num_objects + 1):
        pred_bin = (pred_np == obj_id)
        gt_bin = (gt_np == obj_id)
        
        # If it's a sequence, average over frames
        if pred_bin.ndim == 3:
            obj_j = []
            obj_f = []
            for t in range(pred_bin.shape[0]):
                j = db_eval_iou(gt_bin[t], pred_bin[t])
                f = db_eval_boundary(gt_bin[t], pred_bin[t])
                obj_j.append(j)
                obj_f.append(f)
            j_scores.append(np.mean(obj_j))
            f_scores.append(np.mean(obj_f))
        else:
            j = db_eval_iou(gt_bin, pred_bin)
            f = db_eval_boundary(gt_bin, pred_bin)
            j_scores.append(j)
            f_scores.append(f)
            
    return np.mean(j_scores) if j_scores else 1.0, np.mean(f_scores) if f_scores else 1.0

class DAVISMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("j_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("f_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target: [B, T, H, W] integer masks
        B = preds.shape[0]
        for b in range(B):
            valid_mask = target[b] != 255
            if not valid_mask.any():
                continue
            num_objects = int(target[b][valid_mask].max().item())
            if num_objects == 0:
                continue
            j, f = evaluate_j_f(preds[b], target[b], num_objects)
            self.j_sum += j
            self.f_sum += f
            self.total += 1

    def compute(self):
        total = max(1, self.total)
        return {
            "J": self.j_sum / total,
            "F": self.f_sum / total,
            "J&F": (self.j_sum + self.f_sum) / (2 * total)
        }
