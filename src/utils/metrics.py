# src/utils/metrics.py
import numpy as np
from typing import Dict, List, Tuple
import torch
from collections import defaultdict

def calculate_map(predictions: List[torch.Tensor], 
                 targets: List[torch.Tensor], 
                 iou_thresholds: List[float] = None) -> Dict:
    """Calculate mean Average Precision"""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    metrics = {
        'precision': [],
        'recall': [],
        'map50': 0.0,
        'map50_95': 0.0,
        'class_accuracy': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    }
    
    # Calculate AP for each IoU threshold
    aps = []
    for iou_thresh in iou_thresholds:
        ap, precision, recall = calculate_ap_single_iou(
            predictions, targets, iou_thresh)
        aps.append(ap)
        
        if iou_thresh == 0.5:
            metrics['map50'] = ap
            metrics['precision'] = precision
            metrics['recall'] = recall
    
    metrics['map50_95'] = np.mean(aps)
    
    # Calculate per-class metrics
    class_metrics = calculate_class_metrics(predictions, targets)
    metrics.update(class_metrics)
    
    return metrics

# src/utils/metrics.py (continued)

def calculate_ap_single_iou(predictions: List[torch.Tensor], 
                          targets: List[torch.Tensor], 
                          iou_threshold: float) -> Tuple[float, float, float]:
    """Calculate Average Precision for a single IoU threshold"""
    all_detections = []
    all_ground_truths = []
    
    # Collect all detections and ground truths
    for pred, target in zip(predictions, targets):
        all_detections.extend([{
            'confidence': float(p[4]),
            'class_id': int(p[5]),
            'bbox': p[:4].cpu().numpy()
        } for p in pred])
        
        all_ground_truths.extend([{
            'class_id': int(t[0]),
            'bbox': t[1:5].cpu().numpy()
        } for t in target])
    
    # Sort detections by confidence
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Initialize counters
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    n_gts = len(all_ground_truths)
    
    # Mark ground truths as unmatched
    gt_matched = {i: False for i in range(n_gts)}
    
    # Calculate TP and FP
    for i, detection in enumerate(all_detections):
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for j, gt in enumerate(all_ground_truths):
            if gt['class_id'] == detection['class_id'] and not gt_matched[j]:
                iou = calculate_iou(detection['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    cumsum_fp = np.cumsum(fp)
    cumsum_tp = np.cumsum(tp)
    recalls = cumsum_tp / n_gts if n_gts > 0 else np.zeros_like(cumsum_tp)
    precisions = cumsum_tp / (cumsum_tp + cumsum_fp)
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap, float(precisions[-1]), float(recalls[-1])

def calculate_class_metrics(predictions: List[torch.Tensor], 
                          targets: List[torch.Tensor]) -> Dict:
    """Calculate per-class metrics"""
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred, target in zip(predictions, targets):
        pred_classes = pred[:, 5].cpu().numpy().astype(int)
        target_classes = target[:, 0].cpu().numpy().astype(int)
        
        # Count predictions
        for cls in pred_classes:
            class_stats[cls]['fp'] += 1
        
        # Count ground truths
        for cls in target_classes:
            class_stats[cls]['fn'] += 1
        
        # Match predictions with targets
        pred_matched = set()
        target_matched = set()
        
        for i, p in enumerate(pred):
            best_iou = 0.5  # IoU threshold
            best_target = -1
            
            for j, t in enumerate(target):
                if j in target_matched or int(p[5]) != int(t[0]):
                    continue
                
                iou = calculate_iou(p[:4].cpu().numpy(), t[1:5].cpu().numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_target = j
            
            if best_target >= 0:
                cls = int(p[5])
                class_stats[cls]['tp'] += 1
                class_stats[cls]['fp'] -= 1
                class_stats[cls]['fn'] -= 1
                pred_matched.add(i)
                target_matched.add(best_target)
    
    # Calculate metrics for each class
    class_metrics = {}
    for cls, stats in class_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }
    
    return {'class_metrics': class_metrics}

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two boxes"""
    # Convert to corners format if necessary
    if len(box1) == 4:
        box1_x1 = box1[0]
        box1_y1 = box1[1]
        box1_x2 = box1[2]
        box1_y2 = box1[3]
        
        box2_x1 = box2[0]
        box2_y1 = box2[1]
        box2_x2 = box2[2]
        box2_y2 = box2[3]
    else:
        # Handle center format
        box1_x1 = box1[0] - box1[2]/2
        box1_y1 = box1[1] - box1[3]/2
        box1_x2 = box1[0] + box1[2]/2
        box1_y2 = box1[1] + box1[3]/2
        
        box2_x1 = box2[0] - box2[2]/2
        box2_y1 = box2[1] - box2[3]/2
        box2_x2 = box2[0] + box2[2]/2
        box2_y2 = box2[1] + box2[3]/2
    
    # Calculate intersection coordinates
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    
    # Calculate areas
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    # Calculate IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou