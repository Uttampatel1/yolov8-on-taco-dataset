# src/model/evaluator.py
import numpy as np
from typing import Dict, List, Tuple
from ultralytics import YOLO
import torch
from ..utils.metrics import calculate_map
from ..visualization.plotter import plot_confusion_matrix, plot_pr_curve

class ModelEvaluator:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """Initialize model evaluator"""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.results = {}
        
    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate model on validation dataset"""
        results = {
            'precision': [],
            'recall': [],
            'map50': [],
            'map50_95': [],
            'confusion_matrix': None,
            'class_accuracy': {}
        }
        
        all_predictions = []
        all_targets = []
        
        # Run evaluation
        for batch in val_loader:
            images, targets = batch
            predictions = self.model(images)
            
            # Process predictions and targets
            for pred, target in zip(predictions, targets):
                pred = pred[pred[:, 4] > self.conf_threshold]
                all_predictions.append(pred)
                all_targets.append(target)
        
        # Calculate metrics
        metrics = calculate_map(all_predictions, all_targets)
        results.update(metrics)
        
        self.results = results
        return results
    
    def analyze_failures(self, val_loader: torch.utils.data.DataLoader, 
                        save_dir: str) -> List[Dict]:
        """Analyze failure cases"""
        failure_cases = []
        
        for batch in val_loader:
            images, targets = batch
            predictions = self.model(images)
            
            for img, pred, target in zip(images, predictions, targets):
                # Analyze each prediction
                if not self._is_correct_prediction(pred, target):
                    case = {
                        'image': img,
                        'prediction': pred,
                        'target': target,
                        'analysis': self._analyze_failure(pred, target)
                    }
                    failure_cases.append(case)
        
        return failure_cases
    
    def _is_correct_prediction(self, pred: torch.Tensor, 
                             target: torch.Tensor) -> bool:
        """Check if prediction is correct"""
        if pred is None or len(pred) == 0:
            return len(target) == 0
            
        # Calculate IoU and determine if prediction is correct
        ious = self._calculate_iou_matrix(pred, target)
        return (ious > 0.5).any()
    
    def _analyze_failure(self, pred: torch.Tensor, 
                        target: torch.Tensor) -> Dict:
        """Analyze why a prediction failed"""
        analysis = {
            'type': '',
            'confidence': 0.0,
            'iou': 0.0
        }
        
        if pred is None or len(pred) == 0:
            analysis['type'] = 'false_negative'
            return analysis
            
        if len(target) == 0:
            analysis['type'] = 'false_positive'
            analysis['confidence'] = float(pred[:, 4].max())
            return analysis
            
        # Calculate IoU
        ious = self._calculate_iou_matrix(pred, target)
        max_iou = float(ious.max())
        
        if max_iou < 0.5:
            analysis['type'] = 'low_iou'
            analysis['iou'] = max_iou
            analysis['confidence'] = float(pred[:, 4].max())
        else:
            analysis['type'] = 'misclassification'
            analysis['iou'] = max_iou
            analysis['confidence'] = float(pred[:, 4].max())
            
        return analysis
    
    def _calculate_iou_matrix(self, pred: torch.Tensor, 
                            target: torch.Tensor) -> torch.Tensor:
        """Calculate IoU matrix between predictions and targets"""
        # Extract boxes
        pred_boxes = pred[:, :4]
        target_boxes = target[:, :4]
        
        # Calculate IoU matrix
        return self._box_iou(pred_boxes, target_boxes)
    
    @staticmethod
    def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        
        return inter / union
    
    def generate_report(self, save_dir: str) -> str:
        """Generate evaluation report"""
        report = []
        report.append("Model Evaluation Report")
        report.append("=====================")
        report.append(f"\nMean Average Precision (mAP@0.5): {self.results['map50']:.4f}")
        report.append(f"Mean Average Precision (mAP@0.5-0.95): {self.results['map50_95']:.4f}")
        report.append(f"Average Precision: {self.results['precision']:.4f}")
        report.append(f"Average Recall: {self.results['recall']:.4f}")
        
        # Add class-wise results
        report.append("\nPer-class Results:")
        for class_id, metrics in self.results['class_accuracy'].items():
            report.append(f"\nClass {class_id}:")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1']:.4f}")
        
        # Save report
        report_path = f"{save_dir}/evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        # Generate plots
        plot_confusion_matrix(self.results['confusion_matrix'], save_dir)
        plot_pr_curve(self.results['precision'], self.results['recall'], save_dir)
        
        return report_path