# src/visualization/plotter.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import os

class Plotter:
    def __init__(self, save_dir: str):
        """Initialize plotter with save directory"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training metrics history"""
        metrics = ['loss', 'mAP_0.5', 'mAP_0.5:0.95', 'precision', 'recall']
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            if metric in history:
                plt.subplot(3, 2, i)
                plt.plot(history[metric], label='train')
                if f'val_{metric}' in history:
                    plt.plot(history[f'val_{metric}'], label='val')
                plt.title(f'{metric} over epochs')
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()
    
    def plot_pr_curve(self, precisions: List[float], recalls: List[float], 
                     class_names: List[str] = None):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(10, 8))
        
        if isinstance(precisions, list) and isinstance(recalls, list):
            plt.plot(recalls, precisions, label='PR curve')
        else:
            for i in range(len(precisions)):
                label = f'Class {i}' if class_names is None else class_names[i]
                plt.plot(recalls[i], precisions[i], label=label)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.save_dir, 'pr_curve.png'))
        plt.close()
    
    def plot_confusion_matrix(self, matrix: np.ndarray, class_names: List[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        if class_names is None:
            class_names = [str(i) for i in range(len(matrix))]
        
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()
    
    def plot_bounding_boxes(self, image: np.ndarray, boxes: List[Dict], 
                          class_names: List[str] = None):
        """Plot bounding boxes on image"""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names) if class_names else 10))
        
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            cls = box['class_id']
            conf = box.get('confidence', None)
            
            color = colors[cls % len(colors)]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color=color, linewidth=2)
            plt.gca().add_patch(rect)
            
            label = f"{class_names[cls] if class_names else cls}"
            if conf is not None:
                label += f" {conf:.2f}"
            
            plt.text(x1, y1-5, label, color=color, 
                    backgroundcolor='white')
        
        plt.axis('off')
        plt.savefig(os.path.join(self.save_dir, 'detections.png'))
        plt.close()
    
    def plot_class_distribution(self, class_counts: Dict[int, int], 
                              class_names: List[str] = None):
        """Plot class distribution"""
        plt.figure(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        if class_names:
            labels = [class_names[cls] for cls in classes]
        else:
            labels = [f"Class {cls}" for cls in classes]
        
        plt.bar(labels, counts)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'class_distribution.png'))
        plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict[str, Dict[str, float]], 
                              class_names: List[str] = None):
        """Plot comparison of different metrics across classes"""
        metrics_to_plot = ['precision', 'recall', 'f1']
        n_classes = len(metrics)
        
        plt.figure(figsize=(15, 5))
        x = np.arange(n_classes)
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics[cls][metric] for cls in sorted(metrics.keys())]
            plt.bar(x + i*width, values, width, label=metric.capitalize())
        
        if class_names:
            plt.xticks(x + width, [class_names[i] for i in range(n_classes)], 
                      rotation=45)
        else:
            plt.xticks(x + width, [f'Class {i}' for i in range(n_classes)], 
                      rotation=45)
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Metrics Comparison Across Classes')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_comparison.png'))
        plt.close()