# src/visualization/detector_visualizer.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from ..utils.config import load_config

class DetectorVisualizer:
    def __init__(self, config_path):
        """Initialize detector visualizer with configuration"""
        self.config = load_config(config_path)
        self.model = YOLO(self.config['model_path'])
        
    def visualize_detection(self, image_path):
        """Visualize object detection results for a single image"""
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.config['conf_thresh'],
            save=False,
            show=False
        )[0]
        
        # Create visualization
        fig = self._create_visualization(image_path, results)
        return fig, results
    
    def _create_visualization(self, image_path, results):
        """Create visualization figure"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # [Visualization code here]
        # Similar to the original visualization code but organized as a class method
        
        return fig
    
    def print_detection_details(self, results):
        """Print detailed detection information"""
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                # [Detection details printing code here]
                pass
        else:
            print("No detections found.")
            
    def save_visualization(self, fig, save_path):
        """Save visualization figure"""
        fig.savefig(save_path, bbox_inches='tight', dpi=300)