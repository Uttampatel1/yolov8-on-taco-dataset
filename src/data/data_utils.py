# src/data/data_utils.py
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

class DataUtils:
    @staticmethod
    def verify_image(image_path: str) -> Tuple[bool, str]:
        """Verify if an image file is valid and readable"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Image could not be read"
            if img.size == 0:
                return False, "Image is empty"
            return True, "Image is valid"
        except Exception as e:
            return False, f"Error reading image: {str(e)}"

    @staticmethod
    def check_annotations(annotation_path: str, image_dir: str) -> Tuple[bool, List[str]]:
        """Verify annotation file and corresponding images"""
        missing_images = []
        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    image_name = line.split()[0]
                    image_path = os.path.join(image_dir, image_name)
                    if not os.path.exists(image_path):
                        missing_images.append(image_name)
            return len(missing_images) == 0, missing_images
        except Exception as e:
            return False, [f"Error reading annotations: {str(e)}"]

    @staticmethod
    def bbox_coco_to_yolo(box: List[float], img_w: int, img_h: int) -> List[float]:
        """Convert COCO bbox format to YOLO format"""
        x, y, w, h = box
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        w = w / img_w
        h = h / img_h
        return [x_center, y_center, w, h]

    @staticmethod
    def bbox_yolo_to_coco(box: List[float], img_w: int, img_h: int) -> List[float]:
        """Convert YOLO bbox format to COCO format"""
        x_center, y_center, w, h = box
        w_pixels = w * img_w
        h_pixels = h * img_h
        x = (x_center * img_w) - (w_pixels / 2)
        y = (y_center * img_h) - (h_pixels / 2)
        return [x, y, w_pixels, h_pixels]

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes in YOLO format"""
        # Convert to corners format
        box1_x1 = box1[0] - box1[2]/2
        box1_y1 = box1[1] - box1[3]/2
        box1_x2 = box1[0] + box1[2]/2
        box1_y2 = box1[1] + box1[3]/2
        
        box2_x1 = box2[0] - box2[2]/2
        box2_y1 = box2[1] - box2[3]/2
        box2_x2 = box2[0] + box2[2]/2
        box2_y2 = box2[1] + box2[3]/2
        
        # Intersection coordinates
        xi1 = max(box1_x1, box2_x1)
        yi1 = max(box1_y1, box2_y1)
        xi2 = min(box1_x2, box2_x2)
        yi2 = min(box1_y2, box2_y2)
        
        # Intersection area
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union area
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    @staticmethod
    def calculate_dataset_stats(image_dir: str) -> Dict[str, Union[int, float, List[Tuple[int, int]]]]:
        """Calculate dataset statistics"""
        stats = {
            'total_images': 0,
            'total_size_mb': 0,
            'avg_width': 0,
            'avg_height': 0,
            'resolutions': []
        }
        
        total_width = 0
        total_height = 0
        
        for img_path in Path(image_dir).glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                stats['total_images'] += 1
                stats['total_size_mb'] += os.path.getsize(img_path) / (1024 * 1024)
                
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    total_height += h
                    total_width += w
                    stats['resolutions'].append((w, h))
        
        if stats['total_images'] > 0:
            stats['avg_width'] = total_width / stats['total_images']
            stats['avg_height'] = total_height / stats['total_images']
        
        return stats

    @staticmethod
    def augment_image(image: np.ndarray, augmentation_type: str) -> np.ndarray:
        """Apply basic augmentations to an image"""
        if augmentation_type == 'flip':
            return cv2.flip(image, 1)
        elif augmentation_type == 'brightness':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = hsv[:,:,2] * np.random.uniform(0.8, 1.2)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif augmentation_type == 'rotation':
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            return cv2.warpAffine(image, matrix, (w, h))
        return image