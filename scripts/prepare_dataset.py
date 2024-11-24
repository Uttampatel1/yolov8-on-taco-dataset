# scripts/prepare_dataset.py

import argparse
import os
import yaml
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import splitfolders
from pycocotools.coco import COCO

class DatasetPreparer:
    def __init__(self, config_path: str):
        """Initialize dataset preparer with configuration"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.coco = None
        self.categories = {}
        self.stats = {
            'processed_images': 0,
            'skipped_images': 0,
            'annotations_converted': 0,
            'corrupt_images': 0
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required_fields = [
            'annotation_file',
            'data_dir',
            'output_dir',
            'label_transfer',
            'split_ratio',
            'img_size'
        ]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")

        return config

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(self.config['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'dataset_preparation.log')),
                logging.StreamHandler()
            ]
        )

    def setup_directories(self):
        """Create necessary directories"""
        dirs = {
            'tmp': os.path.join(self.config['output_dir'], 'tmp'),
            'tmp_images': os.path.join(self.config['output_dir'], 'tmp/images'),
            'tmp_labels': os.path.join(self.config['output_dir'], 'tmp/labels'),
            'final': os.path.join(self.config['output_dir'], 'final'),
            'corrupted': os.path.join(self.config['output_dir'], 'corrupted')
        }

        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        return dirs

    def load_annotations(self):
        """Load and verify COCO annotations"""
        try:
            logging.info(f"Loading annotations from {self.config['annotation_file']}")
            self.coco = COCO(self.config['annotation_file'])
            
            # Setup categories
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            self.categories = {cat['id']: idx for idx, cat in enumerate(cats)}
            
            logging.info(f"Found {len(self.categories)} categories")
            return True
        except Exception as e:
            logging.error(f"Error loading annotations: {str(e)}")
            return False

    def verify_image(self, image_path: str) -> bool:
        """Verify if image is valid"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            if img.size == 0:
                return False
            if len(img.shape) != 3:
                return False
            return True
        except Exception:
            return False

    def convert_bbox(self, box: List[float], img_w: int, img_h: int) -> List[float]:
        """Convert COCO bbox format to YOLO format"""
        x, y, w, h = box
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        w = w / img_w
        h = h / img_h
        return [x_center, y_center, w, h]

    def process_images(self, dirs: Dict[str, str]):
        """Process and convert dataset images and annotations"""
        logging.info("Starting dataset processing...")
        
        img_ids = self.coco.getImgIds()
        for img_id in tqdm(img_ids, desc="Processing images"):
            try:
                # Load image info
                img_info = self.coco.loadImgs(img_id)[0]
                src_path = os.path.join(self.config['data_dir'], img_info['file_name'])
                
                # Check if image exists and is valid
                if not os.path.exists(src_path) or not self.verify_image(src_path):
                    self.stats['corrupt_images'] += 1
                    logging.warning(f"Corrupt or missing image: {src_path}")
                    continue

                # Get annotations
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                if not ann_ids:
                    self.stats['skipped_images'] += 1
                    continue

                annotations = self.coco.loadAnns(ann_ids)
                valid_annotations = []

                # Process annotations
                for ann in annotations:
                    cat_id = ann['category_id']
                    if cat_id in self.config['label_transfer']:
                        new_label = self.config['label_transfer'][cat_id]
                        bbox = self.convert_bbox(
                            ann['bbox'], 
                            img_info['width'], 
                            img_info['height']
                        )
                        valid_annotations.append(f"{new_label} {' '.join(map(str, bbox))}")
                        self.stats['annotations_converted'] += 1

                if valid_annotations:
                    # Save label file
                    base_name = os.path.splitext(img_info['file_name'])[0]
                    label_path = os.path.join(dirs['tmp_labels'], f"{base_name}.txt")
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(valid_annotations))

                    # Copy and resize image
                    dst_path = os.path.join(dirs['tmp_images'], os.path.basename(img_info['file_name']))
                    img = cv2.imread(src_path)
                    if self.config.get('img_size'):
                        img = cv2.resize(img, (self.config['img_size'], self.config['img_size']))
                    cv2.imwrite(dst_path, img)
                    
                    self.stats['processed_images'] += 1

            except Exception as e:
                logging.error(f"Error processing image {img_id}: {str(e)}")
                self.stats['skipped_images'] += 1

    def split_dataset(self, dirs: Dict[str, str]):
        """Split dataset into train/val/test sets"""
        logging.info("Splitting dataset...")
        try:
            splitfolders.ratio(
                dirs['tmp'],
                output=dirs['final'],
                seed=42,
                ratio=self.config['split_ratio'],
                group_prefix=None
            )
            return True
        except Exception as e:
            logging.error(f"Error splitting dataset: {str(e)}")
            return False

    def create_data_yaml(self, dirs: Dict[str, str]):
        """Create data configuration YAML file"""
        yaml_content = {
            'path': os.path.abspath(dirs['final']),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(set(self.config['label_transfer'].values())),
            'names': {v: k for k, v in self.config['label_transfer'].items()}
        }

        yaml_path = os.path.join(dirs['final'], 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        return yaml_path

    def print_statistics(self):
        """Print dataset preparation statistics"""
        logging.info("\nDataset Preparation Statistics:")
        logging.info(f"Processed Images: {self.stats['processed_images']}")
        logging.info(f"Skipped Images: {self.stats['skipped_images']}")
        logging.info(f"Converted Annotations: {self.stats['annotations_converted']}")
        logging.info(f"Corrupt Images: {self.stats['corrupt_images']}")

    def prepare(self) -> bool:
        """Run complete dataset preparation pipeline"""
        try:
            # Setup directories
            dirs = self.setup_directories()
            
            # Load annotations
            if not self.load_annotations():
                return False
            
            # Process images and annotations
            self.process_images(dirs)
            
            # Split dataset
            if not self.split_dataset(dirs):
                return False
            
            # Create data.yaml
            yaml_path = self.create_data_yaml(dirs)
            logging.info(f"Created dataset configuration at: {yaml_path}")
            
            # Print statistics
            self.print_statistics()
            
            # Cleanup temporary files
            if not self.config.get('keep_tmp', False):
                shutil.rmtree(dirs['tmp'])
            
            return True
            
        except Exception as e:
            logging.error(f"Dataset preparation failed: {str(e)}")
            return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare TACO dataset for training')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main execution"""
    args = parse_args()
    
    try:
        preparer = DatasetPreparer(args.config)
        if preparer.prepare():
            logging.info("Dataset preparation completed successfully!")
        else:
            logging.error("Dataset preparation failed!")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()