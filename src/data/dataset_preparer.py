# src/data/dataset_preparer.py
import os
import shutil
import yaml
from pycocotools.coco import COCO
import splitfolders
from tqdm import tqdm
from ..utils.config import load_config

class TACODatasetPreparer:
    def __init__(self, config_path):
        """Initialize dataset preparer with configuration"""
        self.config = load_config(config_path)
        self.data_source = COCO(self.config['annotation_file'])
        self.setup_directories()
        self.setup_categories()
        
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [
            os.path.join(self.config['tmp_dir'], 'labels'),
            os.path.join(self.config['tmp_dir'], 'images'),
            self.config['final_dir']
        ]:
            os.makedirs(directory, exist_ok=True)
            
    def setup_categories(self):
        """Setup category mappings"""
        self.img_ids = self.data_source.getImgIds()
        self.catIds = self.data_source.getCatIds()
        self.categories = self.data_source.loadCats(self.catIds)
        self.categories.sort(key=lambda x: x['id'])
        
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        
        for c in self.categories:
            current_len = len(self.classes)
            self.coco_labels[current_len] = c['id']
            self.coco_labels_inverse[c['id']] = current_len
            self.classes[c['name']] = current_len
    
    def convert_annotations(self):
        """Convert COCO annotations to YOLO format"""
        class_counts = {label: 0 for label in self.config['label_transfer'].values()}
        processed_images = 0
        skipped_images = 0
        
        for img_id in tqdm(self.img_ids, desc='Converting annotations'):
            # Process annotations
            # [Original annotation conversion code here]
            pass
            
        return processed_images > 0
    
    def split_dataset(self):
        """Split dataset into train/val/test sets"""
        splitfolders.ratio(
            self.config['tmp_dir'],
            output=self.config['final_dir'],
            seed=1337,
            ratio=self.config['split_ratio']
        )
    
    def create_yaml(self):
        """Create YAML configuration file for YOLOv8"""
        yaml_content = {
            'path': os.path.abspath(self.config['final_dir']),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.config['label_transfer']),
            'names': {v: k for k, v in self.config['label_transfer'].items()}
        }
        
        yaml_path = os.path.join(self.config['final_dir'], 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        return yaml_path