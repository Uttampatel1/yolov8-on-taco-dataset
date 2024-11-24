# src/model/trainer.py
from ultralytics import YOLO
from datetime import datetime
from ..utils.config import load_config
from ..visualization.plotter import plot_training_metrics

class ModelTrainer:
    def __init__(self, config_path):
        """Initialize model trainer with configuration"""
        self.config = load_config(config_path)
        self.model = YOLO(self.config['weights'])
        
    def train(self, yaml_path):
        """Train YOLOv8 model"""
        results = self.model.train(
            data=yaml_path,
            imgsz=self.config['img_size'],
            epochs=self.config['epochs'],
            batch=self.config['batch_size'],
            name=f'taco_v2_{datetime.now().strftime("%Y%m%d_%H%M")}',
            plots=True
        )
        
        # Plot training metrics
        if self.config.get('plot_metrics', True):
            plot_training_metrics(results, self.config['output_dir'])
        
        return results
    
    def validate(self, data_path=None):
        """Validate trained model"""
        results = self.model.val(data=data_path or self.config['val_data'])
        return results