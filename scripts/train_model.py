# scripts/train_model.py
import argparse
from src.data.dataset_preparer import TACODatasetPreparer
from src.model.trainer import ModelTrainer
from src.visualization.detector_visualizer import DetectorVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train TACO detector model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data-only', action='store_true', help='Only prepare dataset')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Prepare dataset
    print("Preparing dataset...")
    preparer = TACODatasetPreparer(args.config)
    if preparer.convert_annotations():
        preparer.split_dataset()
        yaml_path = preparer.create_yaml()
        print(f"Dataset prepared. YAML config saved at: {yaml_path}")
    
    if not args.data_only:
        # Train model
        print("\nStarting model training...")
        trainer = ModelTrainer(args.config)
        results = trainer.train(yaml_path)
        
        # Validate model
        print("\nValidating model...")
        val_results = trainer.validate()
        
        if args.visualize:
            # Visualize results
            print("\nVisualizing results...")
            visualizer = DetectorVisualizer(args.config)
            test_image = "path/to/test/image.jpg"  # Update with actual test image path
            fig, results = visualizer.visualize_detection(test_image)
            visualizer.save_visualization(fig, "detection_results.png")
            visualizer.print_detection_details(results)

if __name__ == "__main__":
    main()
    
    
# scripts/train_model.py

# import argparse
# import os
# import yaml
# from datetime import datetime
# from pathlib import Path
# import logging

# from src.data.dataset_preparer import TACODatasetPreparer
# from src.model.trainer import ModelTrainer
# from src.utils.config import load_config

# def setup_logging(save_dir: str):
#     """Setup logging configuration"""
#     log_file = os.path.join(save_dir, 'training.log')
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )

# def parse_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Train TACO Waste Detection Model')
    
#     parser.add_argument('--config', type=str, required=True,
#                       help='Path to configuration file')
#     parser.add_argument('--data-only', action='store_true',
#                       help='Only prepare dataset without training')
#     parser.add_argument('--resume', type=str,
#                       help='Resume training from checkpoint')
#     parser.add_argument('--eval', action='store_true',
#                       help='Evaluate model after training')
    
#     return parser.parse_args()

# def prepare_dataset(config):
#     """Prepare and validate dataset"""
#     logging.info("Preparing dataset...")
    
#     preparer = TACODatasetPreparer(config)
    
#     # Verify dataset structure
#     if not preparer.verify_dataset():
#         logging.error("Dataset verification failed!")
#         return None
    
#     # Convert annotations
#     if not preparer.convert_annotations():
#         logging.error("Annotation conversion failed!")
#         return None
    
#     # Split dataset
#     preparer.split_dataset()
    
#     # Create YOLO config
#     yaml_path = preparer.create_yaml()
#     logging.info(f"Dataset preparation completed. YAML config saved at: {yaml_path}")
    
#     return yaml_path

# def train_model(config, yaml_path, resume_path=None):
#     """Train the model"""
#     logging.info("Initializing model training...")
    
#     # Create trainer instance
#     trainer = ModelTrainer(config)
    
#     # Setup training parameters
#     train_params = {
#         'yaml_path': yaml_path,
#         'epochs': config['training']['epochs'],
#         'batch_size': config['training']['batch_size'],
#         'img_size': config['training']['img_size'],
#         'save_dir': config['paths']['save_dir'],
#         'resume': resume_path
#     }
    
#     # Start training
#     try:
#         results = trainer.train(**train_params)
#         logging.info("Training completed successfully!")
#         return results
#     except Exception as e:
#         logging.error(f"Training failed: {str(e)}")
#         return None

# def evaluate_model(config, trainer, results):
#     """Evaluate trained model"""
#     logging.info("Starting model evaluation...")
    
#     try:
#         eval_results = trainer.validate()
        
#         # Save evaluation results
#         eval_path = os.path.join(config['paths']['save_dir'], 'evaluation')
#         os.makedirs(eval_path, exist_ok=True)
        
#         with open(os.path.join(eval_path, 'metrics.yaml'), 'w') as f:
#             yaml.dump(eval_results, f)
        
#         logging.info(f"Evaluation results saved to {eval_path}")
#         return eval_results
    
#     except Exception as e:
#         logging.error(f"Evaluation failed: {str(e)}")
#         return None

# def main():
#     """Main training pipeline"""
#     args = parse_args()
    
#     # Load configuration
#     config = load_config(args.config)
    
#     # Setup save directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_dir = os.path.join(config['paths']['save_dir'], f'train_{timestamp}')
#     os.makedirs(save_dir, exist_ok=True)
#     config['paths']['save_dir'] = save_dir
    
#     # Setup logging
#     setup_logging(save_dir)
    
#     # Save configuration
#     with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
#         yaml.dump(config, f)
    
#     # Prepare dataset
#     yaml_path = prepare_dataset(config)
#     if yaml_path is None:
#         return
    
#     if args.data_only:
#         logging.info("Dataset preparation completed. Exiting as --data-only flag was set.")
#         return
    
#     # Train model
#     results = train_model(config, yaml_path, args.resume)
#     if results is None:
#         return
    
#     # Evaluate if requested
#     if args.eval:
#         eval_results = evaluate_model(config, trainer, results)
#         if eval_results is None:
#             return
    
#     logging.info(f"All operations completed. Results saved in {save_dir}")

# if __name__ == "__main__":
#     main()