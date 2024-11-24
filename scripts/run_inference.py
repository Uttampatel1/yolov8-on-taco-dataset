# scripts/run_inference.py

import argparse
import os
import yaml
import logging
from pathlib import Path
import cv2
import torch

from src.visualization.detector_visualizer import DetectorVisualizer
from src.utils.config import load_config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with TACO Waste Detection Model')
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--weights', type=str, required=True,
                      help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                      help='Source path (image/video/directory)')
    parser.add_argument('--save-dir', type=str, default='inference',
                      help='Directory to save results')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                      help='Confidence threshold')
    parser.add_argument('--device', type=str, default='',
                      help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view', action='store_true',
                      help='Display results')
    
    return parser.parse_args()

def setup_inference(args):
    """Setup inference environment"""
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.save_dir, 'inference.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Save inference configuration
    infer_config = vars(args)
    with open(os.path.join(args.save_dir, 'inference_config.yaml'), 'w') as f:
        yaml.dump(infer_config, f)

def process_source(source: str) -> list:
    """Process source path and return list of files to process"""
    files = []
    source = Path(source)
    
    if source.is_file():
        files = [source]
    elif source.is_dir():
        for ext in ['.jpg', '.jpeg', '.png', '.mp4', '.avi']:
            files.extend(source.glob(f'*{ext}'))
    else:
        raise FileNotFoundError(f"Source not found: {source}")
    
    return sorted(files)

def run_inference(args, config):
    """Run inference on files"""
    logging.info("Initializing inference...")
    
    # Create visualizer instance
    visualizer = DetectorVisualizer(args.weights, args.conf_thres, args.device)
    
    # Get files to process
    files = process_source(args.source)
    logging.info(f"Found {len(files)} files to process")
    
    results = []
    for file in files:
        try:
            logging.info(f"Processing: {file}")
            
            # Run detection
            result = visualizer.detect(str(file))
            
            # Save results
            output_path = os.path.join(
                args.save_dir,
                file.stem + '_detection' + file.suffix
            )
            
            visualizer.save_visualization(result, output_path)
            
            if args.view:
                visualizer.display_result(result)
            
            results.append({
                'file': str(file),
                'detections': len(result.boxes),
                'output': output_path
            })
            
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
    
    # Save summary
    summary_path = os.path.join(args.save_dir, 'inference_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(results, f)
    
    return results

def main():
    """Main inference pipeline"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup inference
    setup_inference(args)
    
    # Run inference
    try:
        results = run_inference(args, config)
        logging.info(f"Inference completed. Results saved in {args.save_dir}")
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")

if __name__ == "__main__":
    main()