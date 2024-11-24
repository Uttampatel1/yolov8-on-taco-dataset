# scripts/evaluate_model.py

import argparse
import os
import yaml
import logging
from pathlib import Path

from src.model.evaluator import ModelEvaluator
from src.visualization.plotter import Plotter
from src.utils.config import load_config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate TACO Waste Detection Model')
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--weights', type=str, required=True,
                      help='Path to model weights')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to data yaml file')
    parser.add_argument('--save-dir', type=str, default='evaluation',
                      help='Directory to save results')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                      help='IoU threshold for evaluation')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                      help='Confidence threshold for evaluation')
    
    return parser.parse_args()

def setup_evaluation(args):
    """Setup evaluation environment"""
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.save_dir, 'evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Save evaluation configuration
    eval_config = vars(args)
    with open(os.path.join(args.save_dir, 'eval_config.yaml'), 'w') as f:
        yaml.dump(eval_config, f)

def evaluate_model(args, config):
    """Run model evaluation"""
    logging.info("Initializing model evaluation...")
    
    # Create evaluator instance
    evaluator = ModelEvaluator(
        model_path=args.weights,
        conf_threshold=args.conf_thres
    )
    
    # Create plotter instance
    plotter = Plotter(args.save_dir)
    
    try:
        # Run evaluation
        results = evaluator.evaluate(
            data_path=args.data,
            iou_threshold=args.iou_thres
        )
        
        # Analyze results
        logging.info("Analyzing evaluation results...")
        
        # Plot results
        plotter.plot_pr_curve(
            results['precision'],
            results['recall'],
            config.get('class_names', None)
        )
        
        if 'confusion_matrix' in results:
            plotter.plot_confusion_matrix(
                results['confusion_matrix'],
                config.get('class_names', None)
            )
        
        plotter.plot_metrics_comparison(
            results['class_metrics'],
            config.get('class_names', None)
        )
        
        # Generate report
        report_path = evaluator.generate_report(args.save_dir)
        logging.info(f"Evaluation report saved to: {report_path}")
        
        return results
    
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        return None

def analyze_failures(evaluator, args, results):
    """Analyze failure cases"""
    logging.info("Analyzing failure cases...")
    
    failure_dir = os.path.join(args.save_dir, 'failure_analysis')
    os.makedirs(failure_dir, exist_ok=True)
    
    try:
        # Get failure cases
        failures = evaluator.analyze_failures(
            val_loader=None,  # Need to implement data loading
            save_dir=failure_dir
        )
        
        # Save failure analysis
        with open(os.path.join(failure_dir, 'failure_analysis.yaml'), 'w') as f:
            yaml.dump(failures, f)
        
        logging.info(f"Failure analysis saved to: {failure_dir}")
        
        return failures
    
    except Exception as e:
        logging.error(f"Failure analysis failed: {str(e)}")
        return None

def main():
    """Main evaluation pipeline"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup evaluation
    setup_evaluation(args)
    
    # Run evaluation
    results = evaluate_model(args, config)
    if results is None:
        return
    
    # Analyze failures
    failures = analyze_failures(evaluator, args, results)
    
    logging.info(f"Evaluation completed. Results saved in {args.save_dir}")

if __name__ == "__main__":
    main()