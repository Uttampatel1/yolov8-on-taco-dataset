# src/utils/config.py
import yaml
import os

def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = [
        'data_dir',
        'output_dir',
        'label_transfer',
        'split_ratio',
        'model_config'
    ]
    
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {missing_fields}")
    
    return config

def save_config(config, config_path):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)