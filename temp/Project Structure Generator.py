import os

def create_project_structure(base_dir='taco_detector'):
    """Create the project directory structure"""
    
    # Define the project structure
    structure = {
        'src': {
            'data': {
                'dataset_preparer.py': '',
                'data_utils.py': ''
            },
            'model': {
                'trainer.py': '',
                'evaluator.py': ''
            },
            'visualization': {
                'plotter.py': '',
                'detector_visualizer.py': ''
            },
            'utils': {
                'config.py': '',
                'metrics.py': ''
            },
            '__init__.py': ''
        },
        'config': {
            'dataset_config.yaml': '''
path: /path/to/dataset
train: train/images
val: val/images
test: test/images
nc: 2  # number of classes
names: {0: 'class1', 1: 'class2'}
''',
            'model_config.yaml': '''
model_name: yolov8s
img_size: 320
batch_size: 16
epochs: 100
conf_thresh: 0.25
'''
        },
        'scripts': {
            'prepare_dataset.py': '',
            'train_model.py': '',
            'evaluate_model.py': '',
            'run_inference.py': ''
        },
        'tests': {
            '__init__.py': '',
            'test_dataset_preparer.py': '',
            'test_trainer.py': '',
            'test_evaluator.py': ''
        },
        'notebooks': {
            'data_exploration.ipynb': '',
            'model_training.ipynb': '',
            'results_analysis.ipynb': ''
        },
        'requirements.txt': '''
ultralytics>=8.0.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
PyYAML>=5.4.0
pycocotools>=2.0.0
split-folders>=0.5.0
tqdm>=4.62.0
'''
    }
    
    def create_structure(current_path, structure):
        for name, content in structure.items():
            path = os.path.join(current_path, name)
            
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                create_structure(path, content)
            else:
                if isinstance(content, str):
                    with open(path, 'w') as f:
                        f.write(content.strip() if content else '')
                else:
                    print(f"Warning: Unexpected content type for {path}: {type(content)}")
                    open(path, 'a').close()
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    create_structure(base_dir, structure)
    
    print(f"Project structure created at: {base_dir}")
    return base_dir

if __name__ == "__main__":
    # Create the project structure
    project_dir = create_project_structure()
    
    print("\nProject structure created successfully!")
    print(f"Project directory: {project_dir}")
    
    # Print structure tree
    def print_tree(startpath):
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = '  ' * (level + 1)
            for f in files:
                print(f"{subindent}{f}")
    
    print("\nProject Structure Tree:")
    print_tree(project_dir)