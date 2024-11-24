import os
import shutil
import numpy as np
import tqdm
import yaml
import splitfolders
from pycocotools.coco import COCO
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime



class TACODatasetPreparerV2:
    def __init__(self, annotation_file, data_dir, output_base_dir='taco_dataset_v2'):
        """Initialize TACO dataset preparer"""
        self.data_source = COCO(annotation_file)
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.tmp_dir = os.path.join(output_base_dir, 'tmp')
        self.final_dir = os.path.join(output_base_dir, 'final')
        
        # Create necessary directories
        for directory in [
            os.path.join(self.tmp_dir, 'labels'),
            os.path.join(self.tmp_dir, 'images'),
            self.final_dir
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # Define label mapping (customize as needed)
        self.label_transfer = {5: 0, 12: 1}  # Mapping specific COCO IDs to new IDs
        self.setup_categories()
        
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
        class_counts = {label: 0 for label in self.label_transfer.values()}
        processed_images = 0
        skipped_images = 0
        
        for img_id in tqdm.tqdm(self.img_ids, desc='Converting annotations'):
            img_info = self.data_source.loadImgs(img_id)[0]
            save_name = img_info['file_name'].replace('/', '_')
            file_name = os.path.splitext(save_name)[0]
            
            height = img_info['height']
            width = img_info['width']
            label_path = os.path.join(self.tmp_dir, 'labels', f'{file_name}.txt')
            
            is_valid_image = False
            annotation_ids = self.data_source.getAnnIds(img_id)
            
            if not annotation_ids:
                skipped_images += 1
                continue
                
            annotations = self.data_source.loadAnns(annotation_ids)
            valid_annotations = []
            
            for ann in annotations:
                label = self.coco_labels_inverse[ann['category_id']]
                if label in self.label_transfer:
                    box = ann['bbox']
                    if box[2] >= 1 and box[3] >= 1:  # Valid box dimensions
                        is_valid_image = True
                        # Convert to YOLO format
                        x_center = (box[0] + box[2] / 2) / width
                        y_center = (box[1] + box[3] / 2) / height
                        box_width = box[2] / width
                        box_height = box[3] / height
                        
                        new_label = self.label_transfer[label]
                        class_counts[new_label] += 1
                        
                        valid_annotations.append(
                            f"{new_label} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                        )
            
            if is_valid_image and valid_annotations:
                # Save annotations
                with open(label_path, 'w') as f:
                    f.write('\n'.join(valid_annotations))
                
                # Copy image
                src_img_path = os.path.join(self.data_dir, img_info['file_name'])
                dst_img_path = os.path.join(self.tmp_dir, 'images', save_name)
                shutil.copy(src_img_path, dst_img_path)
                processed_images += 1
            else:
                if os.path.exists(label_path):
                    os.remove(label_path)
                skipped_images += 1
                
        print(f"\nProcessing Summary:")
        print(f"Processed Images: {processed_images}")
        print(f"Skipped Images: {skipped_images}")
        print("\nClass Distribution:")
        for label, count in class_counts.items():
            print(f"Class {label}: {count} annotations")
            
        return processed_images > 0
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split dataset into train/val/test sets"""
        print("\nSplitting dataset...")
        splitfolders.ratio(
            self.tmp_dir,
            output=self.final_dir,
            seed=1337,
            ratio=(train_ratio, val_ratio, test_ratio)
        )
    
    def create_yaml(self):
        """Create YAML configuration file for YOLOv8"""
        yaml_content = {
            'path': os.path.abspath(self.final_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.label_transfer),
            'names': {v: k for k, v in self.label_transfer.items()}
        }
        
        yaml_path = os.path.join(self.final_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        return yaml_path

def train_yolov8_model(yaml_path, img_size=320, batch_size=16, epochs=100, weights='yolov8s.pt'):
    """Train YOLOv8 model"""
    model = YOLO(weights)
    
    # Training
    results = model.train(
        data=yaml_path,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        name=f'taco_v2_{datetime.now().strftime("%Y%m%d_%H%M")}',
        plots=True  # Save training plots
    )
    
    return results

def plot_training_metrics(results, save_dir):
    """Plot and save training metrics"""
    metrics = results.results_dict
    
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train/box_loss'], label='train box loss')
    plt.plot(metrics['train/cls_loss'], label='train cls loss')
    plt.plot(metrics['val/box_loss'], label='val box loss')
    plt.plot(metrics['val/cls_loss'], label='val cls loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'losses.png'))
    plt.close()
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['metrics/precision(B)'], label='Precision')
    plt.plot(metrics['metrics/recall(B)'], label='Recall')
    plt.plot(metrics['metrics/mAP50(B)'], label='mAP50')
    plt.plot(metrics['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'metrics.png'))
    plt.close()

ANNOTATION_FILE = '/kaggle/input/tacotrashdataset/data/annotations.json'
DATA_DIR = '/kaggle/input/tacotrashdataset/data'
OUTPUT_DIR = '/kaggle/working/taco_dataset_v2'

# Initialize dataset preparer
preparer = TACODatasetPreparerV2(
    annotation_file=ANNOTATION_FILE,
    data_dir=DATA_DIR,
    output_base_dir=OUTPUT_DIR
)

# Convert annotations
if preparer.convert_annotations():
    # Split dataset
    preparer.split_dataset()
    
    # Create YAML configuration
    yaml_path = preparer.create_yaml()
    
    print("\nStarting model training...")
    # Train model
    results = train_yolov8_model(
        yaml_path=yaml_path,
        img_size=320,
        batch_size=16,
        epochs=100,
        weights='yolov8s.pt'  # Using small model
    )
    
    
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_validation_metrics(results, save_dir):
    """Plot validation metrics from YOLOv8 results"""
    metrics = results.results_dict
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot mAP metrics
    plt.figure(figsize=(10, 6))
    metric_names = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
    metric_values = [
        metrics['metrics/precision(B)'],
        metrics['metrics/recall(B)'],
        metrics['metrics/mAP50(B)'],
        metrics['metrics/mAP50-95(B)']
    ]
    
    plt.bar(metric_names, metric_values)
    plt.title('Validation Metrics')
    plt.ylabel('Value')
    
    # Add value labels on top of each bar
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(plots_dir, 'validation_metrics.png'))
    plt.close()

    # Save metrics summary
    with open(os.path.join(plots_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("Validation Metrics Summary:\n\n")
        f.write(f"Precision: {metrics['metrics/precision(B)']:.4f}\n")
        f.write(f"Recall: {metrics['metrics/recall(B)']:.4f}\n")
        f.write(f"mAP50: {metrics['metrics/mAP50(B)']:.4f}\n")
        f.write(f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}\n")
        f.write(f"Fitness: {results.fitness:.4f}\n")
        
        if hasattr(results, 'speed'):
            f.write("\nSpeed Metrics (ms):\n")
            for key, value in results.speed.items():
                f.write(f"{key}: {value*1000:.2f}\n")

def analyze_class_performance(results, save_dir):
    """Analyze per-class performance"""
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract per-class mAPs
    maps = results.maps
    names = results.names
    
    # Plot per-class mAP
    plt.figure(figsize=(10, 6))
    classes = list(names.values())
    plt.bar(classes, maps)
    plt.title('Per-class mAP50')
    plt.xlabel('Class')
    plt.ylabel('mAP50')
    
    # Add value labels on top of each bar
    for i, v in enumerate(maps):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(plots_dir, 'per_class_map.png'))
    plt.close()
    
    # Plot confusion matrix if available
    if hasattr(results, 'confusion_matrix'):
        plt.figure(figsize=(10, 8))
        matrix = results.confusion_matrix.matrix
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add class labels
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations with proper float formatting
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, f'{matrix[i, j]:.2f}',
                        ha="center", va="center")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        plt.close()
    
    # Save per-class metrics
    with open(os.path.join(plots_dir, 'class_metrics.txt'), 'w') as f:
        f.write("Per-class Performance:\n\n")
        for class_id, class_name in names.items():
            f.write(f"Class {class_id} ({class_name}):\n")
            f.write(f"mAP50: {maps[class_id]:.4f}\n\n")

def analyze_curves(results, save_dir):
    """Analyze and plot detection curves"""
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if hasattr(results, 'curves') and hasattr(results, 'curves_results'):
        try:
            for curve_name, curve_data in zip(results.curves, results.curves_results):
                plt.figure(figsize=(10, 6))
                x_data, y_data = curve_data[0], curve_data[1]
                
                # Handle both single and multi-class data
                if y_data.ndim > 1:
                    for i in range(y_data.shape[0]):
                        plt.plot(x_data, y_data[i], label=f'Class {i}')
                    plt.legend()
                else:
                    plt.plot(x_data, y_data)
                
                plt.title(f'{curve_name} Curve')
                plt.xlabel(curve_data[2])  # x-axis label
                plt.ylabel(curve_data[3])  # y-axis label
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, f'{curve_name.lower().replace("-", "_")}.png'))
                plt.close()
        except Exception as e:
            print(f"Warning: Error plotting curves: {str(e)}")

def save_validation_results(results, save_dir):
    """Save complete validation results"""
    try:
        # Plot basic metrics
        plot_validation_metrics(results, save_dir)
        
        # Analyze class performance
        analyze_class_performance(results, save_dir)
        
        # Analyze curves
        analyze_curves(results, save_dir)
        
        # Save raw results
        results_file = os.path.join(save_dir, 'plots', 'validation_results.txt')
        with open(results_file, 'w') as f:
            f.write("Complete Validation Results:\n\n")
            
            # Save main metrics
            f.write("Main Metrics:\n")
            for key, value in results.results_dict.items():
                f.write(f"{key}: {value}\n")
            
            # Save speed metrics if available
            if hasattr(results, 'speed'):
                f.write("\nSpeed Metrics (ms):\n")
                for key, value in results.speed.items():
                    f.write(f"{key}: {value*1000:.2f}\n")
            
            # Save model task
            if hasattr(results, 'task'):
                f.write(f"\nTask: {results.task}\n")
            
            # Save class names
            f.write("\nClass Mapping:\n")
            for class_id, class_name in results.names.items():
                f.write(f"Class {class_id}: {class_name}\n")
                
        print(f"Results saved to: {save_dir}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        OUTPUT_DIR = "/kaggle/working/runs"
        save_validation_results(results, OUTPUT_DIR)
    except Exception as e:
        print(f"Error in main: {str(e)}")
        
        
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detection(image_path, model_path, conf_thresh=0.25):
    """
    Visualize object detection results for a single image
    """
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_thresh,
        save=False,
        show=False
    )[0]
    
    # Load image for visualization
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original image with bounding boxes
    ax1.imshow(image)
    ax1.set_title("Detections")
    
    # Plot segmentation masks
    ax2.imshow(image)
    ax2.set_title("Segmentation Masks")
    
    # Colors for different classes
    num_classes = len(model.names)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    
    # Get detection results
    boxes = results.boxes
    masks = results.masks
    
    if boxes is not None:
        for i in range(len(boxes)):
            # Get box coordinates and information
            box = boxes[i].xyxy[0].cpu().numpy()
            conf = float(boxes[i].conf[0].cpu().numpy())  # Convert to float
            cls = int(boxes[i].cls[0].cpu().numpy())      # Convert to int
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=colors[cls],
                facecolor='none'
            )
            
            # Add rectangle to plot
            ax1.add_patch(rect)
            
            # Add label
            label = f'{model.names[cls]}: {conf:.2f}'
            ax1.text(
                box[0], box[1] - 5,
                label,
                color=colors[cls],
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
    
    # Plot masks if available
    if masks is not None:
        masks = masks.cpu().numpy()
        for i in range(len(masks)):
            cls = int(boxes[i].cls[0].cpu().numpy())
            mask = masks[i]
            
            # Create mask overlay
            mask_color = np.array(colors[cls][:3])
            mask_overlay = np.zeros_like(image, dtype=np.float32)
            
            for c in range(3):
                mask_overlay[:, :, c] = mask_color[c]
            
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mask_overlay = mask_overlay * mask * 0.5
            
            # Add mask to image
            ax2.imshow(mask_overlay, alpha=0.5)
    
    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Add detection summary
    if boxes is not None:
        summary = []
        class_counts = {}
        
        for box in boxes:
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            if cls not in class_counts:
                class_counts[cls] = {'count': 0, 'avg_conf': 0}
            
            class_counts[cls]['count'] += 1
            class_counts[cls]['avg_conf'] += conf
        
        for cls in class_counts:
            avg_conf = class_counts[cls]['avg_conf'] / class_counts[cls]['count']
            summary.append(f"{model.names[cls]}: {class_counts[cls]['count']} instances (avg conf: {avg_conf:.2f})")
        
        plt.figtext(0.02, 0.02, '\n'.join(summary), fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def print_detection_details(image_path, model_path, conf_thresh=0.25):
    """
    Print detailed detection information for a single image
    """
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_thresh,
        save=False,
        show=False
    )[0]
    
    print(f"\nDetection Results for: {image_path}")
    print("-" * 50)
    
    if results.boxes is not None:
        for i in range(len(results.boxes)):
            box = results.boxes[i]
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            bbox = box.xyxy[0].cpu().numpy()
            
            print(f"\nDetection {i+1}:")
            print(f"Class: {model.names[cls]}")
            print(f"Confidence: {conf:.3f}")
            print(f"Bounding Box: [x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}]")
            
            if results.masks is not None:
                mask = results.masks[i]
                mask_area = mask.cpu().numpy().sum()
                print(f"Mask Area: {mask_area:.1f} pixels")
    
    else:
        print("No detections found.")

def run_inference_on_image(image_path, model_path, conf_thresh=0.25, save_path=None):
    """
    Run inference and save/display results
    """
    # Create visualization
    fig = visualize_detection(image_path, model_path, conf_thresh)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Display
    plt.show()
    
    # Print details
    print_detection_details(image_path, model_path, conf_thresh)



# Example usage
if __name__ == "__main__":
    IMAGE_PATH = "/kaggle/input/tacotrashdataset/data/batch_1/000000.jpg"
    MODEL_PATH = "/kaggle/working/runs/detect/taco_v2_20241124_1206/weights/best.pt"
    SAVE_PATH = "detection_result.png"  # Optional
    
    run_inference_on_image(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        conf_thresh=0.25,
        save_path=SAVE_PATH
    )