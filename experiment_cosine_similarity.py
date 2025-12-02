import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
import json
from scipy.spatial.distance import cosine
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STRANGER_CLASS = 'stranger'


def extract_embeddings(images, device=DEVICE, class_name=None):
    """Extract embeddings using the same encoder as train_v2.py"""
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    embeddings = []
    valid_indices = []
    
    desc = f"Extracting embeddings" + (f" ({class_name})" if class_name else "")
    for idx, img in enumerate(tqdm(images, desc=desc, leave=False)):
        try:
            # Convert BGR to RGB for facenet
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img_rgb)
            if face is None:
                tqdm.write(f"⚠️ No face detected for image {idx}")
                continue
            
            face = face.unsqueeze(0).to(device)
            emb = resnet(face).detach().cpu().numpy()[0]
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            tqdm.write(f"❌ Error processing image {idx}: {e}")
    
    return np.array(embeddings), valid_indices


def load_images_from_directory(data_dir='data'):
    """Load all images from data directory, organized by class"""
    images_dict = {}
    image_paths_dict = {}
    
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        # Check if there's a nested directory with the same name
        nested_path = os.path.join(class_path, class_dir)
        if os.path.isdir(nested_path):
            image_dir = nested_path
        else:
            image_dir = class_path
        
        images = []
        image_paths = []
        for file in sorted(os.listdir(image_dir)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(image_dir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    image_paths.append(img_path)
        
        if len(images) > 0:
            images_dict[class_dir] = images
            image_paths_dict[class_dir] = image_paths
    
    return images_dict, image_paths_dict


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return 1 - cosine(vec1, vec2)


def predict_class(embedding, class_means, threshold):
    """Predict class based on cosine similarity with class means"""
    best_similarity = -1
    best_class = STRANGER_CLASS
    
    for class_name, mean_emb in class_means.items():
        similarity = cosine_similarity(embedding, mean_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_class = class_name
    
    # If similarity is below threshold, classify as stranger
    if best_similarity < threshold:
        return STRANGER_CLASS, best_similarity
    
    return best_class, best_similarity


def compute_metrics_per_class(y_true, y_pred, classes):
    """Compute metrics for each class (class vs rest)"""
    metrics = {}
    
    # Convert to numpy arrays for vectorized operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for cls in classes:
        if cls == STRANGER_CLASS:
            continue
        
        # Binary classification: this class vs all others
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # ROC AUC (need probabilities, use similarity scores)
        try:
            # For ROC AUC, we need to get similarity scores for this class
            # This is a simplified version - in practice you'd need to store similarity scores
            roc_auc = 0.0  # Will be computed separately with similarity scores
        except:
            roc_auc = 0.0
        
        metrics[cls] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': float(roc_auc)  # Will be updated with actual similarity scores
        }
    
    return metrics


def compute_roc_auc_per_class(y_true, y_similarities, class_means, classes):
    """Compute ROC AUC for each class using similarity scores"""
    roc_aucs = {}
    
    for cls in classes:
        if cls == STRANGER_CLASS:
            continue
        
        # Get similarity scores for this class
        similarities = []
        binary_labels = []
        
        for i, true_label in enumerate(y_true):
            if cls in class_means:
                # Get similarity to this class's mean
                sim = y_similarities[i].get(cls, 0.0)
                similarities.append(sim)
                binary_labels.append(1 if true_label == cls else 0)
        
        if len(similarities) > 0 and len(set(binary_labels)) > 1:
            try:
                roc_auc = roc_auc_score(binary_labels, similarities)
                roc_aucs[cls] = float(roc_auc)
            except:
                roc_aucs[cls] = 0.0
        else:
            roc_aucs[cls] = 0.0
    
    return roc_aucs


def plot_confusion_matrix(y_true, y_pred, classes, threshold, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix (Threshold = {threshold:.3f})', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_vs_threshold(results, output_dir):
    """Plot metrics vs threshold for all classes - separate files for each metric"""
    thresholds = sorted(results.keys())
    
    # Get all classes (excluding stranger)
    classes = set()
    for thresh_results in results.values():
        for cls in thresh_results['metrics_per_class'].keys():
            if cls != STRANGER_CLASS:
                classes.add(cls)
    classes = sorted(list(classes))
    
    # Plot each metric separately
    metrics_to_plot = ['accuracy', 'precision', 'recall']
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for cls in classes:
            values = []
            for thresh in thresholds:
                if cls in results[thresh]['metrics_per_class']:
                    values.append(results[thresh]['metrics_per_class'][cls][metric])
                else:
                    values.append(0.0)
            
            ax.plot(thresholds, values, marker='o', label=cls, linewidth=2, markersize=6)
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} vs Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(thresholds), max(thresholds)])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{metric}_vs_threshold.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {metric.capitalize()} plot saved to {output_path}")


def run_experiment(data_dir='data', output_dir='results/cosine_similarity', 
                   thresholds=None, test_size=0.2, random_state=42):
    """Run the cosine similarity experiment"""
    
    if thresholds is None:
        thresholds = np.arange(0.3, 0.95 + 0.05, 0.05)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("📂 Loading images...")
    images_dict, _ = load_images_from_directory(data_dir)
    
    # Separate stranger class from training classes
    train_classes = []
    stranger_images = []
    
    if STRANGER_CLASS in images_dict:
        stranger_images = images_dict[STRANGER_CLASS]
        print(f"  Found {len(stranger_images)} images in '{STRANGER_CLASS}' class (test only)")
        del images_dict[STRANGER_CLASS]
    
    train_classes = list(images_dict.keys())
    print(f"  Training classes: {train_classes}")
    
    if len(train_classes) == 0:
        print("❌ No training classes found!")
        return
    
    # Split images into train and test for each class
    print("\n📊 Splitting data into train and test sets...")
    train_images_dict = {}
    test_images_dict = {}
    
    for class_name in train_classes:
        images = images_dict[class_name]
        train_imgs, test_imgs = train_test_split(
            images, test_size=test_size, random_state=random_state
        )
        train_images_dict[class_name] = train_imgs
        test_images_dict[class_name] = test_imgs
        print(f"  {class_name}: {len(train_imgs)} train, {len(test_imgs)} test")
    
    # Extract embeddings for all training classes
    print("\n🔄 Extracting embeddings from training data...")
    train_embeddings_dict = {}
    
    for class_name in train_classes:
        train_imgs = train_images_dict[class_name]
        print(f"  Processing {len(train_imgs)} training images for '{class_name}'...")
        embeddings, _ = extract_embeddings(train_imgs, DEVICE, class_name=class_name)
        if len(embeddings) > 0:
            train_embeddings_dict[class_name] = embeddings
    
    # Compute class means from training data
    print("\n📊 Computing class means from training data...")
    class_means = {}
    for class_name, embeddings in train_embeddings_dict.items():
        if len(embeddings) > 0:
            class_means[class_name] = np.mean(embeddings, axis=0)
            print(f"  {class_name}: mean embedding computed from {len(embeddings)} samples")
    
    # Prepare test data
    print("\n🔄 Preparing test data...")
    test_embeddings = []
    test_labels = []
    
    # Add test images from training classes
    for class_name in train_classes:
        test_imgs = test_images_dict[class_name]
        
        print(f"  Processing {len(test_imgs)} test images for '{class_name}'...")
        embeddings, _ = extract_embeddings(test_imgs, DEVICE, class_name=f"{class_name}_test")
        for emb in embeddings:
            test_embeddings.append(emb)
            test_labels.append(class_name)
    
    # Add stranger class images
    if len(stranger_images) > 0:
        print(f"  Processing {len(stranger_images)} images for '{STRANGER_CLASS}' class...")
        embeddings, _ = extract_embeddings(stranger_images, DEVICE, class_name=STRANGER_CLASS)
        for emb in embeddings:
            test_embeddings.append(emb)
            test_labels.append(STRANGER_CLASS)
    
    test_embeddings = np.array(test_embeddings)
    print(f"\n✅ Test set prepared: {len(test_embeddings)} samples")
    
    # Run experiments for different thresholds
    print(f"\n🧪 Running experiments for {len(thresholds)} threshold values...")
    all_results = {}
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        y_pred = []
        y_similarities = []  # Store similarities for each class
        
        for emb in test_embeddings:
            # Compute similarities to all class means
            similarities = {}
            for cls, mean_emb in class_means.items():
                similarities[cls] = cosine_similarity(emb, mean_emb)
            
            y_similarities.append(similarities)
            pred_class, _ = predict_class(emb, class_means, threshold)
            y_pred.append(pred_class)
        
        # Compute metrics
        all_classes = train_classes + [STRANGER_CLASS]
        metrics_per_class = compute_metrics_per_class(test_labels, y_pred, all_classes)
        
        # Update ROC AUC with actual similarity scores
        roc_aucs = compute_roc_auc_per_class(test_labels, y_similarities, class_means, all_classes)
        for cls in roc_aucs:
            if cls in metrics_per_class:
                metrics_per_class[cls]['roc_auc'] = roc_aucs[cls]
        
        # Overall accuracy
        overall_accuracy = accuracy_score(test_labels, y_pred)
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, f'confusion_matrix_thresh_{threshold:.3f}.png')
        plot_confusion_matrix(test_labels, y_pred, all_classes, threshold, cm_path)
        
        all_results[threshold] = {
            'overall_accuracy': float(overall_accuracy),
            'metrics_per_class': metrics_per_class,
            'confusion_matrix_path': cm_path
        }
    
    # Save results to JSON
    results_json = {}
    for thresh, results in all_results.items():
        results_json[str(thresh)] = {
            'overall_accuracy': results['overall_accuracy'],
            'metrics_per_class': results['metrics_per_class']
        }
    
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✅ Results saved to {json_path}")
    
    # Create metrics visualization (separate files)
    plot_metrics_vs_threshold(all_results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Training classes: {train_classes}")
    print(f"Test-only class: {STRANGER_CLASS if len(stranger_images) > 0 else 'None'}")
    print(f"Number of thresholds tested: {len(thresholds)}")
    print(f"Test set size: {len(test_embeddings)}")
    print(f"\nBest threshold (by overall accuracy):")
    
    best_thresh = max(all_results.keys(), key=lambda t: all_results[t]['overall_accuracy'])
    print(f"  Threshold: {best_thresh:.3f}")
    print(f"  Overall Accuracy: {all_results[best_thresh]['overall_accuracy']:.4f}")
    
    print(f"\nMetrics at best threshold:")
    for cls in train_classes:
        if cls in all_results[best_thresh]['metrics_per_class']:
            m = all_results[best_thresh]['metrics_per_class'][cls]
            print(f"  {cls}:")
            print(f"    Accuracy: {m['accuracy']:.4f}")
            print(f"    Precision: {m['precision']:.4f}")
            print(f"    Recall: {m['recall']:.4f}")
            print(f"    ROC AUC: {m['roc_auc']:.4f}")
    
    print("\n🎉 Experiment complete!")
    print(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Cosine similarity threshold experiment')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing class folders')
    parser.add_argument('--output_dir', type=str, default='results/cosine_similarity',
                       help='Output directory for results')
    parser.add_argument('--threshold_min', type=float, default=0.3,
                       help='Minimum threshold value')
    parser.add_argument('--threshold_max', type=float, default=0.95,
                       help='Maximum threshold value')
    parser.add_argument('--threshold_step', type=float, default=0.05,
                       help='Threshold step size')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio (0.0-1.0)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for train/test split')
    
    args = parser.parse_args()
    
    # Make threshold_max inclusive
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    
    run_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        thresholds=thresholds,
        test_size=args.test_size,
        random_state=args.random_state
    )

