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


def compute_fpr_per_class(y_true, y_pred, main_classes):
    """
    Compute False Positive Rate (FPR) for each main class.
    FPR = (stranger images misclassified as this class) / (total stranger images)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        main_classes: List of main classes (excluding stranger)
    
    Returns:
        Dictionary mapping class name to FPR
    """
    fpr_dict = {}
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Count total stranger images
    stranger_mask = (y_true == STRANGER_CLASS)
    total_strangers = np.sum(stranger_mask)
    
    if total_strangers == 0:
        # No stranger images, FPR is undefined
        for cls in main_classes:
            fpr_dict[cls] = 0.0
        return fpr_dict
    
    # For each main class, count how many strangers were misclassified as it
    for cls in main_classes:
        # False positives: stranger images predicted as this class
        fp = np.sum((y_true == STRANGER_CLASS) & (y_pred == cls))
        fpr = fp / total_strangers
        fpr_dict[cls] = float(fpr)
    
    return fpr_dict


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


def plot_fpr_vs_threshold(results, output_dir):
    """Plot FPR vs threshold for all main classes"""
    thresholds = sorted(results.keys())
    
    # Get all main classes (excluding stranger)
    classes = set()
    for thresh_results in results.values():
        if 'fpr_per_class' in thresh_results:
            for cls in thresh_results['fpr_per_class'].keys():
                classes.add(cls)
    classes = sorted(list(classes))
    
    if len(classes) == 0:
        print("⚠️ No FPR data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cls in classes:
        fpr_values = []
        for thresh in thresholds:
            if 'fpr_per_class' in results[thresh] and cls in results[thresh]['fpr_per_class']:
                fpr_values.append(results[thresh]['fpr_per_class'][cls])
            else:
                fpr_values.append(0.0)
        
        ax.plot(thresholds, fpr_values, marker='o', label=cls, linewidth=2, markersize=6)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_title('False Positive Rate vs Threshold\n(Stranger images misclassified)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(thresholds), max(thresholds)])
    max_fpr = max([max(results[t]['fpr_per_class'].values()) if 'fpr_per_class' in results[t] and len(results[t]['fpr_per_class']) > 0 else 0 for t in thresholds])
    ax.set_ylim([0, max_fpr * 1.1 if max_fpr > 0 else 0.1])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fpr_vs_threshold.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ FPR plot saved to {output_path}")


def compute_fpr_on_stranger_dataset(stranger_embeddings, class_means, train_classes, threshold):
    """
    Compute FPR on stranger dataset - how often strangers are classified as group members.
    FPR = (strangers classified as group members) / (total strangers)
    
    Args:
        stranger_embeddings: List of embeddings from stranger dataset
        class_means: Dictionary mapping class names to mean embeddings
        train_classes: List of group member class names
        threshold: Threshold value for classification
    
    Returns:
        float: FPR value (0.0 to 1.0)
    """
    if len(stranger_embeddings) == 0:
        return 0.0
    
    stranger_embeddings = np.array(stranger_embeddings)
    
    # Count how many strangers are classified as group members (above threshold)
    false_positives = 0
    
    for emb in stranger_embeddings:
        pred_class, best_similarity = predict_class(emb, class_means, threshold)
        # If predicted class is one of the group members (not STRANGER_CLASS), it's a false positive
        if pred_class in train_classes:
            false_positives += 1
    
    fpr = false_positives / len(stranger_embeddings)
    return float(fpr)


def load_stranger_dataset_images(stranger_dataset_dir):
    """
    Load images from a separate stranger dataset directory.
    
    Args:
        stranger_dataset_dir: Path to directory containing stranger images
    
    Returns:
        List of images (cv2 images)
    """
    if stranger_dataset_dir is None or not os.path.exists(stranger_dataset_dir):
        return []
    
    images = []
    
    # Check if it's a directory of images or contains subdirectories
    if os.path.isdir(stranger_dataset_dir):
        for file in sorted(os.listdir(stranger_dataset_dir)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(stranger_dataset_dir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
    
    return images


def plot_stranger_dataset_fpr_vs_threshold(stranger_fpr_results, output_dir):
    """
    Plot FPR vs threshold for stranger dataset evaluation - gray lines
    
    Args:
        stranger_fpr_results: Dictionary mapping threshold to FPR value
        output_dir: Output directory for the plot
    """
    if len(stranger_fpr_results) == 0:
        print("⚠️ No stranger dataset FPR data to plot")
        return
    
    thresholds = sorted(stranger_fpr_results.keys())
    fpr_values = [stranger_fpr_results[t] for t in thresholds]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with gray color
    ax.plot(thresholds, fpr_values, marker='o', color='gray', linewidth=2, markersize=6, label='FPR')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('FPR', fontsize=12)
    ax.set_title('FPR vs Threshold\n(Stranger Dataset Evaluation)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(thresholds), max(thresholds)])
    max_fpr = max(fpr_values) if fpr_values else 0
    ax.set_ylim([0, max_fpr * 1.1 if max_fpr > 0 else 0.1])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fpr_stranger_dataset_vs_threshold.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Stranger dataset FPR plot saved to {output_path}")


def run_experiment(data_dir='data', output_dir='results/cosine_similarity', 
                   thresholds=None, test_size=0.2, random_state=42,
                   stranger_dataset_dir=None):
    """
    Run the cosine similarity experiment
    
    Args:
        stranger_dataset_dir: Optional path to separate stranger dataset directory for evaluation
    """
    
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
        
        # Compute FPR for each main class
        fpr_per_class = compute_fpr_per_class(test_labels, y_pred, train_classes)
        
        # Overall accuracy
        overall_accuracy = accuracy_score(test_labels, y_pred)
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, f'confusion_matrix_thresh_{threshold:.3f}.png')
        plot_confusion_matrix(test_labels, y_pred, all_classes, threshold, cm_path)
        
        all_results[threshold] = {
            'overall_accuracy': float(overall_accuracy),
            'metrics_per_class': metrics_per_class,
            'fpr_per_class': fpr_per_class,
            'confusion_matrix_path': cm_path
        }
    
    # Save results to JSON
    results_json = {}
    for thresh, results in all_results.items():
        results_json[str(thresh)] = {
            'overall_accuracy': results['overall_accuracy'],
            'metrics_per_class': results['metrics_per_class'],
            'fpr_per_class': results['fpr_per_class']
        }
    
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✅ Results saved to {json_path}")
    
    # Create metrics visualization (separate files)
    plot_metrics_vs_threshold(all_results, output_dir)
    
    # Create FPR visualization
    plot_fpr_vs_threshold(all_results, output_dir)
    
    # Evaluate on stranger dataset if provided
    stranger_fpr_results = {}
    if stranger_dataset_dir is not None and os.path.exists(stranger_dataset_dir):
        print(f"\n🔍 Loading and evaluating on separate stranger dataset: {stranger_dataset_dir}")
        stranger_dataset_images = load_stranger_dataset_images(stranger_dataset_dir)
        
        if len(stranger_dataset_images) > 0:
            print(f"  Found {len(stranger_dataset_images)} images in stranger dataset")
            print(f"  Extracting embeddings...")
            stranger_embeddings, _ = extract_embeddings(stranger_dataset_images, DEVICE, class_name="stranger_dataset")
            
            if len(stranger_embeddings) > 0:
                print(f"  Evaluating on {len(stranger_embeddings)} stranger embeddings for {len(thresholds)} thresholds...")
                
                for threshold in tqdm(thresholds, desc="Evaluating stranger dataset"):
                    fpr = compute_fpr_on_stranger_dataset(stranger_embeddings, class_means, train_classes, threshold)
                    stranger_fpr_results[threshold] = fpr
                
                # Plot FPR vs threshold for stranger dataset
                plot_stranger_dataset_fpr_vs_threshold(stranger_fpr_results, output_dir)
                
                # Save stranger dataset results to JSON
                stranger_results_json = {str(t): fpr for t, fpr in stranger_fpr_results.items()}
                stranger_json_path = os.path.join(output_dir, 'stranger_dataset_fpr_results.json')
                with open(stranger_json_path, 'w') as f:
                    json.dump(stranger_results_json, f, indent=2)
                print(f"\n✅ Stranger dataset FPR results saved to {stranger_json_path}")
            else:
                print("  ⚠️ No valid embeddings extracted from stranger dataset")
        else:
            print(f"  ⚠️ No images found in stranger dataset directory: {stranger_dataset_dir}")
    
    # Find best threshold and save model
    best_thresh = max(all_results.keys(), key=lambda t: all_results[t]['overall_accuracy'])
    
    # Save best model
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'cosine_similarity_model.pkl')
    model_data = {
        'class_means': class_means,
        'threshold': float(best_thresh),
        'train_classes': train_classes
    }
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✅ Best model saved to {model_path}")
    print(f"   Threshold: {best_thresh:.3f}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Training classes: {train_classes}")
    print(f"Test-only class: {STRANGER_CLASS if len(stranger_images) > 0 else 'None'}")
    print(f"Number of thresholds tested: {len(thresholds)}")
    print(f"Test set size: {len(test_embeddings)}")
    print(f"\nBest threshold (by overall accuracy):")
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
    parser.add_argument('--stranger_dataset_dir', type=str, default=None,
                       help='Optional path to separate stranger dataset directory for evaluation')
    
    args = parser.parse_args()
    
    # Make threshold_max inclusive
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    
    run_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        thresholds=thresholds,
        test_size=args.test_size,
        random_state=args.random_state,
        stranger_dataset_dir=args.stranger_dataset_dir
    )

