import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
import json

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
        for file in sorted(os.listdir(image_dir)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(image_dir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
        
        if len(images) > 0:
            images_dict[class_dir] = images
    
    return images_dict


class EmbeddingDataset(Dataset):
    """Dataset for embeddings"""
    def __init__(self, embeddings, labels, label2idx):
        self.X = [e for e in embeddings if e is not None]
        self.y = [label2idx[labels[i]] for i, e in enumerate(embeddings) if e is not None]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class SimpleMLP(nn.Module):
    """MLP classifier from train_v2.py"""
    def __init__(self, input_dim=512, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


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
        
        metrics[cls] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': 0.0  # Not computed for MLP
        }
    
    return metrics


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


def plot_confusion_matrix(y_true, y_pred, classes, output_path, threshold=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    if threshold is not None:
        plt.title(f'Confusion Matrix (MLP, Threshold = {threshold:.3f})', fontsize=14, fontweight='bold')
    else:
        plt.title('Confusion Matrix (MLP)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir, suffix=''):
    """Plot training curves for loss and accuracy"""
    epochs = range(1, len(train_losses) + 1)
    num_epochs = len(train_losses)
    
    # Plot loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # Set x-axis to only show integer epochs
    ax.set_xticks(range(1, num_epochs + 1))
    ax.set_xlim([0.5, num_epochs + 0.5])
    plt.tight_layout()
    loss_path = os.path.join(output_dir, f'loss_curves{suffix}.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss curves saved to {loss_path}")
    
    # Plot accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2, marker='o')
    ax.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # Set x-axis to only show integer epochs
    ax.set_xticks(range(1, num_epochs + 1))
    ax.set_xlim([0.5, num_epochs + 0.5])
    plt.tight_layout()
    acc_path = os.path.join(output_dir, f'accuracy_curves{suffix}.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Accuracy curves saved to {acc_path}")


def plot_metrics_vs_threshold_mlp(results, output_dir):
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
        ax.set_title(f'{metric.capitalize()} vs Threshold (MLP)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(thresholds), max(thresholds)])
        # Set x-axis to show reasonable number of ticks
        if len(thresholds) <= 15:
            ax.set_xticks(thresholds)
        else:
            # Show every 2nd or 3rd threshold if too many
            step = max(1, len(thresholds) // 15)
            ax.set_xticks(thresholds[::step])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{metric}_vs_threshold.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {metric.capitalize()} plot saved to {output_path}")


def plot_fpr_vs_threshold_mlp(results, output_dir):
    """Plot FPR vs threshold for all main classes (MLP)"""
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
    ax.set_title('False Positive Rate vs Threshold (MLP)\n(Stranger images misclassified)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(thresholds), max(thresholds)])
    max_fpr_vals = [max(results[t]['fpr_per_class'].values()) if 'fpr_per_class' in results[t] and len(results[t]['fpr_per_class']) > 0 else 0 for t in thresholds]
    max_fpr = max(max_fpr_vals) if max_fpr_vals else 0
    ax.set_ylim([0, max_fpr * 1.1 if max_fpr > 0 else 0.1])
    
    # Set x-axis to show reasonable number of ticks
    if len(thresholds) <= 15:
        ax.set_xticks(thresholds)
    else:
        step = max(1, len(thresholds) // 15)
        ax.set_xticks(thresholds[::step])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fpr_vs_threshold.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ FPR plot saved to {output_path}")


def plot_fpr_vs_epochs(fpr_history, output_dir):
    """Plot FPR across epochs for each main class (MLP4)"""
    if len(fpr_history) == 0:
        print("⚠️ No FPR history to plot")
        return
    
    epochs = range(1, len(fpr_history) + 1)
    
    # Get all classes from first epoch
    classes = sorted(list(fpr_history[0].keys()))
    
    if len(classes) == 0:
        print("⚠️ No FPR data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cls in classes:
        fpr_values = [fpr_history[epoch-1].get(cls, 0.0) for epoch in epochs]
        ax.plot(epochs, fpr_values, marker='o', label=cls, linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_title('False Positive Rate vs Epochs (MLP4)\n(Stranger images misclassified)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(epochs) + 1))
    ax.set_xlim([0.5, len(epochs) + 0.5])
    max_fpr = max([max(epoch_fpr.values()) if epoch_fpr else 0 for epoch_fpr in fpr_history])
    ax.set_ylim([0, max_fpr * 1.1 if max_fpr > 0 else 0.1])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fpr_vs_epochs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ FPR vs epochs plot saved to {output_path}")


def run_experiment(data_dir='data', output_dir='results/mlp', 
                   test_size=0.2, random_state=42, epochs=20, batch_size=16, lr=1e-3,
                   stranger_approach='4class', confidence_threshold=0.7,
                   threshold_min=0.4, threshold_max=0.95, threshold_step=0.05):
    """
    Run the MLP experiment
    
    Args:
        stranger_approach: '4class' or 'threshold'
            - '4class': Train with 4 classes including stranger
            - 'threshold': Train with 3 classes, use confidence threshold to detect strangers
        confidence_threshold: Only used if stranger_approach='threshold' (single threshold)
        threshold_min, threshold_max, threshold_step: Used for threshold sweeping when threshold approach is used
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("📂 Loading images...")
    images_dict = load_images_from_directory(data_dir)
    
    # Separate stranger class from training classes
    train_classes = []
    stranger_images = []
    
    if STRANGER_CLASS in images_dict:
        stranger_images = images_dict[STRANGER_CLASS]
        print(f"  Found {len(stranger_images)} images in '{STRANGER_CLASS}' class")
        if stranger_approach == '4class':
            print(f"    Will be included in training")
        else:
            print(f"    Will be used for testing only (confidence threshold approach)")
            del images_dict[STRANGER_CLASS]
    else:
        if stranger_approach == '4class':
            print(f"  ⚠️ Warning: No '{STRANGER_CLASS}' class found, but 4class approach requested")
    
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
    
    # Handle stranger class based on approach
    stranger_train_images = []
    stranger_test_images = []
    
    if stranger_approach == '4class' and len(stranger_images) > 0:
        # Split stranger images for training
        stranger_train, stranger_test = train_test_split(
            stranger_images, test_size=test_size, random_state=random_state
        )
        stranger_train_images = stranger_train
        stranger_test_images = stranger_test
        train_images_dict[STRANGER_CLASS] = stranger_train
        test_images_dict[STRANGER_CLASS] = stranger_test
        print(f"  {STRANGER_CLASS}: {len(stranger_train)} train, {len(stranger_test)} test")
    elif stranger_approach == 'threshold' and len(stranger_images) > 0:
        # All stranger images go to test set
        stranger_test_images = stranger_images
        print(f"  {STRANGER_CLASS}: 0 train, {len(stranger_test_images)} test (threshold approach)")
    
    # Extract embeddings for all training classes
    print("\n🔄 Extracting embeddings from training data...")
    train_embeddings = []
    train_labels = []
    
    # Get all classes that will be in training
    training_class_list = train_classes.copy()
    if stranger_approach == '4class' and STRANGER_CLASS in train_images_dict:
        training_class_list.append(STRANGER_CLASS)
    
    for class_name in training_class_list:
        train_imgs = train_images_dict[class_name]
        print(f"  Processing {len(train_imgs)} training images for '{class_name}'...")
        embeddings, _ = extract_embeddings(train_imgs, DEVICE, class_name=class_name)
        for emb in embeddings:
            train_embeddings.append(emb)
            train_labels.append(class_name)
    
    # Create label mappings
    unique_labels = sorted(list(set(train_labels)))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    print(f"\n📊 Model will have {len(unique_labels)} output classes: {unique_labels}")
    
    # Split training data into train and validation sets
    print("\n📊 Splitting training data into train and validation sets...")
    # Convert to lists for train_test_split
    train_emb_list = train_embeddings.tolist() if isinstance(train_embeddings, np.ndarray) else train_embeddings
    
    train_emb_split, val_emb_split, train_labels_split, val_labels_split = train_test_split(
        train_emb_list, train_labels, test_size=0.2, random_state=random_state
    )
    
    train_dataset = EmbeddingDataset(train_emb_split, train_labels_split, label2idx)
    val_dataset = EmbeddingDataset(val_emb_split, val_labels_split, label2idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
    
    # Create and train MLP
    print(f"\n🧠 Training MLP classifier...")
    mlp = SimpleMLP(input_dim=512, num_classes=len(unique_labels)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    fpr_history = []  # Track FPR across epochs for MLP4
    
    # Prepare validation data with labels for FPR computation (for MLP4)
    val_embeddings_for_fpr = None
    val_labels_for_fpr = None
    if stranger_approach == '4class' and STRANGER_CLASS in unique_labels:
        # Use validation embeddings and labels directly (already split)
        val_embeddings_for_fpr = torch.tensor(np.array(val_emb_split), dtype=torch.float32).to(DEVICE)
        val_labels_for_fpr = val_labels_split  # Original string labels
    
    for epoch in range(epochs):
        # Training phase
        mlp.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = mlp(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc)
        
        # Validation phase
        mlp.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                outputs = mlp(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc)
        
        # Compute FPR on validation set for MLP4 (track across epochs)
        if stranger_approach == '4class' and val_embeddings_for_fpr is not None:
            mlp.eval()
            with torch.no_grad():
                val_outputs = mlp(val_embeddings_for_fpr)
                _, val_predicted = torch.max(val_outputs, 1)
                val_y_pred = [idx2label[pred.item()] for pred in val_predicted]
            
            # Compute FPR for this epoch
            epoch_fpr = compute_fpr_per_class(val_labels_for_fpr, val_y_pred, train_classes)
            fpr_history.append(epoch_fpr)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")
    
    print("✅ Training completed!")
    
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
    
    # Add stranger class images to test set
    if stranger_approach == '4class' and STRANGER_CLASS in test_images_dict:
        test_imgs = test_images_dict[STRANGER_CLASS]
        print(f"  Processing {len(test_imgs)} test images for '{STRANGER_CLASS}' class...")
        embeddings, _ = extract_embeddings(test_imgs, DEVICE, class_name=STRANGER_CLASS)
        for emb in embeddings:
            test_embeddings.append(emb)
            test_labels.append(STRANGER_CLASS)
    elif stranger_approach == 'threshold' and len(stranger_test_images) > 0:
        print(f"  Processing {len(stranger_test_images)} images for '{STRANGER_CLASS}' class...")
        embeddings, _ = extract_embeddings(stranger_test_images, DEVICE, class_name=STRANGER_CLASS)
        for emb in embeddings:
            test_embeddings.append(emb)
            test_labels.append(STRANGER_CLASS)
    
    test_embeddings_tensor = torch.tensor(np.array(test_embeddings), dtype=torch.float32).to(DEVICE)
    
    # Get model outputs once
    mlp.eval()
    with torch.no_grad():
        outputs = mlp(test_embeddings_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Handle different approaches
    if stranger_approach == 'threshold':
        # Test multiple thresholds
        thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
        print(f"\n🧪 Testing {len(thresholds)} threshold values...")
        
        all_results = {}
        
        for threshold in tqdm(thresholds, desc="Testing thresholds"):
            # Use confidence threshold approach
            max_probs, predicted = torch.max(probabilities, 1)
            y_pred = []
            for i, (max_prob, pred_idx) in enumerate(zip(max_probs, predicted)):
                if max_prob.item() < threshold:
                    y_pred.append(STRANGER_CLASS)
                else:
                    y_pred.append(idx2label[pred_idx.item()])
            
            # Compute metrics
            test_classes = set(test_labels + y_pred)
            all_classes = sorted(list(test_classes))
            metrics_per_class = compute_metrics_per_class(test_labels, y_pred, all_classes)
            
            # Compute FPR for each main class
            fpr_per_class = compute_fpr_per_class(test_labels, y_pred, train_classes)
            
            overall_accuracy = accuracy_score(test_labels, y_pred)
            
            # Save confusion matrix
            cm_path = os.path.join(output_dir, f'confusion_matrix_thresh_{threshold:.3f}.png')
            plot_confusion_matrix(test_labels, y_pred, all_classes, cm_path, threshold=threshold)
            
            all_results[threshold] = {
                'overall_accuracy': float(overall_accuracy),
                'metrics_per_class': metrics_per_class,
                'fpr_per_class': fpr_per_class,
                'confusion_matrix_path': cm_path
            }
        
        # Create metrics vs threshold plots
        print("\n📈 Creating metrics vs threshold plots...")
        plot_metrics_vs_threshold_mlp(all_results, output_dir)
        
        # Create FPR vs threshold plot
        plot_fpr_vs_threshold_mlp(all_results, output_dir)
        
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
        
        # Find best threshold and save training curves for it
        best_thresh = max(all_results.keys(), key=lambda t: all_results[t]['overall_accuracy'])
        print(f"\n📊 Best threshold: {best_thresh:.3f} (Accuracy: {all_results[best_thresh]['overall_accuracy']:.4f})")
        print(f"   Saving training curves for best threshold...")
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir, 
                           suffix=f'_best_thresh_{best_thresh:.3f}')
        
        # Save best model
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'mlp3_model.pth')
        model_data = {
            'model_state_dict': mlp.state_dict(),
            'label2idx': label2idx,
            'idx2label': idx2label,
            'threshold': float(best_thresh),
            'input_dim': 512,
            'num_classes': len(unique_labels)
        }
        torch.save(model_data, model_path)
        print(f"✅ Best model saved to {model_path}")
        print(f"   Threshold: {best_thresh:.3f}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Training classes: {training_class_list}")
        print(f"Stranger approach: {stranger_approach}")
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
    
    else:
        # 4class approach - single evaluation
        # Standard classification
        _, predicted = torch.max(outputs, 1)
        y_pred = [idx2label[pred.item()] for pred in predicted]
        
        # Compute metrics
        test_classes = set(test_labels + y_pred)
        all_classes = sorted(list(test_classes))
        metrics_per_class = compute_metrics_per_class(test_labels, y_pred, all_classes)
        overall_accuracy = accuracy_score(test_labels, y_pred)
        
        # Plot training curves
        print("\n📈 Creating training curves...")
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir)
        
        # Plot FPR across epochs
        if len(fpr_history) > 0:
            print("\n📈 Creating FPR vs epochs plot...")
            plot_fpr_vs_epochs(fpr_history, output_dir)
        
        # Compute FPR on test set for final evaluation
        fpr_per_class = compute_fpr_per_class(test_labels, y_pred, train_classes)
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(test_labels, y_pred, all_classes, cm_path)
        
        # Save best model
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'mlp4_model.pth')
        model_data = {
            'model_state_dict': mlp.state_dict(),
            'label2idx': label2idx,
            'idx2label': idx2label,
            'threshold': None,  # No threshold for 4class approach
            'input_dim': 512,
            'num_classes': len(unique_labels)
        }
        torch.save(model_data, model_path)
        print(f"\n✅ Best model saved to {model_path}")
        
        # Save results
        results = {
            'overall_accuracy': float(overall_accuracy),
            'metrics_per_class': metrics_per_class,
            'fpr_per_class': fpr_per_class,
            'hyperparameters': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'stranger_approach': stranger_approach
            }
        }
        
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {json_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Training classes: {training_class_list}")
        print(f"Stranger approach: {stranger_approach}")
        print(f"Test set size: {len(test_embeddings)}")
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
        print(f"\nMetrics per class:")
        for cls in train_classes:
            if cls in metrics_per_class:
                m = metrics_per_class[cls]
                print(f"  {cls}:")
                print(f"    Accuracy: {m['accuracy']:.4f}")
                print(f"    Precision: {m['precision']:.4f}")
                print(f"    Recall: {m['recall']:.4f}")
    
    print("\n🎉 Experiment complete!")
    print(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MLP classifier experiment')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing class folders')
    parser.add_argument('--output_dir', type=str, default='results/mlp',
                       help='Output directory for results')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio (0.0-1.0)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for train/test split')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--stranger_approach', type=str, default='4class',
                       choices=['4class', 'threshold'],
                       help='How to handle stranger class: 4class (train with 4 classes) or threshold (use confidence threshold)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Confidence threshold for stranger detection (only used with threshold approach, single threshold)')
    parser.add_argument('--threshold_min', type=float, default=0.4,
                       help='Minimum threshold value (for threshold approach sweeping)')
    parser.add_argument('--threshold_max', type=float, default=0.95,
                       help='Maximum threshold value (inclusive, for threshold approach sweeping)')
    parser.add_argument('--threshold_step', type=float, default=0.05,
                       help='Threshold step size (for threshold approach sweeping)')
    
    args = parser.parse_args()
    
    # Automatically append stranger approach to output directory
    # This creates separate directories for each approach
    base_output_dir = args.output_dir
    
    # Build output directory path based on approach
    if args.stranger_approach == 'threshold':
        # For threshold approach, use a generic name since we're testing multiple thresholds
        output_dir = os.path.join(base_output_dir, args.stranger_approach)
    else:
        # For 4class approach, just append the approach name
        output_dir = os.path.join(base_output_dir, args.stranger_approach)
    
    run_experiment(
        data_dir=args.data_dir,
        output_dir=output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stranger_approach=args.stranger_approach,
        confidence_threshold=args.confidence_threshold,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step
    )

