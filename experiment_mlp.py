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


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (MLP)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_experiment(data_dir='data', output_dir='results/mlp', 
                   test_size=0.2, random_state=42, epochs=20, batch_size=16, lr=1e-3,
                   stranger_approach='4class', confidence_threshold=0.7):
    """
    Run the MLP experiment
    
    Args:
        stranger_approach: '4class' or 'threshold'
            - '4class': Train with 4 classes including stranger
            - 'threshold': Train with 3 classes, use confidence threshold to detect strangers
        confidence_threshold: Only used if stranger_approach='threshold'
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
    
    # Create dataset and dataloader
    train_dataset = EmbeddingDataset(train_embeddings, train_labels, label2idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create and train MLP
    print(f"\n🧠 Training MLP classifier...")
    mlp = SimpleMLP(input_dim=512, num_classes=len(unique_labels)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    
    for epoch in range(epochs):
        mlp.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = mlp(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
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
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    mlp.eval()
    test_embeddings_tensor = torch.tensor(np.array(test_embeddings), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        outputs = mlp(test_embeddings_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        if stranger_approach == 'threshold':
            # Use confidence threshold approach
            max_probs, predicted = torch.max(probabilities, 1)
            y_pred = []
            for i, (max_prob, pred_idx) in enumerate(zip(max_probs, predicted)):
                if max_prob.item() < confidence_threshold:
                    y_pred.append(STRANGER_CLASS)
                else:
                    y_pred.append(idx2label[pred_idx.item()])
        else:
            # Standard classification
            _, predicted = torch.max(outputs, 1)
            y_pred = [idx2label[pred.item()] for pred in predicted]
    
    # Compute metrics
    all_classes = train_classes + [STRANGER_CLASS]
    metrics_per_class = compute_metrics_per_class(test_labels, y_pred, all_classes)
    overall_accuracy = accuracy_score(test_labels, y_pred)
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, y_pred, all_classes, cm_path)
    
    # Save results
    results = {
        'overall_accuracy': float(overall_accuracy),
        'metrics_per_class': metrics_per_class,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'stranger_approach': stranger_approach,
            'confidence_threshold': confidence_threshold if stranger_approach == 'threshold' else None
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
    if stranger_approach == 'threshold':
        print(f"Confidence threshold: {confidence_threshold}")
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
                       help='Confidence threshold for stranger detection (only used with threshold approach)')
    
    args = parser.parse_args()
    
    run_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stranger_approach=args.stranger_approach,
        confidence_threshold=args.confidence_threshold
    )

