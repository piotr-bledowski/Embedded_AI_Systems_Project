import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import umap
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.manifold import TSNE


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def compute_class_averages(embeddings_dict):
    """Compute average embedding for each class"""
    class_averages = {}
    for class_name, embs in embeddings_dict.items():
        if len(embs) > 0:
            class_averages[class_name] = np.mean(embs, axis=0)
    return class_averages


def reduce_dimensions(embeddings, method='tsne', n_components=2):
    """Perform dimensionality reduction"""
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42, verbose=False)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'tsne' or 'umap'")
    
    return reduced


def visualize_embeddings(embeddings_dict, class_averages, reduced_embeddings, 
                         reduced_averages, output_file='visualization.png'):
    """Create scatter plot with individual points and class averages"""
    STRANGER_CLASS = 'stranger'
    
    # Define bright, contrastive colors for training classes
    class_colors = {
        'Kinga': '#FF6B35',      # Orange
        'Pawel': '#004E89',      # Blue
        'Piotr': '#00A651',      # Green
    }
    
    # Get class names in order (stranger last if present)
    class_names = [name for name in embeddings_dict.keys() if name != STRANGER_CLASS]
    if STRANGER_CLASS in embeddings_dict:
        class_names.append(STRANGER_CLASS)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot individual embeddings for all classes
    start_idx = 0
    for class_name in class_names:
        num_points = len(embeddings_dict[class_name])
        end_idx = start_idx + num_points
        
        if class_name == STRANGER_CLASS:
            # Plot stranger class in gray
            ax.scatter(reduced_embeddings[start_idx:end_idx, 0],
                      reduced_embeddings[start_idx:end_idx, 1],
                      c='gray', label=STRANGER_CLASS, alpha=0.5, s=50, edgecolors='none')
        else:
            # Get color for this class (use predefined or fallback)
            if class_name in class_colors:
                color = class_colors[class_name]
            else:
                # Fallback for other classes
                color = plt.cm.tab10(len(class_names) % 10)
            
            # Plot individual points
            ax.scatter(reduced_embeddings[start_idx:end_idx, 0],
                      reduced_embeddings[start_idx:end_idx, 1],
                      c=color, label=class_name, alpha=0.7, s=50, edgecolors='none')
        
        start_idx = end_idx
    
    # Plot class averages with distinct colors (darker/more saturated)
    for class_name in class_names:
        if class_name != STRANGER_CLASS and class_name in reduced_averages:
            if class_name in class_colors:
                base_color = class_colors[class_name]
                # Convert hex to RGB and darken
                hex_color = base_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                darker_color = tuple(c * 0.6 for c in rgb)  # Darker shade
            else:
                darker_color = (0.3, 0.3, 0.3)  # Fallback dark gray
            
            ax.scatter(reduced_averages[class_name][0],
                      reduced_averages[class_name][1],
                      c=[darker_color], marker='*', s=500, 
                      edgecolors='black', linewidths=2,
                      label=f'{class_name} (mean)', zorder=10)
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Face Embeddings Visualization (2D)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize face embeddings with dimensionality reduction')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'],
                       help='Dimensionality reduction method (tsne or umap)')
    parser.add_argument('--output', type=str, default='visualization.png',
                       help='Output file for the visualization')
    args = parser.parse_args()
    
    # Load images grouped by class
    embeddings_dict = {}
    all_embeddings = []
    all_labels = []
    
    print("📂 Loading images and extracting embeddings...")
    class_dirs = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join('data', class_dir)
        
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
        
        if len(images) == 0:
            tqdm.write(f"⚠️ No valid images found in {class_dir}")
            continue
        
        embeddings, valid_indices = extract_embeddings(images, DEVICE, class_name=class_dir)
        
        if len(embeddings) > 0:
            embeddings_dict[class_dir] = embeddings
            all_embeddings.extend(embeddings)
            all_labels.extend([class_dir] * len(embeddings))
    
    if len(all_embeddings) == 0:
        print("❌ No embeddings extracted. Exiting.")
        return
    
    print(f"\n✅ Extracted {len(all_embeddings)} embeddings from {len(embeddings_dict)} classes")
    
    # Compute class averages (excluding stranger)
    print("\n📊 Computing class averages...")
    STRANGER_CLASS = 'stranger'
    embeddings_dict_for_averages = {k: v for k, v in embeddings_dict.items() if k != STRANGER_CLASS}
    class_averages = compute_class_averages(embeddings_dict_for_averages)
    print(f"✅ Computed averages for {len(class_averages)} classes (excluding {STRANGER_CLASS})")
    
    # Combine all embeddings and averages for dimensionality reduction
    all_embeddings_array = np.array(all_embeddings)
    # Only include averages for non-stranger classes
    average_embeddings_array = np.array([class_averages[cls] for cls in embeddings_dict_for_averages.keys()])
    
    # Perform dimensionality reduction on all data together
    print(f"\n🔄 Performing {args.method.upper()} dimensionality reduction...")
    combined_embeddings = np.vstack([all_embeddings_array, average_embeddings_array])
    with tqdm(total=1, desc=f"Running {args.method.upper()}") as pbar:
        reduced_combined = reduce_dimensions(combined_embeddings, method=args.method)
        pbar.update(1)
    
    # Split back into individual and average embeddings
    n_individual = len(all_embeddings_array)
    reduced_embeddings = reduced_combined[:n_individual]
    reduced_averages_array = reduced_combined[n_individual:]
    
    # Create dictionary for reduced averages (only for non-stranger classes)
    reduced_averages = {}
    for idx, class_name in enumerate(embeddings_dict_for_averages.keys()):
        reduced_averages[class_name] = reduced_averages_array[idx]
    
    # Visualize
    print("\n🎨 Creating visualization...")
    visualize_embeddings(embeddings_dict, class_averages, reduced_embeddings, 
                        reduced_averages, output_file=args.output)
    
    print("\n🎉 Visualization complete!")


if __name__ == '__main__':
    main()