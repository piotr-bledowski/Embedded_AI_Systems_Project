"""
Simple cosine similarity model for face recognition.
Provides functions for training (computing means) and inference.
"""
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
import pickle

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STRANGER_CLASS = 'stranger'


def extract_embedding(image, device=DEVICE):
    """
    Extract embedding from a single image.
    
    Args:
        image: numpy array (BGR format) or path to image file
        device: device to run on ('cuda' or 'cpu')
    
    Returns:
        numpy array: 512-dimensional embedding vector, or None if face not detected
    """
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Load image if path provided
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return None
    else:
        img = image
    
    try:
        # Convert BGR to RGB for facenet
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)
        if face is None:
            return None
        
        face = face.unsqueeze(0).to(device)
        emb = resnet(face).detach().cpu().numpy()[0]
        return emb
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


def extract_embeddings_batch(images, device=DEVICE):
    """
    Extract embeddings from multiple images.
    
    Args:
        images: list of numpy arrays (BGR format) or list of image paths
        device: device to run on ('cuda' or 'cpu')
    
    Returns:
        list: embeddings (None for images where face not detected)
    """
    embeddings = []
    for img in images:
        emb = extract_embedding(img, device)
        embeddings.append(emb)
    return embeddings


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: numpy array
        vec2: numpy array
    
    Returns:
        float: cosine similarity (0 to 1, where 1 is most similar)
    """
    return 1 - cosine(vec1, vec2)


def train(images_dict, device=DEVICE):
    """
    Train the cosine similarity model by computing mean embeddings for each class.
    
    Args:
        images_dict: dictionary mapping class names to lists of images (BGR format or paths)
        device: device to run on ('cuda' or 'cpu')
    
    Returns:
        dict: mapping from class names to mean embedding vectors
    """
    class_means = {}
    
    for class_name, images in images_dict.items():
        print(f"Processing {len(images)} images for class '{class_name}'...")
        embeddings = extract_embeddings_batch(images, device)
        
        # Filter out None embeddings (failed face detection)
        valid_embeddings = [e for e in embeddings if e is not None]
        
        if len(valid_embeddings) > 0:
            class_means[class_name] = np.mean(valid_embeddings, axis=0)
            print(f"  Computed mean embedding from {len(valid_embeddings)} valid samples")
        else:
            print(f"  Warning: No valid embeddings for class '{class_name}'")
    
    return class_means


def predict(embedding, class_means, threshold=0.6):
    """
    Predict class for an embedding using cosine similarity.
    
    Args:
        embedding: numpy array, 512-dimensional embedding vector
        class_means: dict mapping class names to mean embedding vectors
        threshold: float, minimum similarity to classify as a known class
    
    Returns:
        tuple: (predicted_class, similarity_score)
    """
    if embedding is None:
        return STRANGER_CLASS, 0.0
    
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


def predict_image(image, class_means, threshold=0.6, device=DEVICE):
    """
    Predict class for an image.
    
    Args:
        image: numpy array (BGR format) or path to image file
        class_means: dict mapping class names to mean embedding vectors
        threshold: float, minimum similarity to classify as a known class
        device: device to run on ('cuda' or 'cpu')
    
    Returns:
        tuple: (predicted_class, similarity_score)
    """
    embedding = extract_embedding(image, device)
    return predict(embedding, class_means, threshold)


def save_model(class_means, filepath):
    """
    Save the trained model (class means) to a file.
    
    Args:
        class_means: dict mapping class names to mean embedding vectors
        filepath: path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(class_means, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a trained model (class means) from a file.
    
    Args:
        filepath: path to the saved model
    
    Returns:
        dict: mapping from class names to mean embedding vectors
    """
    with open(filepath, 'rb') as f:
        class_means = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return class_means


# Example usage
if __name__ == '__main__':
    # Example: Load images from directory structure
    def load_images_from_directory(data_dir='data'):
        """Helper function to load images from directory"""
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
                    images.append(img_path)  # Store paths, will load on demand
            
            if len(images) > 0:
                images_dict[class_dir] = images
        
        return images_dict
    
    # Example training
    print("Loading images...")
    images_dict = load_images_from_directory('data')
    
    # Remove stranger class if present (not used for training)
    if STRANGER_CLASS in images_dict:
        del images_dict[STRANGER_CLASS]
    
    print("Training model...")
    class_means = train(images_dict, DEVICE)
    
    # Save model
    save_model(class_means, 'cosine_similarity_model.pkl')
    
    # Example inference
    print("\nExample inference:")
    if len(images_dict) > 0:
        # Get first image from first class for testing
        first_class = list(images_dict.keys())[0]
        test_image_path = images_dict[first_class][0]
        
        predicted_class, similarity = predict_image(test_image_path, class_means, threshold=0.6)
        print(f"Image: {test_image_path}")
        print(f"Predicted: {predicted_class} (similarity: {similarity:.4f})")

