import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import numpy as np
import pickle

DATASET = "dataset/"  # folder with images
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------
# 1️⃣ Load images
# -------------------------
def load_images():
    images = []
    labels = []
    persons = os.listdir(DATASET)

    for person in persons:
        folder = os.path.join(DATASET, person)
        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            img = cv2.imread(path)
            if img is None:
                continue

            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # facenet uses RGB
            labels.append(person)

    return images, labels

# -------------------------
# 2️⃣ Extract embeddings
# -------------------------
def extract_embeddings(images, device=DEVICE, verbose=True):
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    embeddings = []

    for idx, img in enumerate(images):
        try:
            face = mtcnn(img)
            if face is None:
                if verbose:
                    print(f"⚠️ No face detected for image {idx}")
                embeddings.append(None)
                continue

            face = face.unsqueeze(0).to(device)
            emb = resnet(face).detach().cpu().numpy()[0]
            embeddings.append(emb)

        except Exception as e:
            if verbose:
                print(f"❌ Error processing image {idx}: {e}")
            embeddings.append(None)

    return embeddings

# -------------------------
# 3️⃣ Dataset for PyTorch
# -------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, label2idx):
        self.X = [e for e in embeddings if e is not None]
        self.y = [label2idx[labels[i]] for i, e in enumerate(embeddings) if e is not None]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# -------------------------
# 4️⃣ MLP model
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# 5️⃣ Main workflow
# -------------------------
images, labels = load_images()
embeddings = extract_embeddings(images)

# Map labels to integers
unique_labels = sorted(list(set([labels[i] for i, e in enumerate(embeddings) if e is not None])))
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
idx2label = {idx: label for label, idx in label2idx.items()}

dataset = EmbeddingDataset(embeddings, labels, label2idx)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create MLP
mlp = SimpleMLP(input_dim=512, num_classes=len(unique_labels)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

# Train
EPOCHS = 20
for epoch in range(EPOCHS):
    mlp.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = mlp(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

# Save model and label mapping
torch.save(mlp.state_dict(), "mlp_classifier.pth")
pickle.dump(label2idx, open("label2idx.pkl", "wb"))

print("🎉 Training completed!")
