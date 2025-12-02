import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import os
import pickle
from datetime import datetime

from host_training.model import FaceRecognitionModel
from host_training.dataset import FaceRecognitionDataset
from host_training.training_utils import train_epoch, evaluate_epoch

DATASET_DIR = "dataset"
MODELS_DIR = "models"
HISTORY_DIR = "history"

LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 2
NUM_EPOCHS = 10

PERSON_LABEL_MAPPING = {
    "Kinga": 0,
    "Pawel": 1,
    "Piotr": 2,
}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    model = FaceRecognitionModel(len(PERSON_LABEL_MAPPING))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    train_dataset = FaceRecognitionDataset(DATASET_DIR, PERSON_LABEL_MAPPING, "train")
    print(f"Initialized train dataset with {len(train_dataset)} samples")
    test_dataset = FaceRecognitionDataset(DATASET_DIR, PERSON_LABEL_MAPPING, "test")
    print(f"Initialized test dataset with {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    history = {
        "Train loss": np.zeros((NUM_EPOCHS,), dtype=np.float64),
        "Train acc": np.zeros((NUM_EPOCHS,), dtype=np.float64),
        "Test loss": np.zeros((NUM_EPOCHS,), dtype=np.float64),
        "Test acc": np.zeros((NUM_EPOCHS,), dtype=np.float64),
    }

    progress_bar = tqdm(
        range(NUM_EPOCHS), desc="Training progress", unit="Epoch", leave=True
    )
    for epoch in progress_bar:
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)

        tqdm.write(f"Epoch {epoch + 1}:")
        tqdm.write(f" - Train loss: {train_loss}")
        tqdm.write(f" - Train acc: {train_acc}")
        tqdm.write(f" - Test loss: {test_loss}")
        tqdm.write(f" - Test acc: {test_acc}")
        tqdm.write("-" * 15)

        history["Train loss"][epoch] = train_loss
        history["Train acc"][epoch] = train_acc
        history["Test loss"][epoch] = test_loss
        history["Test acc"][epoch] = test_acc
    progress_bar.close()

    model_name = f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_path = os.path.join(HISTORY_DIR, f"{model_name}.pkl")
    with open(history_path, "wb") as file:
        pickle.dump(history, file)
    print(f"History saved to {history_path}")
