import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Returns
    ----------
    tuple[float, float]
        Loss and accuracy.

    """
    total_loss = 0
    total_correct = 0
    total = 0

    model.train()
    for batch, labels in loader:
        batch: torch.Tensor = batch.to(device)
        labels: torch.Tensor = labels.to(device)

        optimizer.zero_grad()
        output: torch.Tensor = model(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (output.argmax(-1) == labels.argmax(-1)).sum().item()
        total += batch.shape[0]

    return total_loss / len(loader), total_correct / total


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> tuple[float, float]:
    """Evaluate the model for one epoch.

    Returns
    ----------
    tuple[float, float]
        Loss and accuracy.

    """
    total_loss = 0
    total_correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch, labels in loader:
            batch: torch.Tensor = batch.to(device)
            labels: torch.Tensor = labels.to(device)

            output: torch.Tensor = model(batch)
            loss = criterion(output, labels)

            total_loss += loss.item()
            total_correct += (output.argmax(-1) == labels.argmax(-1)).sum().item()
            total += batch.shape[0]

    return total_loss / len(loader), total_correct / total
