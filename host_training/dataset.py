import cv2
import torch
from torch.utils.data import Dataset
from facenet_pytorch.models.mtcnn import MTCNN

import os
from typing import Literal, Mapping


class FaceRecognitionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        person_label_mapping: Mapping[str, int],
        split: Literal["train", "test"] = "train",
    ) -> None:
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        face_extractor = MTCNN(image_size=160, margin=0)

        for person in os.listdir(os.path.join(dataset_path, split)):
            if person not in person_label_mapping:
                continue

            folder = os.path.join(dataset_path, split, person)

            if not os.path.isdir(folder):
                continue

            for img_name in os.listdir(folder):
                path = os.path.join(folder, img_name)

                img = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
                if img is None:
                    continue

                face = face_extractor(img)
                if face is None:
                    continue

                self.images.append(face)

                label = torch.zeros(len(person_label_mapping), dtype=torch.float)
                label[person_label_mapping[person]] = 1
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]
