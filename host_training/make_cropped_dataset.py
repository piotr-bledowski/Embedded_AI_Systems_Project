import cv2
import torch
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN

import os

images: list[torch.Tensor] = []
face_extractor = MTCNN(image_size=160, margin=0)

for person in os.listdir("dataset/train"):
    folder = os.path.join("dataset/train", person)

    if not os.path.isdir(folder):
        continue

    for i, img_name in enumerate(os.listdir(folder)):
        path = os.path.join(folder, img_name)

        img = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
        if img is None:
            continue

        face: torch.Tensor = face_extractor(img)
        if face is None:
            continue

        cropped_img = face.permute(1, 2, 0).cpu().numpy()
        cropped_img = (cropped_img * 255).astype("uint8")
        cropped_img = Image.fromarray(cropped_img)
        cropped_img.save(f"cropped_dataset/{person}_{i}.jpeg", format="JPEG")
