import torch
import torch.nn as nn
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_of_classes: int) -> None:
        super(FaceRecognitionModel, self).__init__()

        self.resnet = InceptionResnetV1(pretrained="vggface2")
        self.fc = nn.Linear(512, num_of_classes)

        # Freeeze the imported models, only train our classifier
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.resnet(x)
        y = self.fc(embedding)
        return y
