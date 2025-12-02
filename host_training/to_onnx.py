import torch
import torch.onnx as onnx
import torch.nn as nn

from model import FaceRecognitionModel


def convert_model_to_onnx(model: nn.Module, filename: str):
    model.eval()
    example_input = (torch.randn(1, 3, 160, 160),)
    onnx.export(
        model,
        example_input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


if __name__ == "__main__":
    model = FaceRecognitionModel(3)
    model.load_state_dict(
        torch.load(
            "models/model_2025-11-28_00-04-02.pth", map_location=torch.device("cpu")
        )
    )
    convert_model_to_onnx(model, "model.onnx")
