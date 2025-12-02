import onnx
import onnxruntime
import torch

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: torch.randn(1, 3, 160, 160).numpy()}
ort_outputs = ort_session.run(None, ort_inputs)
print(ort_outputs)
