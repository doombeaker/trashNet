import os
import torch
import numpy as np
from torchvision import transforms, datasets
import torchvision

saved_file = "./trash_classifier_40_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE USED:", device)

model_ft = torchvision.models.mobilenet_v2(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2), torch.nn.Linear(model_ft.last_channel, 40),
)

model_ft.to(device)

if os.path.exists(saved_file):
    print("checkpoint file loading...")
    params = torch.load(saved_file)
    model_ft.load_state_dict(params)

args = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(
    model_ft, args, "trash40.onnx", input_names=["input"], output_names=["output"]
)
