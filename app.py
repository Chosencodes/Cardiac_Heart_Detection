
import torch
import torchvision
import pytorch_lightning as pl
import numpy as np
import cv2
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class CardiacDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=None)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=4)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

device = torch.device("cpu")
model = CardiacDetectionModel.load_from_checkpoint(
    "best-checkpoint.ckpt",
    map_location=device
)
model.eval()

def detect_heart(image):
    img = np.array(Image.fromarray(image).convert("L"))
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    img = img / 255.0
    img = (img - 0.494) / 0.253

    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        pred = model(tensor)[0].cpu()

    heart_width = (pred[2] - pred[0]).item()
    ct_ratio    = heart_width / 224
    status      = "Enlarged (Possible Cardiomegaly)" if ct_ratio > 0.5 else "Normal"

    fig, axis = plt.subplots(1, 1, figsize=(6, 6))
    axis.imshow(img, cmap="bone")
    pred_box = patches.Rectangle(
        (pred[0], pred[1]), pred[2]-pred[0], pred[3]-pred[1],
        edgecolor="r", facecolor="none", linewidth=2, label="Heart Detection"
    )
    axis.add_patch(pred_box)
    axis.legend()
    axis.set_title(f"CT Ratio: {ct_ratio:.2f} | {status}")
    axis.axis("off")
    fig.savefig("output.png", bbox_inches="tight")
    plt.close()
    return "output.png", f"CT Ratio: {ct_ratio:.3f}", status

demo = gr.Interface(
    fn=detect_heart,
    inputs=gr.Image(label="Upload Chest X-Ray"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Text(label="Cardiothoracic Ratio"),
        gr.Text(label="Heart Status"),
    ],
    title="Cardiac Heart Detection",
    description="Upload a frontal chest X-ray to detect heart location and calculate cardiothoracic ratio.",
)
demo.launch()
