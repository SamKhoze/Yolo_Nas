
from torchinfo import summary
from super_gradients.common.object_names import Models
from super_gradients.training import models
import torch
torch.__version__

# Inference with YOLONAS pretrained models


# YOLONAS comes in three flavors: yolo_nas_s, yolo_nas_m, and yolo_nas_l.

yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")


summary(model=yolo_nas_l,
        input_size=(16, 3, 640, 640),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
        )

# 💻 Inference via webcam

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.YOLOX_N, pretrained_weights="coco")

# We want to use cuda if available to speed up inference.
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

model.predict_webcam()
