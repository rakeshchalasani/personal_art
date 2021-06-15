import torch

import os
import glob
import cv2
import argparse

import torch
import torch.nn.functional as F

import numpy as np

from torchvision.transforms import Compose
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


class PersonDetection():
    def __init__(self, image, model_type = 'DPT') -> None:
        self.image = image
        self.model_type = 'DPT' 

    def __dpt_prediction(self, dpt_model_type = 'dpt_hybrid', optimize = True):
        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fixed size image on which prediction is done. Resized to original image after prediction using interpolation.
        net_w = net_h = 480 

        # load network
        if dpt_model_type == "dpt_large":
            model_path = './DPT/weights/dpt_large-ade20k-b12dca68.pt'
            model = DPTSegmentationModel(
                150,
                path=model_path,
                backbone="vitl16_384",
            )
        elif dpt_model_type == "dpt_hybrid":
            model_path = './DPT/weights/dpt_hybrid-ade20k-53898607.pt'
            model = DPTSegmentationModel(
                150,
                path=model_path,
                backbone="vitb_rn50_384",
            )
        else:
            assert (
                False
            ), f"model_type '{dpt_model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid]"

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
            ]
        )

        model.eval()

        if optimize == True and device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

        model.to(device)

        img_input = transform({"image": self.image})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            out = model.forward(sample)

            prediction = torch.nn.functional.interpolate(
                out, size= self.image.shape[:2], mode="bicubic", align_corners=False
            )
            prediction = torch.argmax(prediction, dim=1) + 1
            prediction = torch.clamp(torch.where(prediction == 13, prediction, 0), 0, 1) # 13 is person index in ADE15k dataset.
            prediction = prediction.squeeze().cpu().numpy()

        return prediction


    def detect(self):
        """
        Run segmentation network
        """
        print("Personal detection - initializing")

        dpt_prediction_hybrid = self.__dpt_prediction(dpt_model_type='dpt_hybrid', optimize=True)
        dpt_prediction_large = self.__dpt_prediction(dpt_model_type='dpt_large', optimize=True)

        return np.minimum(dpt_prediction_large, dpt_prediction_hybrid)

        

