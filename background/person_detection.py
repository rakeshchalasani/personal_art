import torch

import os
import glob
import cv2
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from PIL import Image

from torchvision.transforms import Compose
import torchvision.transforms as transforms
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from src.models.modnet import MODNet


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

    def __modnet_prediction(self):

        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define hyper-parameters
        ref_size = 512

        # define image to tensor transform
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        model_path = './MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'

        # create MODNet and load the pre-trained ckpt
        modnet = MODNet(backbone_pretrained=False)
        if device == torch.device("cuda"):
            modnet = nn.DataParallel(modnet).cuda()
            modnet.load_state_dict(torch.load(model_path, map_location = device))
        else:
            state_dict = torch.load(model_path, map_location = device)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v

            modnet.load_state_dict(new_state_dict)

        modnet.eval()

        modnet.to(device)

        # unify image channels to 3
        im = np.asarray(self.image * 255.0).astype(np.uint8)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im, True)

        # resize and save matte
        matte = torch.clamp(F.interpolate(matte, size=(im_h, im_w), mode='area'), 0, 1)
        matte = matte[0][0].data.cpu().numpy()

        return matte


    def detect(self, model_type = 'dpt'):
        """
        Run segmentation network
        """
        print("Personal detection - initializing")
        if model_type == 'dpt':

            dpt_prediction_hybrid = self.__dpt_prediction(dpt_model_type='dpt_hybrid', optimize=True)
            dpt_prediction_large = self.__dpt_prediction(dpt_model_type='dpt_large', optimize=True)

            return np.minimum(dpt_prediction_large, dpt_prediction_hybrid)

        if model_type == 'modnet':

            return self.__modnet_prediction()

        return None

