#!/usr/bin/env python
# coding: utf-8
# %%
# # Import Libraries

# %%
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable

from torchvision import datasets, transforms
import torchvision

import sys, os
import matplotlib.pyplot as plt

import custom_resnet
from utils import CIFAR10Dataset, CIFAR10DataModule
from plots import plot_losses,plot_images 

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np

from utils import display_mis_images 
from utils import get_misclassified_data

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

import warnings
warnings.filterwarnings('ignore')
os.system("mkdir images")


# %%
### this is for running in local ###
import os
try:
    os.environ['HTTP_PROXY']='http://185.46.212.90:80'
    os.environ['HTTPS_PROXY']='http://185.46.212.90:80'
    print ("proxy_exported")
except:
    None
# # Model Params, optimizer, loss criterion and model summary
# Can't emphasize on how important viewing Model Summary is.
# Unfortunately, there is no in-built model visualizer, so we have to take external help

# %%
import pytorch_lightning as pl

# Initialize DataModule and Model
data_module = CIFAR10DataModule()
data_module.setup('fit')


# %%
print (len(data_module.train_loader()))
print (len(data_module.val_loader()))


# %%
seed_everything(42, workers=True)

m = custom_resnet.ResNetLightningModel()

# %%
trainer = pl.Trainer(max_epochs=m.EPOCHS,precision=32, accelerator="gpu", devices=1)
# Then you can train
trainer.fit(m,data_module.train_loader(), data_module.val_loader())


# %%
print(trainer.callback_metrics)


# %%
torch.save(m.state_dict(), "model_weights.pt")
m.load_state_dict(torch.load("model_weights.pt"))

# %%
cuda = torch.cuda.is_available()
device = torch.device("cpu" if cuda else "cpu")
print(device)


# %%
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ## Displaying Sample Miss Classified Images

# %%
# Get the misclassified data from test dataset
misclassified_data = get_misclassified_data(m, device, data_module.val_loader())
miss_classified_images = display_mis_images(misclassified_data,10, classes)
miss_classified_images.savefig("images/miss_class.jpg")
#misclassified_data.savefig("images/miss_class.jpg")


# ## Displaying Sample Train Dataset after trasformation

# %%
batch_data_train, batch_label_train = next(iter(data_module.train_loader()))
figure_train = plot_images(batch_data_train, batch_label_train.tolist(), 12, 3, 'CIFAR10')
figure_train.savefig("images/train_dataset.jpg")


# ## Displaying Sample Test Dataset

# %%
batch_data_test, batch_label_test = next(iter(data_module.val_loader()))
figure_test = plot_images(batch_data_test, batch_label_test.tolist(), 20, 4, 'CIFAR10')
figure_test.savefig("images/test_dataset.jpg")


# ## Grad Cam Images

# %%
from utils import display_gradcam_output

# Denormalize the data using test mean and std deviation
inv_normalize = transforms.Normalize(
    mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
    std=[1/0.23, 1/0.23, 1/0.23]
)

target_layers = [m.resnet2[-1]]
targets = None

figure_grad = display_gradcam_output(misclassified_data, classes, inv_normalize, m, target_layers, targets, number_of_samples=20, transparency=0.80)
figure_grad.savefig("images/grad.jpg")


# %%
