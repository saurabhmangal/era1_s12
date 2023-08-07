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

import resnet

from utils import create_train_data_loader, create_test_data_loader
from utils import Cifar10SearchDataset
from utils import train_transforms, test_transforms
from utils import display_mis_images 
from utils import learning_r_finder
from utils import OneCycleLR_policy

from calc_loss_accuracy import model_training, model_testing_old#, model_testing 
from plots import plot_losses,plot_images 

import warnings
warnings.filterwarnings('ignore')

### this is for running in local ###
try:
    os.environ['HTTP_PROXY']='http://185.46.212.90:80'
    os.environ['HTTPS_PROXY']='http://185.46.212.90:80'
    print ("proxy_exported")
except:
    None


SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if cuda else "cpu")
print(device)


# Train/Test Data Loaders with Transformation
means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

train_transforms = train_transforms(means, stds)
test_transforms = test_transforms(means, stds)

train = Cifar10SearchDataset('./data', train=True, download=True, transform=train_transforms)
test = Cifar10SearchDataset('./data', train=False, download=True, transform=test_transforms)

dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)


# Model Params, optimizer, loss criterion and model summary
# Can't emphasize on how important viewing Model Summary is.
# Unfortunately, there is no in-built model visualizer, so we have to take external help

m = resnet.ResNet18().to(device)
optimizer = optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
summary(m, input_size=(3, 32, 32))

# # Calculating the max and min LR using one cycle LR policy

# to reset the model and optimizer to their initial state
learning_r_finder(m,optimizer,criterion, device, train_loader, n_iters=200,  end_lr=10)
#sys.exit()

# Let's Train and test our model
# ## using one cycle lr policy

EPOCHS = 20
scheduler = OneCycleLR_policy(optimizer,train_loader,EPOCHS,peak_value=5.0,div_factor=100,final_div_factor=100,max_lr=1.83E-03)


for epoch in range(EPOCHS):
    print("EPOCH: "+ str(epoch)),
    train_acc,train_losses = model_training(m, device, train_loader, optimizer, scheduler, criterion)
    test_acc,test_losses,miss_classified_data = model_testing_old(m, device, test_loader, criterion)

## Displaying Train Test Accuracy and Loss Plots
os.system('mkdir images')
fig = plot_losses(train_losses, train_acc, test_losses, test_acc)
fig.savefig('images/Accuracy & Loss.jpg')

## Displaying Sample Miss Classified Images
miss_classified_images = display_mis_images(miss_classified_data,10)
miss_classified_images.savefig("images/miss_class.jpg")

# # Displaying Sample Train Dataset after trasformation

batch_data_train, batch_label_train = next(iter(train_loader))
figure_train = plot_images(batch_data_train, batch_label_train.tolist(), 12, 3, 'CIFAR10')
figure_train.savefig("images/train_dataset.jpg")

## Displaying Sample Test Dataset
batch_data_test, batch_label_test = next(iter(test_loader))
figure_test = plot_images(batch_data_test, batch_label_test.tolist(), 20, 4, 'CIFAR10')
figure_test.savefig("images/test_dataset.jpg")


#Grad Cam Images
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np

model = m
target_layers = [m.layer4[-1]]

fig = plt.figure(figsize=(8,5))

for i in range (0,16):
    plt.subplot(4,int(16/4),i+1)
    plt.tight_layout()
    input_tensor = miss_classified_data[2][i].unsqueeze(0) 
    img = miss_classified_data[2][i].permute(1,2,0).cpu()
    img = img / 2.0 + 0.5
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 0, 2))
    npimg = (npimg - np.min(npimg)) / (np.max(npimg) - np.min(npimg))


    targets = [ClassifierOutputTarget(miss_classified_data[0][i])]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(npimg,grayscale_cams[0, :], use_rgb=True)
    

    cam = np.uint8(255*grayscale_cams[0, :])
    #cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255*npimg), cam_image))
    plt.imshow(np.uint8(255*npimg))
    plt.imshow(images)
    plt.xticks([])
    plt.yticks([])
    plt.title("a ="+str(miss_classified_data[0][i])+"  p="+str(miss_classified_data[1][i]),fontsize = 10)
plt.figtext(0.5, 0.01, "Left side image is actual and right one is grad cam overlayed", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.savefig("images/grad_cam.png")


