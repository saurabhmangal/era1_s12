from __future__ import print_function
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl

# +
import numpy as np
import random
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
import math
from typing import NoReturn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# -

import warnings
warnings.filterwarnings('ignore')


# +
class CIFAR10Dataset(Dataset):
    def __init__(self, dataset: CIFAR10, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.means = (0.49139968, 0.48215827 ,0.44653124)
        self.stds = (0.24703233, 0.24348505, 0.26158768)
        
    def train_transforms(self):
        return A.Compose([
            A.Normalize(mean=self.means, std=self.stds, always_apply=True),
            A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.HorizontalFlip(),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=self.means),
            ToTensorV2(),
        ])

    def test_transforms(self):
        return A.Compose([
            A.Normalize(mean=self.means, std=self.stds, always_apply=True),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = CIFAR10(root='./data', train=True, download=True, transform=None)
            self.cifar_train = CIFAR10Dataset(dataset=train_dataset, transform=self.train_transforms())
            
            val_dataset = CIFAR10(root='./data', train=False, download=True, transform=None)
            self.cifar_val = CIFAR10Dataset(dataset=val_dataset, transform=self.test_transforms())

    def train_loader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_loader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=4)


# -

def learning_r_finder(m, optimizer, criterion, device, train_loader, n_iters=200, end_lr=10):
    lr_finder = LRFinder(m, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=n_iters, step_mode="exp")
    fig = lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()


# +
def OneCycleLR_policy(optimizer, train_loader, EPOCHS, peak_value=5.0, div_factor=100, final_div_factor=100,max_lr=1.59E-03):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=peak_value/EPOCHS,
        div_factor=div_factor,
        three_phase=False,
        final_div_factor=final_div_factor,
        anneal_strategy='linear',
    )
    return scheduler


def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()

    # List to store misclassified Images
    misclassified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:

            # Migrate the data to the device
            data, target = data.to(device), target.to(device)

            # Extract single image, label from the batch
            for image, label in zip(data, target):

                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)

                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)

                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def display_mis_images(misclassified_data, n_images,classes):
    random_images = range(0, len(misclassified_data))
    random_selects = random.sample(random_images, n_images)

    fig_miss_class = plt.figure(figsize=(10, 10))
    
    count = 0
    for i in random_selects:
        plt.subplot(4, int(n_images/2), count+1)
        # Use the image tensor for plotting
        plt.imshow(misclassified_data[i][0].cpu().numpy().squeeze().transpose(1, 2, 0))
        # Use the label or classification data for the title
        plt.title(r"Correct: " + classes[misclassified_data[i][1].item()] + '\n' + 'Output: ' + classes[misclassified_data[i][2].item()])
        plt.xticks([])
        plt.yticks([])
        
        count += 1

    return fig_miss_class



# -------------------- GradCam --------------------
def display_gradcam_output(data: list,
                           classes,
                           inv_normalize,
                           model: 'DL Model',
                           target_layers,
                           targets=None,
                           number_of_samples: int = 10,
                           transparency: float = 0.60):
    """
    Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """
    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    # Create an object for GradCam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Get back the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

        # Display the images on the plot
        plt.imshow(visualization)
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])
