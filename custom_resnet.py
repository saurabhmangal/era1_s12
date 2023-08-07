from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.nn as nn
import torch.optim as optim
from utils import CIFAR10Dataset, CIFAR10DataModule

# +
class ResNetLightningModel(pl.LightningModule):
    def __init__(self):
        super(ResNetLightningModel, self).__init__()
        
        self.max_lr = 1.59E-03
        data_module = CIFAR10DataModule()      
        self.train_loader_len = 782 #len(data_module.train_loader())
        self.EPOCHS = 20 #EPOCHS
        self.peak_value = 5.0  #peak_value
        self.div_factor = 100 #div_factor
        self.final_div_factor = 100 #final_div_factor
                
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.valid_acc = Accuracy(task="multiclass", num_classes=10)
        self.validation_step_outputs = []

        # PrepLayer
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # output_size = 32/3/1

        # Layer1        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            
        )
        
        self.resnet1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        

        # Layer2
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),        
        )
        
        # Layer3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride = 1, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
        )

        self.resnet2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        
        self.maxpool2   = nn.MaxPool2d(kernel_size = 4, stride = 2)
                
        self.fc_layer   = nn.Linear(512, 10)

        

    def forward(self, x):
        x  = self.convblock1(x)
        
        x  = self.convblock2(x)
        r1 = self.resnet1(x)
        x  = x + r1
        
        x  = self.convblock3(x)
        
        x  = self.convblock4(x)
        r2 = self.resnet2(x)
        x  = x + r2
        
        x  = self.maxpool2(x) 
        
        x = x.view(x.size(0), -1)
        x  = self.fc_layer(x) 
        
        x = F.softmax(x, dim=-1)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = self.train_acc(torch.argmax(outputs, dim=1), labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        
            # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print (lr)
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = self.valid_acc(outputs.argmax(dim=1), labels)
        self.log('val_loss_step', loss, on_step=True, on_epoch=False, logger=True)
        self.log('val_acc_step', acc, on_step=True, on_epoch=False, logger=True)
        self.validation_step_outputs.append(acc)
        return acc

    def on_validation_epoch_end(self):
        all_acc = torch.stack(self.validation_step_outputs)
        avg_acc = all_acc.mean()
        self.log('val_acc', avg_acc, on_epoch=True, logger=True)
        self.validation_step_outputs.clear()     
        

#     def configure_optimizers(self):
#         # Define optimizer
#         optimizer = optim.Adam(self.parameters(), lr=0.001)
#         return optimizer
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        scheduler = {
            'scheduler': OneCycleLR(optimizer,
                                    max_lr= self.max_lr,
                                    steps_per_epoch=self.train_loader_len,
                                    epochs=self.EPOCHS,
                                    pct_start=self.peak_value/self.EPOCHS,
                                    div_factor=self.div_factor,
                                    three_phase=False,
                                    final_div_factor=self.final_div_factor,
                                    anneal_strategy='linear'),
            'interval': 'step'
        }
        return [optimizer], [scheduler]
# -


