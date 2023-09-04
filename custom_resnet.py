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
        data_module.setup('fit')
        self.train_loader_len = len(data_module.train_loader()) #2*782 #
        self.EPOCHS = 24 #EPOCHS
        self.peak_value = 5.0  #peak_value
        self.div_factor = 100 #div_factor
        self.final_div_factor = 100 #final_div_factor
        self.validation_step_outputs = []
        self.training_step_outputs = []

        
        
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
        train_inputs, train_labels = batch
        train_preds = self.forward(train_inputs)

        train_loss = nn.CrossEntropyLoss()(train_preds, train_labels)
        
                
        train_acc  = (torch.argmax(train_preds, dim=1) == train_labels).float().mean()
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True,prog_bar=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, logger=True)
        self.training_step_outputs.append(train_acc)

        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True,prog_bar=True)

        return train_loss
    
    
    def on_train_epoch_end(self):
        # do something with all training_step outputs, for example:
        train_epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("training_acc_epoch_mean", train_epoch_mean, on_step=False, on_epoch=True, logger=True,prog_bar=True)
        print(f"\nEpoch: {self.current_epoch}"," Training Acc Epoch Mean:", train_epoch_mean.item())
        print ("*"*50)
        self.training_step_outputs.clear()
        
        (f"Current Epoch: {self.current_epoch}")
    
    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels = batch
        val_preds = self.forward(val_inputs)
        
        val_loss  = nn.CrossEntropyLoss()(val_preds, val_labels)
        val_acc   = (torch.argmax(val_preds, dim=1) == val_labels).float().mean()
        self.log('val_loss', val_loss, on_step=True, on_epoch=False, logger=True,prog_bar=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=False, logger=True)
        self.validation_step_outputs.append(val_acc)
        return val_acc

    def on_validation_epoch_end(self):
        all_acc = torch.stack(self.validation_step_outputs)
        avg_acc = all_acc.mean()
        self.log('val_acc_epoch_mean', avg_acc, on_epoch=True, logger=True,prog_bar=True)
        print (f"\nEpoch: {self.current_epoch}"," Validation Acc Epoch Mean:",avg_acc.item()),
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
            'interval': 'step',
            'frequency':1
        }
        return [optimizer], [scheduler]
