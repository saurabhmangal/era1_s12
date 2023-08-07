**This is the submission for assigment number 11 of ERA V1 course.**<br> 

**Problem Statement**<br> 
The Task given was to use CIFAR 10 data and uses resnet model and monitor the accuracy of the model for 20 EPOCHS. Further, show 10 GRADCAM images. The code should be wriiten in a modular way<br> 

**File Structure**<br> 
-resnet.py           - has the resnet model copied from the github library suggested in the assignment<br>
-era_s11_cifar.ipynb  - the main .ipynb file<br> 
-Colab_notebook.ipynb - Google Colab file to executed<br> 
-main.py              - main file in .py mode<br> 
-plots.py             - contains function to plot<br> 
-utils.py             - contains different functions which are<br>      Cifar10SearchDataset,<br> 
create_train_data_loader,<br> 
create_test_data_loader,<br> 
train_transforms,<br> 
test_transform,<br>  
imshow,<br>    
display_mis_images,<br>               learning_r_finder,<br>     OneCycleLR_policy<br> 
-calc_loss_accuracy.py - function to train and test loss and accuracy while model training<br> 
-images:<br> 
  -Accuracy & Loss.jpg        -- Plot of train and test accuracy and loss with respect to epochs<br> 
  -miss_classified_image.jpg  -- sample mis classified images. <br> 
  -test_dataset.jpg           -- sample test dataset<br> 
  -train_dataset.jpg          -- sample train dataset after tranformation<br> 
  -grad_cam.png               -- image gallery for grad cam images<br> 
        
Following are the sample images of train dataset:<br> 
<img src="https://github.com/saurabhmangal/era1_s11/blob/main/images/train_dataset.jpg" alt="alt text" width="600px">

Following are the sample imagese of the test dataset:<br> 
<img src="https://github.com/saurabhmangal/era1_s11/blob/main/images/test_dataset.jpg" alt="alt text" width="600px">

**Epoch Results:**<br>
EPOCH: 0<br>
Loss=1.3281242847442627 LR =0.00038138098159509206 Batch_id=97 Accuracy=38.49: 100% 98/98 [00:11<00:00,  8.68it/s]<br>
Test set: Average loss: 0.0029, Accuracy: 4864/10000 (48.64%)<br>
EPOCH: 1<br>
Loss=1.0123437643051147 LR =0.000744461963190184 Batch_id=97 Accuracy=55.77: 100% 98/98 [00:10<00:00,  9.35it/s]  <br>
Test set: Average loss: 0.0028, Accuracy: 5417/10000 (54.17%)<br>
EPOCH: 2<br>
Loss=0.9629431366920471 LR =0.001107542944785276 Batch_id=97 Accuracy=63.06: 100% 98/98 [00:10<00:00,  9.16it/s] <br>
Test set: Average loss: 0.0038, Accuracy: 4917/10000 (49.17%)<br>
EPOCH: 3<br>
Loss=0.8593705296516418 LR =0.0014706239263803681 Batch_id=97 Accuracy=67.80: 100% 98/98 [00:10<00:00,  9.28it/s]<br>
Test set: Average loss: 0.0041, Accuracy: 4862/10000 (48.62%)<br>
EPOCH: 4<br>
Loss=0.8024423718452454 LR =0.0018287552265306122 Batch_id=97 Accuracy=70.74: 100% 98/98 [00:10<00:00,  9.11it/s]<br>
Test set: Average loss: 0.0019, Accuracy: 6817/10000 (68.17%)<br>
EPOCH: 5<br>
Loss=0.8186142444610596 LR =0.0017067674265306124 Batch_id=97 Accuracy=72.40: 100% 98/98 [00:10<00:00,  9.23it/s]<br>
Test set: Average loss: 0.0021, Accuracy: 6454/10000 (64.54%)<br>
EPOCH: 6<br>
Loss=0.6541144251823425 LR =0.0015847796265306122 Batch_id=97 Accuracy=75.35: 100% 98/98 [00:10<00:00,  9.30it/s]<br>
Test set: Average loss: 0.0014, Accuracy: 7536/10000 (75.36%)<br>
EPOCH: 7<br>
Loss=0.6365342140197754 LR =0.0014627918265306124 Batch_id=97 Accuracy=76.73: 100% 98/98 [00:10<00:00,  9.17it/s]<br>
Test set: Average loss: 0.0014, Accuracy: 7536/10000 (75.36%)<br>
EPOCH: 8<br>
Loss=0.7041124105453491 LR =0.0013408040265306123 Batch_id=97 Accuracy=78.19: 100% 98/98 [00:10<00:00,  9.19it/s]<br>
Test set: Average loss: 0.0013, Accuracy: 7792/10000 (77.92%)<br>
EPOCH: 9<br>
Loss=0.5653582215309143 LR =0.0012188162265306121 Batch_id=97 Accuracy=79.49: 100% 98/98 [00:10<00:00,  9.14it/s]<br>
Test set: Average loss: 0.0014, Accuracy: 7683/10000 (76.83%)<br>
EPOCH: 10<br>
Loss=0.6183575391769409 LR =0.0010968284265306123 Batch_id=97 Accuracy=80.20: 100% 98/98 [00:10<00:00,  9.17it/s] <br>
Test set: Average loss: 0.0011, Accuracy: 8190/10000 (81.90%)<br>
EPOCH: 11<br>
Loss=0.5111405849456787 LR =0.0009748406265306123 Batch_id=97 Accuracy=81.33: 100% 98/98 [00:10<00:00,  9.38it/s] <br>
Test set: Average loss: 0.0010, Accuracy: 8298/10000 (82.98%)<br>
EPOCH: 12<br>
Loss=0.45070329308509827 LR =0.0008528528265306123 Batch_id=97 Accuracy=82.83: 100% 98/98 [00:10<00:00,  9.28it/s]<br>
Test set: Average loss: 0.0009, Accuracy: 8532/10000 (85.32%)<br>
EPOCH: 13<br>
Loss=0.4618381857872009 LR =0.0007308650265306122 Batch_id=97 Accuracy=84.02: 100% 98/98 [00:10<00:00,  9.28it/s] <br>
Test set: Average loss: 0.0011, Accuracy: 8274/10000 (82.74%)<br>
EPOCH: 14<br>
Loss=0.4043424129486084 LR =0.0006088772265306123 Batch_id=97 Accuracy=85.02: 100% 98/98 [00:10<00:00,  9.39it/s] <br>
Test set: Average loss: 0.0009, Accuracy: 8428/10000 (84.28%)<br>
EPOCH: 15<br>
Loss=0.4370374083518982 LR =0.00048688942653061216 Batch_id=97 Accuracy=85.74: 100% 98/98 [00:10<00:00,  9.19it/s]<br>
Test set: Average loss: 0.0008, Accuracy: 8712/10000 (87.12%)<br>
EPOCH: 16<br>
Loss=0.3477949798107147 LR =0.00036490162653061227 Batch_id=97 Accuracy=87.35: 100% 98/98 [00:10<00:00,  9.16it/s] <br>
Test set: Average loss: 0.0006, Accuracy: 8978/10000 (89.78%)<br>
EPOCH: 17<br>
Loss=0.3061493933200836 LR =0.00024291382653061238 Batch_id=97 Accuracy=88.92: 100% 98/98 [00:10<00:00,  9.15it/s] <br>
Test set: Average loss: 0.0006, Accuracy: 8987/10000 (89.87%)<br>
EPOCH: 18<br>
Loss=0.34783899784088135 LR =0.00012092602653061228 Batch_id=97 Accuracy=89.88: 100% 98/98 [00:10<00:00,  9.40it/s]<br>
Test set: Average loss: 0.0005, Accuracy: 9116/10000 (91.16%)<br>
EPOCH: 19<br>
Loss=0.30003660917282104 LR =-1.061773469387614e-06 Batch_id=97 Accuracy=91.10: 100% 98/98 [00:10<00:00,  9.40it/s]<br>
Test set: Average loss: 0.0005, Accuracy: 9193/10000 (91.93%)<br>

Following are the plot of train and test losses and accuracies:<br> 
<img src="https://github.com/saurabhmangal/era1_s11/blob/main/images/Accuracy%20%26%20Loss.jpg" alt="alt text" width="600px"><br> 

Some of the sample misclassified images are as follows:<br> 
<img src="https://github.com/saurabhmangal/era1_s11/blob/main/images/mis_classified_image.jpg" alt="alt text" width="600px"><br> 

Some of the sample misclassified grad cam images are as follows:<br> 
<img src="https://github.com/saurabhmangal/era1_s11/blob/main/images/grad_cam.png" alt="alt text" width="600px"><br>
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
