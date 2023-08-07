import matplotlib.pyplot as plt


def plot_losses(train_losses, train_acc, test_losses, test_acc):
    t = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    return fig


def plot_images (batch_data, batch_label, total_number_of_images, grid_row, type_of_data ):
    fig = plt.figure()
    for i in range(total_number_of_images):
        plt.subplot(grid_row,int(total_number_of_images/grid_row),i+1)
        plt.tight_layout()
        if (type_of_data =='CIFAR10'):
            plt.imshow(batch_data[i].permute(1,2,0))
        
        else:
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        
        plt.title(batch_label[i])
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        #plt.close()
    return fig

