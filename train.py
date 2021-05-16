import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import warp, AffineTransform
from sklearn.model_selection import  KFold



def transform_mask(path):
    """Takes RGB image of mask (annotation image)
       and maps it tho the 2D array with annotation
       values of pixels 0,1,2.

    Args:
        path (string): Path to the image

    Returns:
        Numpy 2d array: Transformed image
    """

    image = mpimg.imread(path)
    blue_mask = image[:,:,2]
    mask =  np.where(blue_mask>0, blue_mask*2, image[:,:,1])
    mask = mask[:336,:336]
    return mask


def seed_all(seed):
    """ Seeds pytorch, numpy and random libraries
        to possibly produce reproducible results.

    Args:
        seed (int): seed value
    """
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class img_dataset(Dataset):
    """Dataset class to handle data in pytorch.
        It needs to specify 2 methods-
        1) len()
        2) getitem() 

    Args:
        Dataset : Parent class from torch.utils.data 
    """
    def __init__(self, filelist, train_paths, phase):
        """Loads the images, labels and provides data augmentation, if the data
           are marked as 'train'

        Args:
            filelist (list): list of labels for which data should be prepared
            train_paths (list): paths to the folders with labels
            phase (string): 'train' or 'valid', augmentations are applied in train phase
        """
        
        self.images = []
        self.masks = []       
 
        for file in filelist:
            filename_train = file.split('_')[0]+'_'+file.split('_')[1]+'.npz'
            filename_mask = file.split('_')[0]+'_'+file.split('_')[1]+'_labels.png'

            for path in train_paths:
                if filename_mask in os.listdir(path):
                    file_train = path.split('labels')[0] + 'data' +  path.split('labels')[1] + filename_train
                    file_mask= path + filename_mask 
                    break
 

            # Load npz image    
            image = np.load(file_train)["arr_0"]
            
            # Takes only its r,g,b and nir bands and crops to 336 pixels h,w
            crop_pix = 336
            red = image[:crop_pix,:crop_pix,3]
            green = image[:crop_pix,:crop_pix,2]
            blue = image[:crop_pix,:crop_pix,1]
            nir = image[:crop_pix,:crop_pix,4]
            
            image = np.stack((red, green, blue, nir), axis = 2)
            
            
            #NORMALIZE
            image = image/255
            
            if phase == 'train':

                #Pytorch accepts images in shape(ch, h, w), not (h, w, ch)
                #therefore transposition of images is necessary

                self.images.append(image.transpose(2,0,1))
                self.masks.append(transform_mask(file_mask))
                
                #UP-DOWN flip augmentation
                self.images.append(np.flipud(image).transpose(2,0,1).copy())
                self.masks.append(np.flipud(transform_mask(file_mask)).copy())
                
                #LEFT-RIGHT flip augmentation
                self.images.append(np.fliplr(image).transpose(2,0,1).copy())
                self.masks.append(np.fliplr(transform_mask(file_mask)).copy())
                
                # Affine shift of images
                transform = AffineTransform(translation = (-200,80))
                        
                self.images.append(warp(image, transform, mode = 'wrap').transpose(2,0,1).copy())
                self.masks.append(warp(transform_mask(file_mask), transform, mode = 'wrap').copy())
                
                transform = AffineTransform(translation = (20,-300))
                        
                self.images.append(warp(image, transform, mode = 'wrap').transpose(2,0,1).copy())
                self.masks.append(warp(transform_mask(file_mask), transform, mode = 'wrap').copy())
 
            else:
                self.images.append(image.transpose(2,0,1))
                self.masks.append(transform_mask(file_mask))
                
                
        #Random shuffling of data 
        c = list(zip(self.images, self.masks))
        random.shuffle(c)
        self.spectograms, self.label_array = zip(*c)

    
    def __getitem__(self, index):
        return self.images[index], self.masks[index]
        
    
    def __len__(self):
        return len(self.images)



class UNET(nn.Module):
    """Create class of UNET convolution encoder-decoder model with depth of 3

       Huge inspiration was found here:
       https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-
       with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
    
        Same results were achieved with Pytorch pretrained SMP with resnet18
        as backbone.

    Args:
        nn.Module : Parent class 
    """

    def __init__(self, in_channels, out_channels):
        """Initialization of blocks

        Args:
            in_channels (int): Number of input channels (4 in out case)
                                R,G,B,NIR
            out_channels (int): Number of input channels (clsses) (3 in out case)
                                R,G,B
        """
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(128, 32, 3, 1)
        self.upconv1 = self.expand_block(64, out_channels, 3, 1)

    def __call__(self, x):
        """U net workflow
            1) 2x convolution + max pooling - downsamples image by 2
            2) 2x convolution + max pooling - downsamples image by 2
            3) 2x convolution + max pooling - downsamples image by 2

            4) 2x convolution + upsampling conv - upsamples image by 2
            5) - concats previous output and output from 2)
               - 2x convolution + upsampling conv - upsamples image by 2
            6) - concats previous output and output from 1)
               - 2x convolution + upsampling conv - upsamples image by 2
               This output has original HxW

        Args:
            x (Pytorch batch): input batch of images

        Returns:
            [Pytorch batch]: Predicted mask
        """

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        # upsampling part
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        """Do following operations
            2X repeat:
                1) 2D conv
                2) Batch normalization for better stability 
                3) Acctivation with RELU
    
            Max pooling in the end

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (): size of filter mask
            padding (int): whether to use padding 

        Returns:
            Torch sequential container: Sequention of operations defined above
        """

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract.double()

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        """Do following operations
            2X repeat:
                1) 2D conv
                2) Batch normalization for better stability 
                3) Acctivation with RELU
    
            Max pooling in the end

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (): size of filter mask
            padding (int): whether to use padding 

        Returns:
            Torch sequential container: Sequention of operations defined above
        """

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand.double()


def train_l(model, train_loader, valid_loader, train_data,  valid_data,
            criterion, optimizer, curr_fold, num_epochs, device, save_path):
    """
    Workflow for training of model :
    for n in num_epochs:
        1)predict outputs [outputs = model(inputs)]
        2)computes loss function [criterion(outputs, labels)]
        3)computes gradient of loss fn (backprop) [loss.backward()]
        4)updates weights [optimizer.step()]

    Note: For real saving of the model, uncomment part in the end of
          this function

    Args:
        model (torch nn.Module): model itself
        train_loader : torch train Dataloader
        valid_loader : torch valid Dataloader
        train_data : torch train Dataset
        valid_data : torch valid Dataset
        criterion : torch loss function
        optimizer : torch optimizer
        curr_fold (int): number of current CV fold
        num_epochs (int): How long should be training 
        device : GPU or CPU
        save_path (string) : where to store model

    Returns:
        lists: list with accuracies and losses
    """

    # TRAIN PHASE
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    
    for epoch in range(num_epochs):

        model.train() 
        #Tells the model to use train mode
        #Dropout layer behaves differently for train/eval phases
        
        actual_loss = 0
        num_corrects = 0  
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad() 
            # otherwise by calling loss.backward() gradient of parameters would be summed
            
            outputs = model(inputs) 
            
            loss = criterion(outputs, labels)     
            # creates graph of parameters, is connected to model throught outputs
        
            loss.backward()  #computes gradient of loss with respect to the parameters
            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step() #updates models parameters
                       
            
            actual_loss += loss.item() * inputs.size(0) #sum of losses for given batch
            num_corrects += torch.sum(outputs.argmax(dim=1) == labels).item()  #number of correct

        train_loss.append(actual_loss / len(train_data)) 
        train_acc.append(num_corrects / (len(train_data)*336*336))
       

        #VALIDATION_PHASE 
        with torch.no_grad(): 
            #DROPOUT is not applied
            model.eval() 

            valid_actual_loss = 0
            valid_num_corrects = 0
        
            for inputs, labels in valid_loader:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
            
                outputs = model(inputs)               
                loss = criterion(outputs, labels)
            
                valid_actual_loss += loss.item() * inputs.size(0)
                valid_num_corrects += torch.sum(outputs.argmax(dim=1) == labels).item()
            
            valid_loss.append(valid_actual_loss / len(valid_data))
            valid_acc.append(valid_num_corrects / (len(valid_data)*336*336))
   
            #Save the model if it have 3 times in a row accuracy > 90%
            '''
            if epoch > 10:
                if (np.min(valid_acc[epoch]+valid_acc[epoch-1]+valid_acc[epoch-2]) > 0.9):
                    torch.save(model, f'{save_path}model_{curr_fold}')
            '''
            
    return train_loss, valid_loss, train_acc, valid_acc

def train_CV(train_paths, save_path, device):
    """Trains Cross-validation
        N times calls train_l function

    Args:
        train_path (list): list of paths to folders with labels
        save_path (string): path to folder where models should be stored
        device : CPU or GPU
    """
    
    labeled_images = []
    for path in train_paths:
        labeled_images += [x.split('.')[0] for x in os.listdir(path)] 

    random.shuffle(labeled_images)
    
    n_splits = 4
    cross_val = KFold(n_splits= n_splits, shuffle=True, random_state = 10)
    i = 0
    
    for train_indexes, valid_indexes in cross_val.split(labeled_images):
    
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(f'Starting next CV fold')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    
        list_train = []
        for i in train_indexes:
            list_train.append(labeled_images[i])
        
        list_valid = []
        for i in valid_indexes:
            list_valid.append(labeled_images[i])
        
        train_data = img_dataset(list_train, train_paths, phase ='train')
        valid_data = img_dataset(list_valid, train_paths, phase ='valid')
    
        train_loader = DataLoader(train_data, batch_size=4, shuffle = True)
        valid_loader = DataLoader(valid_data, batch_size=4, shuffle = True)
    
        loss_fn = nn.CrossEntropyLoss()
    
        model = UNET(4,3)
        model = model.float()
        model.to(device)

        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        train_loss, valid_loss, train_acc, valid_acc = train_l(model, train_loader,
                                                                valid_loader,
                                                                train_data,
                                                                valid_data, 
                                                                loss_fn, opt, i, 120,
                                                                device,
                                                                save_path)

        fig, axs = plt.subplots(2, figsize=(14,8))
        axs[0].plot(train_loss)
        axs[0].plot(valid_loss)
        axs[0].set_title('blue = Train Loss, orange = Valid Loss')

        axs[1].plot(train_acc)
        axs[1].plot(valid_acc)
        axs[1].set_title('Blue = Train Accuracy, orange = Valid Accuracy')
        plt.show()


#if __name__ == 'main':
def train_and_save_model(train_paths, save_path):
    """Calls train_CV with given arguments

    Args:
        train_paths (list): paths to folders with labels of images
        save_path (string): where trained models should be stored
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_all(20)
    train_CV(train_paths, save_path, device)