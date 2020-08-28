#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'


# ### Import Libraries

# In[2]:


#Torch Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR, OneCycleLR

#Utility and Plot Libs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm


# ### Pytorch Version check

# In[3]:


torch.__version__


# ### Data Transformation

# In[4]:


#Need to create a data Trandform function here. there transform objects will be use fulll to apply on data


# In[5]:


train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       ])


test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       ])


# ### Creating the Trainning and Testing data  

# In[6]:


#Here applying the transform function we hafve created.
#What is target_transform , Need to understand

train = datasets.MNIST('./', train = True,  transform = train_transforms,  target_transform = None,  download = True)
test  = datasets.MNIST('./', train = False,  transform = test_transforms,   target_transform = None,  download = True)


# In[7]:


type(train)
type(test)


# ### Creating Data Loader

# #### Cuda Checker

# In[8]:


cuda = torch.cuda.is_available()
print ('Is Cuda Available ?',cuda)


# In[9]:


SEED = 1

if cuda:
    torch.cuda.manual_seed(SEED)
else:
    torch.manual_seed(SEED)


# #### Based on the cuda availability we are defining the things

# In[10]:


dataloader_args = dict(shuffle = True,batch_size = 128,num_workers = 4, pin_memory = True) if cuda else dict(shuffle = True,batch_size = 64)


# In[11]:


dataloader_args


# In[12]:


train_loader = torch.utils.data.DataLoader(train,**dataloader_args)
test_loader = torch.utils.data.DataLoader(test,**dataloader_args)


# In[13]:


pwd


# ### Extracting Statistics
train_data = train.train_data
train_data = train.transform(train_data.numpy())print('[Train]')
print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
print(' - Tensor Shape:', train.train_data.size())
print(' - min:', torch.min(train_data))
print(' - max:', torch.max(train_data))
print(' - mean:', torch.mean(train_data))
print(' - std:', torch.std(train_data))
print(' - var:', torch.var(train_data))
# In[14]:


dataiter = iter(train_loader)
images,label = dataiter.next()


# In[15]:


print(images.shape)
print(label.shape)


# In[16]:


plt.imshow(images[0].numpy().squeeze(),cmap = 'gray_r')


# ### Plotting all the image

# In[17]:


figure = plt.figure()
num_of_images = 63
for index in range(1, num_of_images + 1):
    plt.subplot(10, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


# ## Model Architecture

# ## GBN

# In[ ]:





# In[ ]:





# In[18]:


class GhostBatchNorm(nn.BatchNorm2d):
    """
    From : https://github.com/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb

    Batch norm seems to work best with batch size of around 32. The reasons presumably have to do 
    with noise in the batch statistics and specifically a balance between a beneficial regularising effect 
    at intermediate batch sizes and an excess of noise at small batches.
    
    Our batches are of size 512 and we can't afford to reduce them without taking a serious hit on training times, 
    but we can apply batch norm separately to subsets of a training batch. This technique, known as 'ghost' batch 
    norm, is usually used in a distributed setting but is just as useful when using large batches on a single node. 
    It isn't supported directly in PyTorch but we can roll our own easily enough.
    """
    def __init__(self, num_features, num_splits, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super(GhostBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias        
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
        return super(GhostBatchNorm, self).train(mode)
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C*self.num_splits, H, W), self.running_mean, self.running_var, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W) 
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features], 
                self.weight, self.bias, False, self.momentum, self.eps)

        


# In[19]:



class Net(nn.Module):
    def __init__(self,ghost = False):
        super(Net, self).__init__()
        self.dropout_value = 0.05
        # Input Block
        print('GHOST BHAI is ',ghost)
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10) if ghost is False else GhostBatchNorm(10, num_splits=4, weight=False),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # input_size = 28 output_size = 26 receptive_field = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10) if ghost is False else GhostBatchNorm(10, num_splits=4, weight=False),
            nn.Dropout(self.dropout_value),
            nn.ReLU(),
        ) # input_size = 26 output_size = 24 receptive_field = 5
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10) if ghost is False else GhostBatchNorm(10, num_splits=64//32, weight=False),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # input_size = 24 output_size = 22 receptive_field = 7       
        
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10) if ghost is False else GhostBatchNorm(10, num_splits=64//32, weight=False),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # input_size = 22 output_size = 20 receptive_field = 9        
        
        
        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 20 output_size = 10 receptive_field = 18

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10) if ghost is False else GhostBatchNorm(10, num_splits=64//32, weight=False),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # input_size = 10 output_size = 8 receptive_field = 20
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16) if ghost is False else GhostBatchNorm(16, num_splits=64//32, weight=False),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # input_size = 8 output_size = 6 receptive_field = 22
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16) if ghost is False else GhostBatchNorm(16, num_splits=64//32, weight=False),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # input_size = 6 output_size = 4 receptive_field = 24
        
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 1 output_size = 1  receptive_field = 24
        

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)  
        x = self.convblock8(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)


# ### Model Param

# In[20]:


from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(1, 28, 28))


# In[21]:


#L1 Loss Calculator
def L1_Loss_calc(model, factor=0.0005):
    l1_crit = nn.L1Loss(size_average=False)
    reg_loss = 0
    for param in model.parameters():
        #zero_vector = torch.rand_like(param)*0
        zero_vector = torch.zeros_like(param)
        reg_loss += l1_crit(param,zero_vector)
    return factor * reg_loss


# In[22]:


from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch,scheduler,l1_check =False):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        
        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        #adding L1 loss function call 

        if( l1_check == True ):
            regloss = L1_Loss_calc(model, 0.0005)
            regloss /= len(data) # by batch size
            loss += regloss

        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

#         pbar.set_description(desc= f'Loss={loss.item():0.6f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        pbar.set_description(desc= f'Loss={train_loss/(batch_idx+1):0.6f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_loss /= len(train_loader)
    
    acc = np.round(100. * correct/len(train_loader.dataset),2) #processed # 
    scheduler.step()
    return acc, np.round(train_loss,5)


# In[23]:



def test(model, device, test_loader,l1_check = False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if( l1_check == True ):
            regloss = L1_Loss_calc(model, 0.0005)
            regloss /= len(data) # by batch size
            test_loss += regloss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100 * correct / len(test_loader.dataset)))
    acc = 100 * correct /len(test_loader.dataset)
    acc = np.round(acc,2)
    
    return acc, test_loss
    


# In[24]:


EPOCHS = 15
learning_rate = 0.04


# ### Temp Store

# ### Lets add L2 

# In[25]:


len(train_loader)


# In[26]:


def model_building(typ, model, device,train_loader, test_loader,l1_check = False, l2_val = 0, EPOCHS = 3):
    
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.2, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    print ('Model with ', typ)
    for epoch in range(1,EPOCHS):
        print("EPOCH:", epoch,'LR : ',scheduler.get_lr())
#         scheduler.step()
        acc,loss =  train(model, device, train_loader, optimizer, epoch,scheduler,l1_check= True)
        train_acc.append(acc)
        train_losses.append(loss)
        
        acc,loss =  test(model, device, test_loader,l1_check= True)
        test_acc.append(acc)
        test_losses.append(loss)
        
    return train_acc,train_losses,test_acc,test_losses


# ## Missclassifications
# 

# In[27]:


def miss_classification(typ, model, device, testloader, num_of_images = 25, save_filename="misclassified"):
    model.eval()
    misclassified_cnt = 0
    fig = plt.figure(figsize=(12,12))
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
        pred_marker = pred.eq(target.view_as(pred))   
        wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
#         print ('Missclassification on ', typ)
        for idx in wrong_idx:
            index = idx[0].item()
            title = "Actul:{}, Pred:{}".format(target[index].item(), pred[index][0].item())
            #print(title)
            ax = fig.add_subplot(5, 5, misclassified_cnt+1, xticks=[], yticks=[]) #ax.axis('off')
            ax.set_title(title)
            plt.imshow(data[index].cpu().numpy().squeeze(), cmap='gray_r')
            misclassified_cnt += 1
            if(misclassified_cnt==num_of_images):
                break
        if(misclassified_cnt==num_of_images):
            break
    
    fig.savefig("{}.png".format(save_filename)) 
    return


# In[28]:


def plot_acc_loss(typ, train_acc,train_losses,test_acc,test_losses):
    fig, axs = plt.subplots(1,2,figsize=(20,5))
    print ('Accuracy model on  ', typ)
    plt.autoscale()
    axs[0].plot(train_acc,color = 'red')
    axs[0].plot(test_acc,color = 'green')
    title = 'Training/testing accuracy'
    axs[0].set_title(title)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train','test'], loc='best')
       
    axs[1].plot(train_losses,color = 'red')
    axs[1].plot(test_losses,color = 'green')
    title = 'Training/Testing Loss'
    axs[1].set_title(title)
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train','test'], loc='best')
    
    plt.show()


# In[29]:


def modelcalling(models):

    for typ in models:
        if typ == 'L1+BN':
            l1_check = True
            l2_val = 0
            ghost = False
        elif typ == 'L2+BN':
            l1_check = False
            l2_val = 0.03
            ghost = False
        elif typ == 'L1 & L2 + BN':
            l1_check = True
            l2_val = 0.03
            ghost = False
        elif typ == 'GBN':
            l1_check = False
            l2_val = 0
            ghost = True
        elif typ == 'L1 & L2 + GBN':
            l1_check = True
            l2_val = 0.03
            ghost = True
            
        model =  Net(ghost).to(device)
        train_acc,train_losses,test_acc,test_losses = model_building(typ, model, device,train_loader, test_loader,l1_check = False,l2_val = 0,EPOCHS = 20)
        plot_acc_loss(typ, train_acc,train_losses,test_acc,test_losses)
        miss_classification(typ, model, device, testloader = test_loader, num_of_images = 25, save_filename="misclassified")
        
        modelparams = {'train_acc':train_acc,'test_acc':test_acc,'train_losses':train_losses,'test_losses':test_losses}
        file = pd.DataFrame(modelparams)
        file.to_csv(typ + 'param'+'.csv')

models = ['L1+BN','L2+BN','L1 & L2 + BN','GBN','L1 & L2 + GBN']
# In[30]:


models = ['L1+BN','L2+BN','L1 & L2 + BN','GBN','L1 & L2 + GBN']
modelcalling(models)


# In[ ]:





# In[ ]:





# In[ ]:




