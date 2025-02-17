import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, sys, datetime, glob
from time import perf_counter
from omegaconf import OmegaConf
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from datasets import Dataset
from ldm.data.util import VirtualBatchWrapper
from ldm.util import instantiate_from_config
from ldm.privacy.myopacus import MyDPLightningDataModule
import torch.optim as optim
from callbacks.cuda import CUDACallback                         # noqa: F401
from callbacks.image_logger import ImageLogger                  # noqa: F401
from callbacks.setup import SetupCallback                       # noqa: F401
from ldm.data.util import DataModuleFromConfig, WrappedDataset  # noqa: F401
from torch.utils.data import DataLoader, Dataset
        
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def CNN_downstream(sample_path,epoch):  
    # create a complete CNN
    net = Net()
    class DatasetWrapper(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
    
        def __len__(self):
            return len(self.dataset)
    
        def __getitem__(self, i):
            item = self.dataset.__getitem__(i)
            label = torch.as_tensor(item["class_label"])
            item = torch.as_tensor(item["image"])
            item = item.permute(2, 0, 1)
            
            return item,label
    data_configs = [OmegaConf.load('configs/latent-diffusion/mnist32-conditional.yaml')][0]
    dataset_config = data_configs.data
    dataset = instantiate_from_config(dataset_config)
    dataset.prepare_data()
    dataset.setup()
    
    batch_size = 64
    trainset = DatasetWrapper(dataset.datasets["train"])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = DatasetWrapper(dataset.datasets["validation"])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    my_data = torch.load(sample_path)
    new_labels = []
    for i in range (10):
        for j in range (5000):
            new_labels.append(i)
    class DatasetWrapper(Dataset):
        def __init__(self, new_dataset):
            self.images = new_dataset
            self.labels = new_labels
    
        def __len__(self):
            return self.images.shape[0]
    
        def __getitem__(self, i):
            image = self.images[i]
            my_label = self.labels[i]
            assert image.shape[0] == 3 and image.shape[1] == image.shape[2], \
                   f"Samples not in CxHxW format, instead got {image.shape}"
            image = image.clamp(min=-1, max=1)
            return image,my_label
    mydataset = DatasetWrapper(my_data)
    dataloader = DataLoader(dataset=mydataset, batch_size=64,shuffle=True)
    
    for epoch in range(epoch):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs.cpu())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    classes = ["0","1","2","3","4","5","6","7","8","9"]
    
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def main(args):
    CNN_downstream(args.sample_path,args.epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sample_path", type=str, required=True, help="Path to the generated sample")
    parser.add_argument("--epoch", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()
    main(args)
    