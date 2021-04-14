import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch,torchvision
from numpy import asarray
from skimage.transform import resize
from torch.utils.data import Dataset,random_split
from torchvision.io import read_image
from torchvision import transforms,models
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL.Image import  fromarray
#DATASET

"""
A custom Dataset class must implement three functions:
 __init__, __len__, and __getitem__
"""

import os
import pandas as pd
from torchvision.io import read_image
import  cv2
from torch.utils.data import DataLoader

img_dir="C:\\Users\\aycae\\OneDrive\\Belgeler\\GitHub\\Torch\\Main\\best-artworks-of-all-time"
categories = ["Edgar_Degas","Pablo_Picasso","Vincent_van_Gogh"]
IMG_SIZE=250

#Resizing images
# for category in categories:
#     path = os.path.join(img_dir, category)
#     for img in os.listdir(path):
#         class_no = categories.index(category)
#         try:
#             filename=os.path.join(path, img)
#             img_array = asarray(Image.open(filename).convert('RGB'))
#             img_array = resize(img_array, (IMG_SIZE, IMG_SIZE))
#             plt.imsave(filename,img_array)
#
#         except Exception as e:
#             print("error")

class ImageDataset(Dataset):

    def __init__(self, img_dir, categories, transform=None, target_transform=None):
        self.X = list()
        self.y = list()
        self.img_dir = img_dir
        self.categories=categories
        self.transform = transform
        self.target_transform = target_transform

        for category in categories:
            path = os.path.join(img_dir, category)
            for img in os.listdir(path):
                class_no = categories.index(category)
                try:
                    filename = os.path.join(path, img)
                    img = Image.open(filename).convert('RGB')
                    img=np.asarray(img)
                    self.X.append(img)
                    self.y.append(class_no)
                except Exception as e:
                    print("error")


    def __len__(self):
        return len(self.y)

    #loads and returns a sample from the dataset at the given index idx.
    def __getitem__(self, idx):
        image = Image.fromarray(self.X[idx]).convert('RGB')
        label = np.array(self.y[idx]).astype('float')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {'image': image, \
                  'label': torch.from_numpy(label)}

        return sample

data_transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageDataset(img_dir,categories,transform=data_transforms)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
training_data, test_data = random_split(dataset, [train_size, test_size])


#DataLoader is an iterable
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_epochs=15

#MODEL

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


