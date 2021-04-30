import torch
from torch import nn
import matplotlib.pyplot as plt

# PyTorch has two primitives to work with data:
# DataLoader wraps an iterable around the Dataset.
from torch.utils.data import DataLoader

# Dataset stores the samples and their corresponding labels.
from torch.utils.data import Dataset

from torchvision import datasets  # this module contains Dataset objects for many real-world vision data
from torchvision.transforms import ToTensor, Lambda, Compose

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",  # root is the path where the train/test data is stored
    train=True,  # specifies training or test dataset
    download=True,  # downloads the data from the internet if it’s not available at root
    transform=ToTensor(),  # and  "target_transform" specify the feature and label transformations
)

# Download test data from open datasets.
# 60,000 training examples and 10,000 test examples
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

"""We pass the Dataset as an argument to DataLoader. 
This wraps an iterable over our dataset, and supports automatic batching, 
sampling, shuffling and multiprocess data loading. Here we define a batch 
size of 64, i.e. each element in the dataloader 
iterable will return a batch of 64 features and labels."""

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# You  can index Datasets manually like a list: training_data[index]
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#CREATING THE MODEL

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#To define a neural network
#create a class that inherits from nn.Module.
# Define model
class NeuralNetwork(nn.Module):
    #define the layers of the network in the __init__ function
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
#to specify how data will pass through the network in the forward function
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)