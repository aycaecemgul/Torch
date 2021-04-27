import torch
import torch.autograd #PyTorch’s automatic differentiation engine that powers neural network training.
import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
prediction = model(data) # forward pass/prediction function
#calculation of loss
loss = (prediction - labels).sum()
loss.backward() # backward pass/backpropagate. Autograd then calculates and stores
# the gradients for each model parameter in the parameter’s .grad attribute.

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9) #optimizer

optim.step() #gradient descent
#optimizer adjusts each parameter by its gradient stored in .grad.

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
#requires_grad=True signals to autograd that every operation on them should be tracked.

Q = 3*a**3 - b**2 #creating another tensor from a and b

"""
assume a and b to be parameters of an NN, and Q to be the error.
In NN training, we want gradients of the error w.r.t. parameters. (örneğin
türevlerinin alınması)
When we call .backward() on Q, autograd calculates these gradients
and stores them in the respective tensors’ .grad attribute.
We need to explicitly pass a gradient argument in Q.backward() because it is 
a vector.gradient is a tensor of the same shape as Q, and it represents 
the gradient of Q 
We need to explicitly pass a gradient argument in Q.backward() because it is a vector. 
gradient is a tensor of the same shape as Q, and it represents the gradient of Q 

"""
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

"""
Exclusion from the DAG
torch.autograd tracks operations on 
all tensors which have their requires_grad flag set to True. 
For tensors that don’t require gradients, setting this attribute to 
False excludes it from the gradient computation DAG.

The output tensor of an operation will require gradients even if only a 
single input tensor has requires_grad=True.

In a NN, parameters that don’t compute gradients are usually called frozen
 parameters. It is useful to “freeze” part of your model if you know in advance 
 that you won’t need the gradients of those parameters 
(this offers some performance benefits by reducing autograd computations).
Another common usecase where exclusion from the DAG is important is for finetuning 
a pretrained network. In finetuning, we freeze most of the model and typically 
only modify the classifier layers to make predictions on new labels. 
Let’s walk through a small example to demonstrate this. As before, we load a 
pretrained resnet18 model, and freeze all the parameters.
"""
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

"""
Let’s say we want to finetune the model on a new dataset with 10 labels. 
In resnet, the classifier is the last linear layer model.fc. We can simply 
replace it  with a new linear layer (unfrozen by default) that acts as 
our classifier.
"""
model.fc = nn.Linear(512, 10) #Now all parameters in the model, except the parameters of model.fc, are frozen
#The only parameters that compute gradients are the weights and bias of model.fc.

# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)







