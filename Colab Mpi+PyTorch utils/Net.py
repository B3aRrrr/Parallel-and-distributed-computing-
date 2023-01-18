import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Adagrad,RMSprop,Adadelta

import torch
import torchvision

# from utils import *
# from train_test_maker import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    def GetName(self):
      for i, j in globals().items():
        if j is self:
          return i

optimizes = [SGD,Adam,Adagrad,RMSprop,Adadelta]
list_names = ['SGD','Adam','Adagrad','RMSprop','Adadelta']
dict_Optimizers = dict(zip(list_names,optimizes))

def Optimizer(name:str):
    return dict_Optimizers[name]

def train(name, model_optimizer_dict,epoch):
    # 1
    model = model_optimizer_dict['model']
    optimizer = model_optimizer_dict['optimizer']
    train_losses = model_optimizer_dict['train_losses']
    train_counter = model_optimizer_dict['train_counter']
    test_losses = model_optimizer_dict['test_losses'] 
    test_counter = model_optimizer_dict['test_counter'] 
    #2 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), f'/content/results/{name}/model.pth')
            torch.save(optimizer.state_dict(), f'/content/results/{name}/optimizer.pth')

def epochesTest(name, model_optimizer_dict, n_epochs):
    import time
    print(f'Optimizer: {name} - Start')

    startTime = time.time()
    test(name, model_optimizer_dict)
    for epoch in range(1, n_epochs + 1):
        train(name, model_optimizer_dict,epoch)
        test(name, model_optimizer_dict)

    print(f'Optimizer: {name} - End')
    _time = "{:.4f}".format(time.time() - startTime)
    print(f'Time:{_time} sec\n')

    txtForPlot(
        train_losses=model_optimizer_dict['train_losses'],
        train_counter=model_optimizer_dict['train_counter'],
        test_losses=model_optimizer_dict['test_losses'],
        test_counter=model_optimizer_dict['test_counter'],
        pathToSave='/content/results',
        name=name)

def test(name, model_optimizer_dict):
    # 1
    model = model_optimizer_dict['model']
    optimizer = model_optimizer_dict['optimizer']
    train_losses = model_optimizer_dict['train_losses']
    train_counter = model_optimizer_dict['train_counter']
    test_losses = model_optimizer_dict['test_losses'] 
    test_counter = model_optimizer_dict['test_counter'] 
    #2 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))