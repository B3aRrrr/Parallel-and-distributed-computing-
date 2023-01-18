import torch
import torchvision

def make_train_loader_test_loader(
    n_epochs = 3,
    batch_size_train = 64,
    batch_size_test = 1000,
    learning_rate = 0.01,
    momentum = 0.5,
    log_interval = 10,
    random_seed = 1):
  import torch
  import torchvision

  torch.backends.cudnn.enabled = False
  torch.manual_seed(random_seed)
  
  train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_test, shuffle=True)
  return train_loader,test_loader

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1