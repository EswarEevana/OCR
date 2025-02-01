import torch
import torch.nn as nn
import torch.optim as optim
from architecture import net9
from dataset import create_dataset

def validator(val_loader = None, net = None):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            # perform max along dimension 1, since dimension 0 is batch dimension
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

def train(epochs = 100, net = None, train_loader = None, val_loader = None, device = 'cpu'):
  '''
    epochs : number of epochs
    net : network (instantiation of architecture class)
    train_loader : training dataset
    test_loader : testing_dataset
  '''
  best_accuracy = -1.0
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  for epoch in range(epochs):  # loop over the dataset multiple times
  
      running_loss = 0.0
      for i, data in enumerate(trainloader):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.to(device)
          labels = labels.to(device)
  
          # zero the parameter gradients
          optimizer.zero_grad()
  
          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
  
          # print statistics
          running_loss += loss.item()
          if i % 10 == 0:    # print every 10 mini-batches
              print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
              running_loss = 0.0
  
      current_accuracy = validator(val_loader = val_loader, net = net)
      if current_accuracy>best_accuracy:
          best_accuracy = current_accuracy
  
          torch.save(
              {'epoch':epoch,
               'model_state_dict': net.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()
               },
               'best_model.pth')
  
      #Save model as checkpoint
      torch.save(
          {'epoch':epoch,
           'model_state_dict': net.state_dict(),
           'optimizer_state_dict': optimizer.state_dict()
           },
           'model_checkpoint.pth')
  
  print('Training was finished!!!')
  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network = net9()
train_loader, test_loader, val_loader = create_dataset()
train(net = network, train_loader = train_loader, val_loader = val_loader, device = device)
