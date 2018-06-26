'''
    This script creates a CNN model and trains it on the MNIST dataset.
    It provides a tensorboard visualization during the training phase, while
    providing the trained model at the end.
'''

import argparse
from tqdm import trange, tqdm
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Define the network modules
        self.conv1 = nn.Conv2d(1, 32, 3) # input_filters=1 because MNIST is gray-scale
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(9216, 128) # 9216 is the size of the flattened layer
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Defines what happens in the forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':

    # Constants
    BATCH_SIZE = 4
    LEARNING_RATE = 0.01
    EPOCHS = 2
    LOG_STEP = 10

    # Argparse: save_directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='model',
        help="""\
        The directory in which to save the trained model.
    """)
    parser.add_argument('--alias', type=str, default='mark1',
        help="""\
        The alias to use when logging for tensorboard.
    """)
    FLAGS, unparsed = parser.parse_known_args()

    # Selecting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data: we only need trainset here
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Instantiate net
    net = Net().to(device)
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

    # Define tensorboard writer
    writer = SummaryWriter('runs/' + FLAGS.alias)
    log_step = 1

    # Train
    for epoch in trange(EPOCHS):

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainset)//BATCH_SIZE):
            # Get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Stats
            running_loss += loss.item()
            if i % LOG_STEP == LOG_STEP-1:
                writer.add_scalar('data/loss', running_loss, log_step)
                log_step += 1
                running_loss = 0.0

    writer.close()

    # Save the model for inference
    filepath = FLAGS.save_dir + '/' + FLAGS.alias + '.torch'
    torch.save(net.state_dict(), filepath)
