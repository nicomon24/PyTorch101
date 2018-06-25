'''
    This script creates the same CNN model as in the train script,
    loads the weights from a specified checkpoint file and
    finally, computes the accuracy of the trained model on the MNIST test set
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

    # Argparse: save_directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='model',
        help="""\
        File path of the trained model to evaluate.
    """)
    FLAGS, unparsed = parser.parse_known_args()

    transform = transforms.Compose([
        # Space for other transformations
        transforms.ToTensor() # We need this to get a tensor instead of a PIL image
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Load the model
    net = Net()
    net.load_state_dict(torch.load(FLAGS.checkpoint))
    net.eval()

    # Compute predictions
    accuracy = 0
    for i, data in tqdm(enumerate(testloader, 0), total=len(testset)//BATCH_SIZE):
        inputs, labels = data
        outputs = net(inputs)
        o = outputs.detach().numpy()
        predictions = np.argmax(o, axis=1)
        # Add to accumulator
        accuracy += np.equal(predictions, labels).sum().item()

    # Final accuracy
    accuracy = accuracy / len(testset)
    print("Final accuracy:", accuracy)
