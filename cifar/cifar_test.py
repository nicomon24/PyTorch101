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

if __name__ == '__main__':

    print("PyTorch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)

    # Constants
    BATCH_SIZE = 16

    # Argparse: save_directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='model',
        help="""\
        File path of the trained model to evaluate.
    """)
    parser.add_argument('--arch', type=str, default='baseline',
        help="""\
        The architecture to use.
    """)
    FLAGS, unparsed = parser.parse_known_args()

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        # Space for other transformations
        transforms.ToTensor() # We need this to get a tensor instead of a PIL image
    ])
    testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Instantiate net
    if FLAGS.arch == 'baseline':
        from archs import BaselineNet
        net = BaselineNet().to(device)
    elif FLAGS.arch == 'vgg':
        from archs import VggNet
        net = VggNet().to(device)
    else:
        raise Exception('You need to specify a valid architecture.')

    # Load the model
    net.load_state_dict(torch.load(FLAGS.checkpoint, map_location=device_name))
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
