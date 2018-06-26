'''
    This script creates a CNN model and trains it on the MNIST dataset.
    It provides a tensorboard visualization during the training phase, while
    providing the trained model at the end.
'''

import argparse, os
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

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Constants
    BATCH_SIZE = 16
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
    parser.add_argument('--arch', type=str, default='baseline',
        help="""\
        The architecture to use.
    """)
    parser.add_argument('--epochs', type=int, default=2,
        help="""\
        Number of training epochs.
    """)
    parser.add_argument('--lr', type=float, default=0.01,
        help="""\
        Learning rate
    """)
    FLAGS, unparsed = parser.parse_known_args()

    EPOCHS = FLAGS.epochs
    LEARNING_RATE = FLAGS.lr

    # Selecting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data: we only need trainset here
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Instantiate net
    if FLAGS.arch == 'baseline':
        from archs import BaselineNet
        net = BaselineNet().to(device)
    elif FLAGS.arch == 'vgg':
        from archs import VggNet
        net = VggNet().to(device)
    else:
        raise Exception('You need to specify a valid architecture.')
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

    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    torch.save(net.state_dict(), filepath)
