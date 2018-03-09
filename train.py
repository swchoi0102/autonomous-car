import torch
import argparse
import numpy as np
import torch.optim as optim

from torch.autograd import Variable
from utils import batch_generator, load_data
from models import *


def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='./data/IMG')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=20000)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=40)
    # parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-cuda', help='ables CUDA training', dest='cuda', type=bool, default=False)
    args = parser.parse_args()

    net = Model()
    if args.cuda:
        net.cuda()

    print(net)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(args)
    trainloader = batch_generator('./data/IMG', x_train, y_train, 32, is_training=True)
    testloader = batch_generator('./data/IMG', x_test, y_test, 32, is_training=False)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, target = data
            inputs = np.transpose(inputs, [0, 3, 1, 2])

            inputs = torch.from_numpy(inputs).float()
            target = torch.from_numpy(target).float()

            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            # wrap them in Variable
            inputs, target = Variable(inputs), Variable(target)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, target)
            # loss.backward()
            # optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            print(i)
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

    running_loss = list()
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, target = data
        inputs = np.transpose(inputs, [0, 3, 1, 2])

        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()

        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        # wrap them in Variable
        inputs, target = Variable(inputs), Variable(target)

        # forward
        outputs = net(inputs)
        loss = criterion(outputs, target)

        # store loss
        running_loss.append(loss.data[0])

    print('Average test MSE loss : {}'.format(sum(running_loss) / len(running_loss)))


if __name__ == '__main__':
    main()
