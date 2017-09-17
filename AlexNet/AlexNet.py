import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return(x)


net = AlexNet(1000)
print(net)
