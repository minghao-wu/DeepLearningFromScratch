import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool3(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool5(x)
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = F.softmax(self.fc3(x))
        return(x)

vgg16 = VGG16(1000)
print(vgg16ß)
