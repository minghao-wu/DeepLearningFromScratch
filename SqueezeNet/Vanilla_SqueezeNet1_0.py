class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1)
        self.squ_relu = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand1x1_channels, kernel_size=1)
        self.relu_1x1 = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand3x3_channels, kernel_size=3, padding=1)
        self.relu_3x3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squ_relu(x)
        x_1x1 = self.expand1x1(x)
        x_1x1 = self.relu_1x1(x_1x1)
        x_3x3 = self.expand3x3(x)
        x_3x3 = self.relu_3x3(x_3x3)
        return(torch.cat([x_1x1, x_3x3], 1))


class SqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNet, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fire2 = Fire(in_channels=96, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64)
        self.fire3 = Fire(in_channels=128, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64)
        self.fire4 = Fire(in_channels=128, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire5 = Fire(in_channels=256, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=128)
        self.fire6 = Fire(in_channels=256, squeeze_channels=48, expand1x1_channels=192, expand3x3_channels=192)
        self.fire7 = Fire(in_channels=384, squeeze_channels=48, expand1x1_channels=192, expand3x3_channels=192)
        self.fire8 = Fire(in_channels=384, squeeze_channels=64, expand1x1_channels=256, expand3x3_channels=256)
        self.pool8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire9 = Fire(in_channels=512, squeeze_channels=64, expand1x1_channels=256, expand3x3_channels=256)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=13, stride=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.head(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.pool4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.pool8(x)
        x = self.fire9(x)
        x = self.conv10(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
