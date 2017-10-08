class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return(x)


class MobileNet(nn.Module):
    def __init__(self, num_classes, alpha=1.0):
        super(MobileNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=int(alpha * 32), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(alpha * 32)),
            nn.ReLU(inplace=True)
        )
        self.entry = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(alpha * 32), out_channels=int(alpha * 64), stride=1, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 64), out_channels=int(alpha * 128), stride=2, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 128), out_channels=int(alpha * 128), stride=1, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 128), out_channels=int(alpha * 256), stride=2, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 256), out_channels=int(alpha * 256), stride=1, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 256), out_channels=int(alpha * 512), stride=2, padding=1)
        )

        self.middle = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(alpha * 512), out_channels=int(alpha * 512), stride=1, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 512), out_channels=int(alpha * 512), stride=1, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 512), out_channels=int(alpha * 512), stride=1, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 512), out_channels=int(alpha * 512), stride=1, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 512), out_channels=int(alpha * 512), stride=1, padding=1)
        )
        self.exit = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(alpha * 512), out_channels=int(alpha * 1024), stride=2, padding=1),
            DepthwiseSeparableConv2d(in_channels=int(alpha * 1024), out_channels=int(alpha * 1024), stride=1, padding=1)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(in_features=int(alpha * 1024), out_features=num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return(x)
        
