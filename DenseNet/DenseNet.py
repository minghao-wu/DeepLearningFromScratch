import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_factor = 4, drop_rate = 0.5):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=growth_rate * bottleneck_factor, kernel_size=1, stride=1),

            nn.BatchNorm2d(growth_rate * bottleneck_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=growth_rate * bottleneck_factor, out_channels=growth_rate, kernel_size=3, stride=1, padding=1),

            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):
        identity = x
        x = self.layer(x)
        total = torch.cat([identity, x], 1)
        return(total)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.layer(x)
        return(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate, compression_factor, blocks):
        super(DenseNet, self).__init__()
        num_features = growth_rate * 2
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.DenseBlock1 = self._make_block(DenseLayer, blocks[0], in_channels=num_features, growth_rate=growth_rate)
        num_features = num_features + blocks[0] * growth_rate
        compressed_features = int(num_features * compression_factor)
        self.Transition1 = Transition(in_channels=num_features, out_channels=compressed_features)

        self.DenseBlock2 = self._make_block(DenseLayer, blocks[1], in_channels=compressed_features, growth_rate=growth_rate)
        num_features = compressed_features + blocks[1] * growth_rate
        compressed_features = int(num_features * compression_factor)
        self.Transition2 = Transition(in_channels=num_features, out_channels=compressed_features)

        self.DenseBlock3 = self._make_block(DenseLayer, blocks[2], in_channels=compressed_features, growth_rate=growth_rate)
        num_features = compressed_features + blocks[2] * growth_rate
        compressed_features = int(num_features * compression_factor)
        self.Transition3 = Transition(in_channels=num_features, out_channels=compressed_features)

        self.DenseBlock4 = self._make_block(DenseLayer, blocks[3], in_channels=compressed_features, growth_rate=growth_rate)
        num_features = compressed_features + blocks[3] * growth_rate
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(in_features=num_features, out_features=num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.entry(x)
        x = self.DenseBlock1(x)
        x = self.Transition1(x)
        x = self.DenseBlock2(x)
        x = self.Transition2(x)
        x = self.DenseBlock3(x)
        x = self.Transition3(x)
        x = self.DenseBlock4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return(x)

    def _make_block(self, layer, num_layers, in_channels, growth_rate, bottleneck_factor = 4, drop_rate = 0.5):
        block = []
        for i in range(num_layers):
            block.append(layer(in_channels=in_channels + i * growth_rate, growth_rate=growth_rate, bottleneck_factor=bottleneck_factor, drop_rate=drop_rate))
        return(nn.Sequential(*block))

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
