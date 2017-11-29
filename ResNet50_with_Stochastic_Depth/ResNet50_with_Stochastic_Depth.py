class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=(out_channels * 4), kernel_size=1),
            nn.BatchNorm2d((out_channels * 4)),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=(out_channels * 4), kernel_size=1, stride=stride),
            nn.BatchNorm2d((out_channels * 4))
        )
        self._initialize_weights()

    def forward(self, x, active, prob):      
        if self.training:
            if active == 1:
#                 print("active")
                identity = x
                identity = self.downsample(identity)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = x + identity
                x = self.relu(x)
                return(x)
            else:
#                 print("inactive")
                x = self.downsample(x)
                x = self.relu(x)
                return(x)
        else:
            identity = x
            identity = self.downsample(identity)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = prob * x + identity
            x = self.relu(x)
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

class Group(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, stride):
        super(Group, self).__init__()
        self.num_blocks = num_blocks
        self.head_layer = BottleNeck(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.tail_layer = BottleNeck(in_channels=(out_channels * 4), out_channels=out_channels, stride=1)
    def forward(self, x, active, probs):
        x = self.head_layer(x, active[0], probs[0])
        for i in range(1, self.num_blocks):
            x = self.tail_layer(x, active[i], probs[i])
        return(x)

class ResNet50_Stochastic_Depth(nn.Module):
    def __init__(self, num_classes, pL=0.5):
        super(ResNet50_Stochastic_Depth, self).__init__()
        self.num_classes = num_classes
        self.probabilities = torch.linspace(start=1, end=pL, steps=16)
        self.actives = torch.bernoulli(self.probabilities)
        
        self.head = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = Group(num_blocks = 3, in_channels=64, out_channels=64, stride=1)
        self.group2 = Group(num_blocks = 4, in_channels=256, out_channels=128, stride=2)
        self.group3 = Group(num_blocks = 6, in_channels=512, out_channels=256, stride=2)
        self.group4 = Group(num_blocks = 3, in_channels=1024, out_channels=512, stride=2)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):

        self.actives = torch.bernoulli(self.probabilities)
#         print("The sum of actives blocks: ", int(torch.sum(self.actives)))
        x = self.head(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.group1(x, self.actives[:3], self.probabilities[:3])
        x = self.group2(x, self.actives[3:7], self.probabilities[3:7])
        x = self.group3(x, self.actives[7:13], self.probabilities[7:13])
        x = self.group4(x, self.actives[13:], self.probabilities[13:])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return(x)
