def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class Conv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(Conv_1x1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        x = self.layer(x)
        return(x)

class DWConv_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DWConv_3x3, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        x = self.layer(x)
        return(x)

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups, mode):
        super(ShuffleUnit, self).__init__()
        self.mode = mode
        self.groups = groups
        if mode == "add":
            stride = 1
        if mode == "cat":
            stride = 2
        self.gconv_1x1_head = Conv_1x1(in_channels=in_channels, out_channels=(out_channels // 4), groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.dwconv_3x3 = DWConv_3x3(in_channels=(out_channels // 4), out_channels=(out_channels // 4), stride=stride)
        self.gconv_1x1_cat_tail = Conv_1x1(in_channels=(out_channels // 4), out_channels=(out_channels - in_channels), groups=groups)
        self.gconv_1x1_add_tail = Conv_1x1(in_channels=(out_channels // 4), out_channels=out_channels, groups=groups)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.mode == "add":
            identity = x
            x = self.gconv_1x1_head(x)
            x = channel_shuffle(x, self.groups)
            x = self.dwconv_3x3(x)
            x = self.gconv_1x1_add_tail(x)
            x = x + identity
            x = self.relu(x)
            return(x)
        if self.mode == "cat":
            identity = x
            identity = self.avgpool(identity)
            print("stage head identity: ", identity.size())
            x = self.gconv_1x1_head(x)
            x = channel_shuffle(x, self.groups)
            x = self.dwconv_3x3(x)
            x = self.gconv_1x1_cat_tail(x)
            x = torch.cat([x, identity], 1)
            x = self.relu(x)
            return(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, repeats, groups):
        super(Stage, self).__init__()
        self.head = ShuffleUnit(in_channels=in_channels, out_channels=out_channels, groups=groups, mode="cat")
        self.body = self._make_stage(in_channels=out_channels, out_channels=out_channels, groups=groups, repeats=repeats)

    def forward(self, x):
        x = self.head(x)
        print("stage head: ", x.size())
        x = self.body(x)
        return(x)

    def _make_stage(self, in_channels, out_channels, groups, repeats):
        layers = []
        for i in range(repeats):
            layers.append(ShuffleUnit(in_channels=in_channels, out_channels=out_channels, groups=groups, mode="add"))
        return(nn.Sequential(*layers))


class ShuffleNet(nn.Module):
    def __init__(self, num_classes, groups=3):
        super(ShuffleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = Stage(in_channels=24, out_channels=240, repeats=3, groups=groups)
        self.stage2 = Stage(in_channels=240, out_channels=480, repeats=7, groups=groups)
        self.stage3 = Stage(in_channels=480, out_channels=960, repeats=3, groups=groups)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(in_features=960, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        print("conv1: ", x.size())
        x = self.pool(x)
        print("pool: ", x.size())
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return(x)
