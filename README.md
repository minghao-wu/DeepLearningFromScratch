# Deep Learning From Scratch
## Introduction
This repository stores a series of reproductions of deep learning milestones in Computer Vision.

For obvious reason, I'm unable to train the model on ImageNet from scratch. Therefore, for the image classification task, all the models are trained on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), whose images are rescaled to be as large as those described in the original papers.

If my model achieves comparable performance in both accuracy and speed with the corresponding model given in the model zoo without pre-trained weight under the same setting in the first three epochs, I consider it is a valid reproduction.

**Stars are welcomed!**

I guess no one will read my code line by line. LOL.

## Image Classification
* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* [VGG](https://arxiv.org/abs/1409.1556)
* [GoogLeNet](https://arxiv.org/abs/1409.4842)
* [ResNet](https://arxiv.org/abs/1512.03385)
* [ResNeXt](https://arxiv.org/abs/1611.05431)
* [SqueezeNet](https://arxiv.org/abs/1602.07360)
* [MobileNet](https://arxiv.org/abs/1704.04861)
* [DenseNet](https://arxiv.org/abs/1608.06993)
* [ShuffleNet](https://arxiv.org/abs/1707.01083)
## Model Zoo
My models are compared with models in this section.
* [AlexNet](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.alexnet)
* [VGG16](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.vgg16)
* [VGG19](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.vgg19)
* [GoogLeNet](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/googlenet.py)
* [ResNet34](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.resnet34)
* [ResNet50](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.resnet50)
* [ResNeXt50-32x4d](https://github.com/prlz77/ResNeXt.pytorch)
* [SqueezeNet 1.0](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.squeezenet1_0)
* [MobileNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)
* [DenseNet121](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.densenet121)
* [ShuffleNet](https://github.com/jaxony/ShuffleNet/blob/master/model.py)

