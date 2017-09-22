# Deep Learning From Scratch
## Introduction
I aim to reproduce some selected deep learning models from scratch as a kind of practice.

For obvious reason, I'm unable to train the model on ImageNet from scratch.

Therefore, all the models are trained on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), whose images is rescaled to 256 * 256 and randomly cropped to a 227 * 227 patch.

If my model achieves comparable performance with the corresponding model given in the model zoo without pre-trained weight under the same setting, I consider it is a valid replicate.

## Models
* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* [VGG16 & VGG19](https://arxiv.org/abs/1409.1556)
* [GoogLeNet](https://arxiv.org/abs/1409.4842)

## Model Zoo
My models are compared with models in this section.
* [AlexNet](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.alexnet)
* [VGG16](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.vgg16)
* [VGG19](http://pytorch.org/docs/master/torchvision/models.html#torchvision.models.vgg19)
* [GoogLeNet](https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py)
