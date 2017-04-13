#!/bin/sh
~/anaconda3/bin/python bench.py --n 1 --net vgg19
~/anaconda3/bin/python bench.py --n 2 --net vgg19
~/anaconda3/bin/python bench.py --n 3 --net vgg19
~/anaconda3/bin/python bench.py --n 4 --net vgg19
~/anaconda3/bin/python bench.py --n 1 --net Inception-BN
~/anaconda3/bin/python bench.py --n 2 --net Inception-BN
~/anaconda3/bin/python bench.py --n 3 --net Inception-BN
~/anaconda3/bin/python bench.py --n 4 --net Inception-BN
~/anaconda3/bin/python bench.py --n 1 --net resnet-152
~/anaconda3/bin/python bench.py --n 2 --net resnet-152
~/anaconda3/bin/python bench.py --n 3 --net resnet-152
~/anaconda3/bin/python bench.py --n 4 --net resnet-152
~/anaconda3/bin/python bench.py --n 1 --net resnext-101
~/anaconda3/bin/python bench.py --n 2 --net resnext-101
~/anaconda3/bin/python bench.py --n 3 --net resnext-101
~/anaconda3/bin/python bench.py --n 4 --net resnext-101