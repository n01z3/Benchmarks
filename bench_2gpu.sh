#!/bin/sh

PYTHON_BIN=$HOME/anaconda3/bin/python

$PYTHON_BIN bench.py --n 1 --net vgg19
$PYTHON_BIN bench.py --n 2 --net vgg19
$PYTHON_BIN bench.py --n 1 --net Inception-BN
$PYTHON_BIN bench.py --n 2 --net Inception-BN
$PYTHON_BIN bench.py --n 1 --net resnet-152
$PYTHON_BIN bench.py --n 2 --net resnet-152
$PYTHON_BIN bench.py --n 1 --net resnext-101
$PYTHON_BIN bench.py --n 2 --net resnext-101
