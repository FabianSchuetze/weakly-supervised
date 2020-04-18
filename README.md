Weakly-Supervised Computer Vision Learning
--------------------------------------------


This repo implements the common weakly-supervised CV algorithms to faciliate
comparison between them.


Install
-------
Please install the dependencies of the module with `pip install -r
requirements.txt`. Then, install the library with
`python setup.py build develop`. The installation was tested successfully
with the AWS deep learning instaces at Ubuntu 18.04.


Run Models
----------
The models can be run from the root directory with
`python examples/learner.py --dataset 'DS' --weakly_supervised` for training a weakly supervised model on dataset 'DS'. If you prefer the conventional supervised instance segmentation framework, you have to replace `--weakly_supervised` with `supervised`. Please read below which datasets are supported by default. When choosing `weakly-supervised` you create a model from the [Learning to Segment Every Thing](https://arxiv.org/abs/1703.06870) paper. This model trains feature embedding extensively on bounding boxes, and then trains a segmentation-head on a small set of images with segmented annotations.   If you instead choose the `supervised`, you work with the [Mask
R-CNN](https://arxiv.org/abs/1703.06870) framework. 

Datasets
--------
The module so far has been tested with the PennFudanDataSet, the Vaihingen
2D dataset, Coco 2014, and Pascal VOC 2007. For a detailed description of how to integrate these datasets
into the current code, please go to the data [folder](./data/README.md)

