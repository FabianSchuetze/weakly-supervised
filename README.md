[![Build Status](https://travis-ci.com/FabianSchuetze/weakly-supervised.svg?branch=master)](https://travis-ci.com/FabianSchuetze/weakly-supervised)
Weakly-Supervised Computer Vision Learning
--------------------------------------------


This repo implements common weakly-supervised CV algorithms to facilitate comparison.

Install
-------
Please install the dependencies of the module with `pip install -r
requirements.txt`. Then, install the library with
`python setup.py build develop`. The installation was successfully tested on travis-ci and a AWS deep learning instances. To check the asusmed system configuration, please look at teh [Docker image](./.travis.yaml).


Run Models
----------
The models can be run from the root directory with
`python examples/learner.py --dataset 'DS' --weakly_supervised` for training a weakly supervised model on dataset 'DS'. If you prefer the conventional supervised instance segmentation framework, you have to replace `--weakly_supervised` with `supervised`. Please read below which datasets are supported by default. When choosing `weakly-supervised` you create a model from the [Learning to Segment Every Thing](https://arxiv.org/abs/1703.06870) paper. This model trains feature embedding extensively on bounding boxes, and then trains a segmentation-head on a small set of images with segmented annotations. If you instead choose the `supervised`, you work with the [Mask
R-CNN](https://arxiv.org/abs/1703.06870) framework.


Preliminary Results
-------------------
Figure XX shows the effectivness of weakly-suprevised learning on the Vaihigen dataset. The training protocol for each of the experimant is as follows:

*Weakly-Supervised*
Each epoch begins by training 1000 images exclusively on bounding boxes and object labels. This helps to refine the backbone of the model and the ROI proposal network. Then, 500 images are used to train it's instance segmentation capabilities by transfer learning: The backbone and ROI proposal network transforms the input picture in a feature representation. Given a set of ROIs, the network predicts pixel-by-pixel annotations. These annotations are estimated with a de-convolutional network with weights $w_{\text{Seg}}$. With weak-supervision, these weights are a function of the weights $w_{\text{Box}}$ which are used to generate bounding box annotations for each class, $ w_{\text{Seg}} = f(w_{\text{Box}}).$. The 500 images are thus used to train the small network $f$ to predict good instance segmentation weights.
*Supervised*
The training protocol for the fully supervised model is different: The model uses the 500 images with pixel-by-pixel annotations for end-to-end training. The weights $w_{\text{Seg}}$ are not a function of the bounding box weights anymore but trained independently. Of course, such training grants more flexibility to train the weights $w_{\text{Seg}}$ but the reduced number of sample images also hampers the richness of the dataset to train the entire model.
*Oracle*
Finally, the last training protocol uses 1500 images for end-to-end training and serves a the upper bound for the accuracy.


Datasets
--------
The module so far has been tested with the PennFudanDataSet, the Vaihingen
2D dataset, Coco 2014, and Pascal VOC 2007. For a detailed description of how to integrate these datasets
into the current code, please go to the data [folder](./data/README.md).


ToDo
-----
1. Run experiments with the other datasets to compare against SOTA results.
2. Use caching of the Vaihingen database to allow faster data loading (At the moment, the data serving latency
slows down the GPU)
3. Comapre the result with a Deeplab implementation
4. Better Unit tests

