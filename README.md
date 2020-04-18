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


Run Test
--------
The predictions can be run from `examples/learner.py`. The by-default supported
datasets are described below. The example creates a [Mask
R-CNN](https://arxiv.org/abs/1703.06870) model and predicts bounding boxes,
labels and instacen segmentation. The
[network](./transferlearning/supervised.py) contains experimental support for
the weakly-supervised mask predictions from bounding boxes as described in
[Learning to Segment Every Thing](https://arxiv.org/abs/1703.06870). To allow
for weak-supervision, please modify line `77` from 
```python
MODEL = Supervised(N_GROUPS, PROCESSING, weakly_supervised=False)
to
MODEL = Supervised(N_GROUPS, PROCESSING, weakly_supervised=True)
```

Datasets
--------
The module so far has been tested with the PennFudanDataSet, the Vaihingen
2D dataset, Coco 2014, and Pascal VOC 2007. For a detailed description of how to integrate these datasets
into the current code, please go to the data [folder](./data/README.md)

