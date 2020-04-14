Weakly-Supervised Computer Vision Learning
--------------------------------------------


This repo implements the common weakly-supervised CV algorithms to faciliate
comparison between them.


Install
-------
Please isntall the library via pip with the `setup.py` file. Please not that at
the moment there is not yet a `requirements.txt` file which helps with the 
dependencies. That will be added soon.



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
The module so far has been tested with the PennFudanDataSet and the Vaihingen
2D dataset. For a detailed description of how to integrate these two dataset
into the current code, please go to the data [folder](./data/README.md)
