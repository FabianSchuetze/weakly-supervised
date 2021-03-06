The code has been tested with two datasets which are described below. If you
want to add other datasets, please speficy your database in the same way as the
examples in `transferlearing/data/`.

1. Vaihing, ISPRS
-----------------
High-Density Aerial Image for 2D semantic labeling, [described in
detail](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html) Please download the files `ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip` (the ground truth annotations) and `ISPRS_semantic_labeling_Vaihingen.zip` (the images). The default root directory is located in the folder `data/vaihingen`. The images are assumed to be located in the subfolder `images` and the annotations in `annotations`. The database to generate samples is written [here](./transferlearning/data/vaihingen.py).

2. PennFudanDataSet
-------------------
Pedestrian Detection and Segmentation, defined [here](https://www.cis.upenn.edu/~jshi/ped_html/) There are only two classes
(people and back-ground). The root folder for the data is assumed to be in `data\PennFudanPen`. The database to generate samples is specified
[here](./transferlearning/data/penndata.py).

3. CoCo Database
----------------
The [Common Objects in Context dataset](http://cocodataset.org/#home). The 2014 version of the dataset has about 8000 images with 91 classes. The root folder for the data is assumed to be in `data\coco`. The images are then located in `train2014` (equivalent for the 2017 database) and the annotatations in `annotations`. The database to generate samples is specified [here](./transferlearning/data/coco.py).

4. Pascal VOC
-------------
So far, the 2007 version is tested. This version has 20 target classes. The root folder for the data is assumed to be in `data\VOC2007`. Which contains the `VOCDevkit` as can be downloaded [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html). The database to generate samples is specified [here](./transferlearning/data/pascal_voc.py).


