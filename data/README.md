The code has been tested with two datasets which are described below. If you
want to add other datasets, please speficy your database in the same way as the
examples in `transferlearing/data/`. 

1. Vaihing, ISPRS
-----------------
High Density Aerial Image for 2D semantic labeling, [described in
detail](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html)
Please download the files
`ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip` (the ground truth
annotations) and `ISPRS_semantic_labeling_Vaihingen.zip` (the images).  The
default root directory is located in the folder `data/vaihingen`.  The images
are then assumed to be located in the subfolder 'images' and the annotations in
'annotations'. If you want to specify a different root folder, please do that
in in 'examples/learner.py'. The database to generates samples is written
[here](./transferlearning/data/vaihingen.py). 

2. PennFudanDataSet
---------------------
Pedestrian Detection and Segmentation, defined
[here](https://www.cis.upenn.edu/~jshi/ped_html/) There are only two classes
(people and back-ground). The root folder for the data is assumed to be in
`data\PennFudanPen`. The database to generated samples is specified
[here](./transferlearning/data/penndata.py)

TODO
----
Add default support for Pascal VOC and COCO datasets.
