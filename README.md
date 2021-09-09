# Sea Segmentation - A Bag Of Words approach
## Introduction
This project was developed to solve an Univeristy assignmed which was about Boat Detection/tracking and sea segmentation.
In particular in this part of the code it will be developped the sea segmentation part using C++ as codebase with the OpenCV library.

## The assignment
Traffic analysis methods on road scenes are widely used nowadays, however similar methods can be used to analyze boat traffic in the sea.
The goal of this project is to develop a system capable of 1) detecting boats 2) segment the sea 3) track boats in videos.
Keep in mind that sea surface is not regular and it is moving (it might contain white trails dued to boats) and, furthermore, the appearance of a boat change sensibly from model to model (we have high intraclass variability!) as well as from the viewpoint of the picture.

Regarding the boat segmentation task, pixels belonging to the sea should be highlighted by the algorithm which also has to properly analyze: 
* Images without sea or boats
* Boats from any viewpoint
* Presence of boats doesn't imply the presence of the sea and viceversa

The algorithm, in particular, stores the semantic mask assigning white color to pixels belonging to sea regions whereas assigns black to pixels not belonging to sea regions.

The performance evaluation for the Sea Segmentation is done by computing the Pixel Accuracy index.
This performance measure is the easiest to implement for what concerns the semantic segmentation problem, it is not the most accurate though.

### Worktime pipeline

| Task      | Time (hours) |
| ----------- | ----------- |
| Meanshift analysis      | 40       |
| Comparison Meanshift - Graph Segmentation   | 4        |
| Adaptation of the Felzenszwalb-Huttenlocher      | 4       |
| Bag Of Visual Words - Implementation   | 32        |
| Trials with different hyperparameters & performance measure      | 48      |
| Implementation of Local Binary Pattern in Bag Of Words   | 32        |
| Trials with different hyperparameters for LBP & performance measure      | 56     |
| **Total**   |      **216**   |

## The approach
### Why not using neural networks? 
Nowadays everything (i mean this is not an euphemism, **literally everything!**) could be done with deep learning techniques; in fact by doing some researches
about the newest approaches to tackle the semantic segmentation problem you'll notice that everyone uses convolutional neural networks to solve the problem
(UNet,R-CNN,YOLO...).
Despite their actractiveness in terms of performances, those methods are not so instructive, since i could have just downloaded some weights of a pre-trained neural
network and adapt it to my problem by means of transfer learning.
What was more insteresting to test is whether it was possible to achieve great results by using non deep-learning techniques and exploiting the knowledge of classical Computer Vision techniques (**spoiler: yes, it's possible!**).

### The idea
The intuition about the implemented approach is based on a two phases algorithm: the first step consists in having preprocessing phase leading to a great initial segmentation of the image, and the second step involves the classification of each segment found in the previous step.
In the end, what one should obtain is a binary mask highlighting pixels belonging to the sea.

This stuff is really cool, but are we sure that is going to actually work? According to this paper: https://r-libre.teluq.ca/982/1/articleiwatchlife.pdf ,this algorithm can work.

## Phase 1: Pre-segmentation of the scene
### Meanshift
As a first trial, the meanshift algorithm was implemented as it was the only algorithm we studied during the course capable of finding clusters (segments) with no bias whatsoever regarding their shape. 
Unfortunately it was really unintuitive to find a set of hyperparameters for this algorithm capable of achieving great results with decent execution time.

In summary:  Meanshift actually **sucks!** (i don't have a NASA computer, and some images required several minutes to get processed)

### Graph Segmentation 
Graph Segmentation techniques, on the other hand, are fast and work like magic. 
This code uses the original implementation of the Felzenszwalb-Huttenlocher graph segmentor 

### Bag of Visual Words
This is cool.
