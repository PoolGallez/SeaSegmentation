# Sea Segmentation - A Bag Of Words approach
## Introduction
This project was developed to solve an Univeristy assignmed which was about Boat Detection/tracking and sea segmentation.
In particular in this code it will be developped the sea segmentation part using C++ as codebase with the OpenCV library.

## The approach
### Why not using neural networks? 
Nowadays everything (i mean this is not an euphemism, **literally everything!**) could be done with deep learning techniques; in fact by doing some researches
about the newest approaches to tackle the semantic segmentation problem you'll notice that everyone uses convolutional neural networks to solve the problem
(UNet,R-CNN,YOLO...).
Despite their actractiveness in terms of performances i found them to be not so instructive, since i could have just downloaded some weights of a pre-trained neural
network and adapt it to my problem by means of transfer learning.

### Graph Segmentation 
Meanshift actually **sucks!**, that's why i've used Graph Segmentation techniques, they are fast and work like magic with these things called graphs whoose theory 
is barely known to me :)

### Bag of Visual Words
This is cool.
