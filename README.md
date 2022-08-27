# Sea Segmentation - A Bag Of Words approach
## Introduction
This project was developed to solve an Univeristy assignmed which was about Boat Detection/tracking and sea segmentation.
In particular in this part of the code it will be developed the sea segmentation part using C++ as codebase with the OpenCV library.

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

This stuff is really cool, but are we sure that this is going to actually work? According to this paper: https://r-libre.teluq.ca/982/1/articleiwatchlife.pdf ,this approach should do the job.

## Phase 1: Pre-segmentation of the scene
### Meanshift
As a first trial, the meanshift algorithm was implemented as it was the only algorithm we studied during the course capable of finding clusters (segments) with no bias whatsoever regarding their shape. 
Unfortunately it was really unintuitive to find a set of hyperparameters for this algorithm capable of achieving great results with decent execution time.

TLDR:  Meanshift actually **sucks!** (i don't have a NASA computer, and some images required several minutes to get processed)

### Graph Segmentation 
Graph Segmentation techniques, on the other hand, are fast and work like magic. 
This code uses the original implementation of the Felzenszwalb-Huttenlocher graph segmentor which can be found on the paper cited previously.
This algorithm relies to an underlying weighted graph , where each node corresponds to a
pixel and the edges weighted by some measure of dissimilarity (such as the intensity) connect
neighbouring pixels. With this framework, a segmentation is a partition the original Graph and, therefore, is defined by a set of edges of the original graph. 
The dissimilarity criterion used in the paper can be proved to guarantee neither too coarse nor too fine results, and
this algorithm has a time complexity _O (n log(n) )_, where n is the number of pixel, making it possible to be used with videos.

![Input Image](https://github.com/PoolGallez/SeaSegmentastion/blob/main/markdown/images/20.png?raw=true "Input Image")
![Segmented Image](https://github.com/PoolGallez/SeaSegmentastion/blob/main/markdown/images/20_seg.png?raw=true "Segmentation Result")


### The mask extraction process

One could now think that each pixel is assigned to a number i = 1,...,\# of segments corresponding to the segment that pixel belongs to, however each segment is identified by an number (which in some cases is really high) corresponding to the merged intensities of two similar pixels.
Therefore it was not straitforward the grouping of the pixels into the segments since we couldn't predict the number a pixel will be assigned to. 
Thus, to isolate each segment with binary masks, it was necessary to process the segmentation resut by isolating each individual segment.
The code relies on an unordered map implemented in C++ used in the following function: 

    /**
     * Method that isolates each single segment
     * @author Paolo Galletta
     * @param u the segmentation graph
     * @param width image width
     * @param height image height
     * @return the vector containing the mask of each single segment
     **/
    std::vector<cv::Mat> get_segment_mask(universe *u, int width, int height)
    {
      std::vector<cv::Mat> output;
      int num_sets = u->num_sets();

      // To isolate the segments, an hash map is used
      std::unordered_multimap<int, cv::Point> hash_map;

      // Traverse the image
      for (int y = 0; y < height; y++)
      {
        for (int x = 0; x < width; x++)
        {
          int comp = u->find(y * width + x);

          // Group the points belonging to the same segment
          hash_map.insert({comp, cv::Point(x, y)});
        }
      }

      // Create the Mat single masks
      for (auto it = hash_map.begin(); it != hash_map.end();)
      {
        cv::Mat current_segment(cv::Size(width, height), CV_8UC1);
        auto const &key = it->first;

        auto range = hash_map.equal_range(key);

        auto iter = range.first;
        for (; iter != range.second; ++iter)
        {
          current_segment.at<uchar>(iter->second) = 255;
        }
        output.push_back(current_segment);

        while (++it != hash_map.end() && it->first == key)
          ;
      }
      return output;
    }
    
### What is this code achieving 
In essence the code above scans the entire result of the segmentation process (which is an image having each pixel assigned to a number corresponding to the merged intensities) and by means of an hash map it groups the pixels having the same value ( uses as the key of the map the value of the pixels, if two or more pixels have the same value they will be concatenated in the same list). 
Finally the hash map is traversed to create the binary mask for each segment, it might seem laborious, maybe it could be improved, but it does the job.

### Bag of Visual Words
Once that the individual segment masks are extracted, we can classify each segment by using a Bag Of Visual Words approach which performs classification by obtaining an histogram representation of the SIFT features extracted in each individual segment.
In particular, for the dictionary it has been used the K-Means clustering algorithm with 200 clusters, and for the classification an SVM linear classifiers (as SIFT descriptors lie in a 128th dimentional space, it shouldn't be a problem to find a separating hyperplane).

Dataset Used to identify the Sea: MASATI.

![Semantic Segmentation Result](https://github.com/PoolGallez/SeaSegmentastion/blob/main/markdown/images/20_sem_seg.png?raw=true "Result of the semantic segmentation")

## Performance achieved

|Filename	|	Sea Accuracy |		Non Sea Accuracy	|	Total Accuracy                              |
|-----------|----------------|--------------------------|-----------------------------------------------|
|00.png		|	0.89		 |   0.87				    |        0.88				                    |
|01.png		|	0.83		 |	0.87				    |        0.85				                    |				
|02.png		|	0.90		 |	0.96				    |        0.92				                    |				
|03.png		|	0.002		 |	0.99				    |        0.73				                    |
|04.png		|	0.93		 |	0.96				    |        0.94				                    |
|05.png		|	0.02		 |	0.85				    |        0.43				                    |				
|06.png		|	0.85		 |	0.94				    |        0.88				                    |
|07.png		|	0.89		 |	0.98				    |        0.93				                    |
|08.png		|	0.01		 | 	0.93				    |        0.40				                    |				
|09.png		|	0.88		 |	0.99				    |        0.91				                    |
|10.png		|	0.91		 |	0.95				    |        0.93				                    |			
|11.png		|	0.002		 |	0.98				    |        0.68		<-- fine dataset venezia    |
|12.png		|	0			 |   0.98				    |        0.81		<-- aidaship                |
|13.png		|	0.99		 |	0.99				    |        0.99				                    |
|14.png		|	0.99		 |	0.999				    |        0.99				                    |
|15.png		|	0.99		 |	0.99				    |        0.99				                    |
|16.png		|	0.01		 |	0.997				    |        0.81				                    |
|17.png		|	0.99		 |	0.99				    |        0.99				                    |
|18.png		|	0.99		 |	0.99				    |        0.99				                    |
|19.png		|	0.99		 |	0.98				    |        0.99				                    |
|20.png		|	0.97		 |	0.99				    |        0.99				                    |
|21.png		|	0.023		 |	0.99				    |        0.14				                    |
|22.png		|	0.99		 |	0.99				    |        0.99				                    |
|24.png		|	0			 |   1 				        |        1				                        |
|25.png		|	0.94		 |	0.99				    |        0.94				                    |
|26.png		|	0.92		 |	0.75				    |        0.98				                    |
|27.png		|	0.99		 |	0.84				    |        0.95				                    |
|28.png		|	0.96		 |	0.59				    |        0.88				                    |				
|29.png		|	0.90		 |	0.99				    |        0.94				                    |
