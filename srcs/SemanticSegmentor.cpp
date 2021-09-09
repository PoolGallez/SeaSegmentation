#include "SemanticSegmentor.h"

/**
 * Semantic Segmentor Constructor
 * @param img the image to be segmented
 * @param graphSegmented the result of the graph segmentation algorithm
 **/
SemanticSegmentor::SemanticSegmentor(cv::Mat img, cv::Mat graphSegmented)
{
    image = img;
    this->graphSegmented = graphSegmented;
}

/**
 * Semantic Segmentor Constructor
 * @param img the image to be segmented
 **/
SemanticSegmentor::SemanticSegmentor(cv::Mat img)
{
    image = img;
}

/**
 * Semantic Segmentor Constructor
 * @param img the image to be segmented
 * @param graphSegmented the result of the graph segmentation algorithm
 * @param masks the single segment's binary masks
 **/
SemanticSegmentor::SemanticSegmentor(cv::Mat img, cv::Mat graphSegmented, std::vector<cv::Mat> masks)
{
    image = img;
    this->graphSegmented = graphSegmented;
    this->masks = masks;
}

/**
 * Destructor
 **/
SemanticSegmentor::~SemanticSegmentor()
{
    image.release();
    graphSegmented.release();
    std::vector<cv::Mat>().swap(masks);
}

/**
 * Masks setter
 * @param masks masks to be set
 **/
void SemanticSegmentor::setMasks(std::vector<cv::Mat> masks)
{
    this->masks = masks;
}

/**
 * GraphSegmented Setter
 * @param graphSegmented the result of the graph segmentation algorithm to be set
 **/
void SemanticSegmentor::setGraphSegmented(cv::Mat graphSegmented)
{
    this->graphSegmented = graphSegmented;
}

/**
 * GraphSegmented getter
 * @return the results of the graph segmentation algorithm relative to the image
 **/
cv::Mat SemanticSegmentor::getGraphSegmented()
{
    return graphSegmented;
}

/**
 * Image Getter
 * @return the image being processed
 **/
cv::Mat SemanticSegmentor::getImage()
{
    return image;
}

/**
 * Masks getter
 * @return the single segment's binary masks
 **/
std::vector<cv::Mat> SemanticSegmentor::getMasks()
{
    return masks;
}

/**
 * Single mask getter
 * @param i chooses among the masks
 * @return the i-th binary mask
 **/
cv::Mat SemanticSegmentor::getMask(int i)
{
    return masks[i];
}

/**
 * Execute binary Segmentation
 * @param bow the Bag Of Words utility class
 * @return the binary mask separating sea and non sea
 **/
cv::Mat SemanticSegmentor::getSemanticSegmented(BagOfWords bow)
{
    /**
     * Steps for semantic segmentation
     * 1. Extract features from each segment
     * 2. Get the Bag Of Words representation of descriptor for each segment
     * 3. Classify the obtained histogram
     **/

    cv::Mat output = image.clone();

    // Iterate single binary mask
    for (cv::Mat mask : masks)
    {
        // Predict the histogram related to the current mask
        int prediction = bow.predict(image, mask);

        if (prediction == BagOfWords::POSITIVE_CLASS)
        {
            output.setTo(255, mask);
        }
        else
        {
            output.setTo(0, mask);
        }
    }
    std::cout << "Semantic Segmentation finished" << std::endl;
    return output;
}

/**
 * PixelAccuracy metric getter
 * @param semanticSegmentation the result of the Semantic Segmentation
 * @param groundTruth the ground truth used to compare with the result
 * @param seaPixelAccuracy output containing the sea pixel accuracy related to the number of water pixels
 * @param nonSeaPixelAccuracy output containing the non sea pixel accuracy related to the number of non water pixels
 * @param totalAccuracy output containing the general pixel accuracy (both sea and non sea)
 **/
void SemanticSegmentor::getPixelAccuracy(cv::Mat semanticSegmentation, cv::Mat groundTruth, double &seaPixelAccuracy, double &nonSeaPixelAccuracy, double &totalAccuracy)
{
    double seaAcc = 0, nonSeaAcc = 0, seaCount = 0, nonSeaCount = 0, totAcc = 0;

    for (int i = 0; i < semanticSegmentation.rows; i++)
    {
        // Use the pointers to the rows to have a quicker traversal
        uchar *sem_ptr = semanticSegmentation.ptr<uchar>(i);
        uchar *grnd_ptr = groundTruth.ptr<uchar>(i);
        for (int j = 0; j < semanticSegmentation.cols; j++)
        {
            
            if (grnd_ptr[j] == 0)
            {
                nonSeaCount += 1;
                if (sem_ptr[j] == 0)
                {
                    nonSeaAcc += 1;
                    totAcc += 1;
                }
            }
            else
            {
                seaCount += 1;
                if (sem_ptr[j] == 255)
                {
                    seaAcc += 1;
                    totAcc += 1;
                }
            }
        }
    }
    totalAccuracy = totAcc / (semanticSegmentation.rows * semanticSegmentation.cols);
    if (seaCount == 0)
    {
        seaPixelAccuracy = 0;
        nonSeaPixelAccuracy = nonSeaAcc / nonSeaCount;
    }
    else if (nonSeaCount == 0)
    {
        nonSeaPixelAccuracy = 0;
        seaPixelAccuracy = seaAcc / seaCount;
    }
    else
    {
        seaPixelAccuracy = seaAcc / seaCount;
        nonSeaPixelAccuracy = nonSeaAcc / nonSeaCount;
    }
}