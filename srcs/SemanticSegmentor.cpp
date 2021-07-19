#include "SemanticSegmentor.h"

SemanticSegmentor::SemanticSegmentor(cv::Mat img, cv::Mat graphSegmented)
{
    image = img;
    this->graphSegmented = graphSegmented;
}
SemanticSegmentor::SemanticSegmentor(cv::Mat img)
{
    image = img;
}
SemanticSegmentor::SemanticSegmentor(cv::Mat img, cv::Mat graphSegmented, std::vector<cv::Mat> masks)
{
    image = img;
    this->graphSegmented = graphSegmented;
    this->masks = masks;
}
SemanticSegmentor::~SemanticSegmentor()
{
    image.release();
    graphSegmented.release();
    std::vector<cv::Mat>().swap(masks);
}

void SemanticSegmentor::setMasks(std::vector<cv::Mat> masks)
{
    this->masks = masks;
}
void SemanticSegmentor::setGraphSegmented(cv::Mat graphSegmented)
{
    this->graphSegmented = graphSegmented;
}

cv::Mat SemanticSegmentor::getGraphSegmented()
{
    return graphSegmented;
}

cv::Mat SemanticSegmentor::getImage()
{
    return image;
}
std::vector<cv::Mat> SemanticSegmentor::getMasks()
{
    return masks;
}

cv::Mat SemanticSegmentor::getMask(int i)
{
    return masks[i];
}

cv::Mat SemanticSegmentor::getSemanticSegmented(BagOfWords bow)
{
    std::cout << "Semantic Segmentation ..." << std::endl;
    cv::Mat output = image.clone();
    for (cv::Mat mask : masks)
    {
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

void SemanticSegmentor::getPixelAccuracy(cv::Mat semanticSegmentation, cv::Mat groundTruth, double &seaPixelAccuracy, double &nonSeaPixelAccuracy)
{
    double seaAcc = 0, nonSeaAcc = 0, seaCount = 0, nonSeaCount = 0;
    for (int i = 0; i < semanticSegmentation.rows; i++)
    {
        uchar *sem_ptr = semanticSegmentation.ptr<uchar>(i);
        uchar *grnd_ptr = groundTruth.ptr<uchar>(i);
        for (int j = 0; j < semanticSegmentation.cols; j++)
        {
            // row_ptr[j] will give you access to the pixel value
            // any sort of computation/transformation is to be performed here
            if (grnd_ptr[j] == 0)
            {
                nonSeaCount += 1;
                if (sem_ptr[j] == 0)
                {
                    nonSeaAcc += 1;
                }
            }
            else
            {
                seaCount += 1;
                if (sem_ptr[j] == 255)
                {
                    seaAcc += 1;
                }
            }
        }
    }
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
    std::cout << seaAcc << std::endl;
    std::cout << nonSeaAcc << std::endl;
    std::cout << nonSeaCount << std::endl;
}