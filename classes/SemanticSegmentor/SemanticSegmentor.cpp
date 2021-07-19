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
~SemanticSegmentor::SemanticSegmentor(){
    delete this->image;
    delete this->graphSegmented;
    delete [] this->masks;
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
    cv::Mat output = cv::zeros(image.size());
    bow.getExtractor();
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