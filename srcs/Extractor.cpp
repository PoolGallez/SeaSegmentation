#include <FeatureExtractor.h>

/**
 * Default Constructor
 **/
FeatureExtractor::FeatureExtractor(){
    extractor = cv::SIFT::create();
    matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_L2);
}

/**
 * Default Constructor
 * @param extractor the feature extractor to be used
 * @param matcher the descriptormatcher to be used
 **/
FeatureExtractor::FeatureExtractor(cv::Ptr<cv::DescriptorExtractor> extractor, cv::Ptr<cv::DescriptorMatcher> matcher)
{
    this->extractor = extractor;
    this->matcher = matcher;
}

/**
 * Extractor getter
 * @return the feature extractor used
 **/
cv::Ptr<cv::DescriptorExtractor> FeatureExtractor::getExtractor()
{
    return extractor;
}

/**
 * Matcher getter
 * @return the descriptor matcher used
 **/
cv::Ptr<cv::DescriptorMatcher> FeatureExtractor::getMatcher()
{
    return matcher;
}

/**
 * Extract the descriptors for the training set
 * @param positiveImages the images related to label +1
 * @param negativeImages the images related to label -1
 * @param positiveDescriptors the output vector containing the descriptors related to label +1
 * @param negativeDescriptors the output vector containing the descriptors related to label -1
 * @param totalDescriptors the output vector containing both positive and negative descriptors
 * @param split used to reduce the dataset
 **/
void FeatureExtractor::descriptorExtraction(std::vector<cv::Mat> positiveImages, std::vector<cv::Mat> negativeImages, std::vector<cv::Mat> &positiveDescriptors, std::vector<cv::Mat> &negativeDescriptors, cv::Mat &totalDescriptors, bool split)
{
    cv::Ptr<cv::DescriptorExtractor> detector = getExtractor();
    int i = 0;
    for (cv::Mat image : positiveImages)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detect(image, keypoints);
        detector->compute(image, keypoints, descriptors);
        if (!descriptors.empty())
        {
            positiveDescriptors.push_back(descriptors);
            totalDescriptors.push_back(descriptors);
        }
        else
        {
            positiveImages.erase(positiveImages.begin() + i);
        }
        i++;
    }
    i = 0;
    for (cv::Mat image : negativeImages)
    {
        std::cout << i << std::endl;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detect(image, keypoints);
        detector->compute(image, keypoints, descriptors);
        if (!descriptors.empty())
        {
            negativeDescriptors.push_back(descriptors);
            totalDescriptors.push_back(descriptors);
        }
        else
        {
            negativeImages.erase(negativeImages.begin() + i);
        }
        i++;
    }
}
