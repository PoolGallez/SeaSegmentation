#include <FeatureExtractor.h>
FeatureExtractor::FeatureExtractor(){
    extractor = cv::SIFT::create();
    matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_L2);
}
FeatureExtractor::FeatureExtractor(cv::Ptr<cv::DescriptorExtractor> extractor, cv::Ptr<cv::DescriptorMatcher> matcher)
{
    this->extractor = extractor;
    this->matcher = matcher;
}
cv::Ptr<cv::DescriptorExtractor> FeatureExtractor::getExtractor()
{
    return extractor;
}

cv::Ptr<cv::DescriptorMatcher> FeatureExtractor::getMatcher()
{
    return matcher;
}

void FeatureExtractor::descriptorExtraction(std::vector<cv::Mat> positiveImages, std::vector<cv::Mat> negativeImages, std::vector<cv::Mat> &positiveDescriptors, std::vector<cv::Mat> &negativeDescriptors, cv::Mat &totalDescriptors, bool split)
{
    std::cout << "*** Feature Extraction" << std::endl;
    int minHessian = 100;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    int i = 0;
    std::cout << "* Extraction from water images" << std::endl;
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
            std::cout << "Index " << i << " has an empty descriptor water, removing the image" << std::endl;
            positiveImages.erase(positiveImages.begin() + i);
        }
        i++;
    }
    i = 0;
    std::cout << "* Extraction from non water images" << std::endl;
    for (cv::Mat image : negativeImages)
    {
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
            std::cout << "Index " << i << " has an empty non water descriptor, removing the image" << std::endl;
            negativeDescriptors.erase(negativeDescriptors.begin() + i);
        }
        i++;
    }
    std::cout << "*** Feature Extracted" << std::endl;
}