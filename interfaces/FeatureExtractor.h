#include<iostream>;
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
class FeatureExtractor{
	public: 
		FeatureExtractor(cv::Ptr<cv::DescriptorExtractor> extractor, cv::Ptr<cv::DescriptorMatcher> matcher);
		void descriptorExtraction(std::vector<cv::Mat> positiveImages, std::vector<cv::Mat> negativeImages, std::vector<cv::Mat> positiveDescriptors, std::vector<cv::Mat> negativeDescriptors, cv::Mat totalDescriptors, bool split );
		cv::Ptr<cv::DescriptorExtractor> getExtractor();
		cv::Ptr<cv::DescriptorMatcher> getMatcher();
	private:
		cv::Ptr<cv::DescriptorExtractor> extractor;
		cv::Ptr<cv::DescriptorMatcher> matcher;
}
