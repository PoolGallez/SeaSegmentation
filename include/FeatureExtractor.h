#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

/**
 * Feature Extractor utility class
 * @author Paolo Galletta
 * @version 1.0
 **/
class FeatureExtractor{
	public: 

		/**
		 * Constructors
		 **/
		FeatureExtractor();
		FeatureExtractor(cv::Ptr<cv::DescriptorExtractor> extractor, cv::Ptr<cv::DescriptorMatcher> matcher);

		/**
		 * Extracts the descriptors for the training images
		 **/
		void descriptorExtraction(std::vector<cv::Mat> positiveImages, std::vector<cv::Mat> negativeImages, std::vector<cv::Mat> & positiveDescriptors, std::vector<cv::Mat> & negativeDescriptors, cv::Mat & totalDescriptors, bool split );


		/**
		 * Getters
		 **/
		cv::Ptr<cv::DescriptorExtractor> getExtractor();
		cv::Ptr<cv::DescriptorMatcher> getMatcher();
	private:
		cv::Ptr<cv::DescriptorExtractor> extractor;
		cv::Ptr<cv::DescriptorMatcher> matcher;
};
