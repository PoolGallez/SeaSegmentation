#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ml/ml.hpp>
#include <../FeatureExtractor/FeatureExtractor.h>
class BagOfWords{
	public:
		BagOfWords(FeatureExtractor extractor, cv::TermCriteria termCriteria);
		BagOfWords(FeatureExtractor extractor, cv::Ptr<cv::ml::SVM> svm);
		void computeHistograms(cv::Mat & histograms, cv::Mat & labels, std::vector<cv::Mat> positives, std::vector<cv::Mat> negatives);
		void getImageHistogram(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat & bowDescriptor);
		void cluster(int clusterNumber, cv::TermCriteria termCriteria, cv::Mat totalDescriptors);
		float trainSVM(cv::Ptr<cv::ml::TrainData >trainData);
		bool saveDictionary(std::string path);
		bool saveSVM(std::string path);
		bool loadDictionary(std::string path);
		bool loadSVM(std::string path);
		FeatureExtractor getExtractor();
		cv::BOWImgDescriptorExtractor getBowDescriptorExtractor();
		cv::Ptr<cv::ml::SVM> getSVM();



		const int  WATER_CLASS = 1;
		const int  NON_WATER_CLASS = -1;

	private: 
		FeatureExtractor extractor;
		cv::BOWImgDescriptorExtractor bowDE;
		cv::Ptr<cv::ml::SVM> svm;
};		
		
