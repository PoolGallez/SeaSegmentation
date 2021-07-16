#include<iostream>;
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
class BagOfWords{
	public:
		void computeHistograms(cv::Mat & histograms, cv::Mat & labels, std::vector<cv::Mat> positives, std::vector<cv::Mat> negatives);
		void getImageHistogram(cv::Mat image, std::vector<cv::KeyPoint keypoints>, cv::Mat & bowDescriptor);
		void cluster(int clusterNumber, cv::TermCriteria termCriteria, Mat totalDescriptors);
		void trainSVM(cv::Ptr<cv::ml::SVM> & svm, cv::TrainData trainData);
		void loadDictionary(char * path, cv::Mat & bowDescriptorExtractor);
		void loadSVM(char * path, cv::Ptr<cv::ml::SVM> & svm);

	private: 
		FeatureExtractor extractor;
		cv::Mat bagOfWords;
		BOWImgDescriptorExtractor bowDE;
		
