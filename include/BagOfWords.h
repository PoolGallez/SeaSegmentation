#pragma once
#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ml/ml.hpp>
#include <FeatureExtractor.h>

/**
 * Utility class for the Bag of Words implementation
 * @author Paolo Galletta
 * @version 1.0
 **/
class BagOfWords{
	public:

		/**
		 * Constuctors
		 **/
		BagOfWords();
		BagOfWords(FeatureExtractor extractor);
		BagOfWords(FeatureExtractor extractor, cv::Ptr<cv::ml::SVM> svm);

		/**
		 * Histogram Computation for the training set
		 **/
		void computeHistograms(cv::Mat & histograms, cv::Mat & labels, std::vector<cv::Mat> positives, std::vector<cv::Mat> negatives);

	
		/**
		 * Cluster to obtain the codebook
		 **/
		void cluster(int clusterNumber, cv::TermCriteria termCriteria, cv::Mat totalDescriptors);

		/**
		 * SVM training
		 **/
		float trainSVM(cv::Ptr<cv::ml::TrainData >trainData);

		/**
		 * Savers
		 **/
		bool saveDictionary(std::string path);
		bool saveSVM(std::string path);

		/**
		 * Loaders
		 **/
		bool loadDictionary(std::string path);
		bool loadSVM(std::string path);

		/** 
		 * Predictor
		 **/
		int predict(cv::Mat image,cv::Mat mask);

		/**
		 * Getters
		 **/
		FeatureExtractor getExtractor();
		cv::BOWImgDescriptorExtractor getBowDescriptorExtractor();
		cv::Ptr<cv::ml::SVM> getSVM();
		void getImageHistogram(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat & bowDescriptor);


		/**
		 * Constants
		 **/
		static const int  POSITIVE_CLASS = 1;
		static const int  NEGATIVE_CLASS = -1;

	private: 
		FeatureExtractor extractor;
		cv::BOWImgDescriptorExtractor bowDE;
		cv::Ptr<cv::ml::SVM> svm;
};		
		
