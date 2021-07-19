#include<Loader.h>
Loader::Loader(std::string positiveRegEx, std::string negativeRegEx, std::string pattern){
	std::vector<cv::String> tmpPos;
	std::vector<cv::String> tmpNeg;
	cv::utils::fs::glob(positiveRegEx,pattern,positives);
	cv::utils::fs::glob(negativeRegEx,pattern,negatives);
}


void Loader::loadImages(std::vector<cv::Mat> & posImages, std::vector<cv::Mat> & negImages){
	for (cv::String name : positives){
		posImages.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
	}
	for (cv::String name : negatives){
		negImages.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
	}
}
