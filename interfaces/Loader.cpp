#include<Loader.h>
Loader::Loader(std::string positiveRegEx, std::string negativeRegEx){
	cv::util::fs::glob(positiveRegEx,positives);
	cv::util::fs::glob(negativeRegEx,negatives);
}


void Loader::loadImages(std::vector<cv::Mat> & posImages, std::vector<cv::Mat> & negImages){
	for (String name : positives){
		posImages.push_back(imread(name, IMREAD_GRAYSCALE));
	}
	for (String name : negatives){
		negImages.push_back(imread(name, IMREAD_GRAYSCALE));
	}
}
