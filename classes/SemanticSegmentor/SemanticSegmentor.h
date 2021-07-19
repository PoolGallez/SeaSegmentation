#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include<BagOfWords.h>
class SemanticSegmentor
{
public:
	SemanticSegmentor(cv::Mat img,cv::Mat graphSegmented);
	SemanticSegmentor(cv::Mat img);
	SemanticSegmentor(cv::Mat img, cv::Mat graphSegmented, std::vector<cv::Mat> masks);
	cv::Mat getSemanticSegmented(BagOfWords bow);
	cv::Mat getImage();
	cv::Mat getGraphSegmented();
	std::vector<cv::Mat> getMasks();
	cv::Mat getMask(int i);
	std::vector<cv::Mat> computeMasks();
	void setMasks(std::vector<cv::Mat> masks);
	void setGraphSegmented(cv::Mat graphSegmented );

private:
	cv::Mat image;
	cv::Mat graphSegmented;
	std::vector<cv::Mat> masks;
};
