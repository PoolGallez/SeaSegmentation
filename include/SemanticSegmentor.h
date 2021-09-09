#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <BagOfWords.h>
/**
 * Class for Semantic Segmentation
 * @author Paolo Galletta
 * @version 1.0
 **/

class SemanticSegmentor
{
public:
	/**
	 * Constructors
	 **/
	SemanticSegmentor(cv::Mat img, cv::Mat graphSegmented);
	SemanticSegmentor(cv::Mat img);
	SemanticSegmentor(cv::Mat img, cv::Mat graphSegmented, std::vector<cv::Mat> masks);

	/**
	 * Destructor
	 **/
	~SemanticSegmentor();

	/** 
	 * Getters
	 **/
	cv::Mat getSemanticSegmented(BagOfWords bow);
	cv::Mat getImage();
	cv::Mat getGraphSegmented();
	std::vector<cv::Mat> getMasks();
	cv::Mat getMask(int i);
	static void getPixelAccuracy(cv::Mat segmenticSegmentation, cv::Mat groundTruth, double &seaPixelAccuracy, double &nonSeaPixelAccuracy, double &totalAccuracy);

	/**
	 * Setters
	 **/
	void setMasks(std::vector<cv::Mat> masks);
	void setGraphSegmented(cv::Mat graphSegmented);
	
	/**
	 * Best constants for the graph segmentor (Empirically found)
	 **/
	const static int SIGMA_DEF = 2;
	const static int K_DEF = 600;
	const static int MINSIZE_DEF = 600;

private:

	/**
	 * Class attributes
	 **/
	cv::Mat image;
	cv::Mat graphSegmented;
	std::vector<cv::Mat> masks;
};
