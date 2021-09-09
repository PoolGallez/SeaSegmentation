#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

/**
 * Utility class for loading the images
 * @author Paolo Galletta
 * @version 1.0
 **/
class Loader
{
public:
	Loader(std::string positiveRegEx, std::string negativeRegEx, std::string pattern);
	void loadImages(std::vector<cv::Mat> &posImages, std::vector<cv::Mat> &negImages);

private:
	std::vector<cv::String> positives;
	std::vector<cv::String> negatives;
};
