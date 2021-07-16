#include<iostream>;
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

class Loader{
	public:
		std::vector<cv::String> positives;
		std::vector<cv::String> negatives;
		Loader(std::string positiveRegEx, std::string negativeRegEx);
		void loadImages(std::vector<cv::Mat> & posImages, std::vector<cv::Mat> & negImages);

}
