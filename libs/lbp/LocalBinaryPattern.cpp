#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "lbp.hpp"
#include "histogram.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char *argv[]) {

	// initial values
    int radius = 1;
    int neighbors = 8;

    // windows
    namedWindow("original",WINDOW_AUTOSIZE);
    namedWindow("lbp",WINDOW_AUTOSIZE);

    // matrices used
    Mat frame = imread("../BoW_Segmentation/training/non_water/l1004.png"); // always references the last frame
    Mat dst; // image after preprocessing
    Mat lbp; // lbp image

    // just to switch between possible lbp operators
    vector<string> lbp_names;
    lbp_names.push_back("Extended LBP"); // 0
    lbp_names.push_back("Fixed Sampling LBP"); // 1
    lbp_names.push_back("Variance-based LBP"); // 2
    int lbp_operator=0;

	cvtColor(frame, dst, COLOR_BGR2GRAY);
	GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
	// comment the following lines for original size
	resize(frame, frame, Size(), 0.5, 0.5);
	resize(dst,dst,Size(), 0.5, 0.5);
	//
	switch(lbp_operator) {
	case 0:
		lbp::ELBP(dst, lbp, radius, neighbors); // use the extended operator
		break;
	case 1:
		lbp::OLBP(dst, lbp); // use the original operator
		break;
	case 2:
		lbp::VARLBP(dst, lbp, radius, neighbors);
		break;
	}
	// now to show the patterns a normalization is necessary
	// a simple min-max norm will do the job...
	normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);

	imshow("original", frame);
	imshow("lbp", lbp);

	char key = (char) waitKey(0);

	// exit on escape
	if(key == 27)
		return 0;

	// to make it a bit interactive, you can increase and decrease the parameters
	switch(key) {
	case 'q': case 'Q':
		return 0;
	// lower case r decreases the radius (min 1)
	case 'r':
		radius-=1;
		radius = std::max(radius,1);
		cout << "radius=" << radius << endl;
		return 0;
	// upper case r increases the radius (there's no real upper bound)
	case 'R':
		radius+=1;
		radius = std::min(radius,32);
		cout << "radius=" << radius << endl;
		return 0;
	// lower case p decreases the number of sampling points (min 1)
	case 'p':
		neighbors-=1;
		neighbors = std::max(neighbors,1);
		cout << "sampling points=" << neighbors << endl;
		return 0;
	// upper case p increases the number of sampling points (max 31)
	case 'P':
		neighbors+=1;
		neighbors = std::min(neighbors,31);
		cout << "sampling points=" << neighbors << endl;
		return 0;
	// switch between operators
	case 'o': case 'O':
		lbp_operator = (lbp_operator + 1) % 3;
		cout << "Switched to operator " << lbp_names[lbp_operator] << endl;
		return 0;
	case 's': case 'S':
		imwrite("original.jpg", frame);
		imwrite("lbp.jpg", lbp);
		cout << "Screenshot (operator=" << lbp_names[lbp_operator] << ",radius=" << radius <<",points=" << neighbors << ")" << endl;
		return 0;
	default:
		return 0;
	}

    	return 0; // success
}
